# BioDSH 增量压缩插件 · 开发·编译·部署指南

> **目标读者**：App 端开发 agent。读完本文档即可自主完成「增量压缩插件」的开发、打包、挂载、部署、验证，全程不需要额外上下文。
>
> **要解决的问题**：LLM 用量日报（2026-11-10）显示整体缓存命中率仅 **24.3%**，且 **checkpoint 次数与命中率负相关**（0 次→49.2%、30 次→27.3%、60 次→20%）。根因是 dsh 默认压缩机制每次 checkpoint 都「整段重写历史」，把稳定前缀打碎，导致 DeepSeek 前缀缓存失效。
>
> **目标产物**：一个 dsh 插件（`CompactionEngine` 子类），把默认的「滚动全量重摘要」换成「冻结前缀 + 只压新增段 + 稀有合并」，命中率从 24.3% 拉到 60%+。

---

## 1. 根因（代码级定位）

dsh 的对话压缩由 `@deepseek-ai/dsh-compaction-basic` 包实现。它的 `selectCompactableRange` 里，压缩范围起点被硬编码：

```js
const firstIdx = systemHead(session, surfaceNodes[0]) === void 0 ? 0 : 1;  // 永远从最开头
return { start: surfaceNodes[firstIdx], end: surfaceNodes[keepFromIdx - 1] };
```

所以每次 checkpoint 都把「从对话最开头到最近尾部之前」的**全部历史（含上一次的摘要）**重新喂给 LLM 摘要成一条新消息，`replace` 掉整段。摘要文本是 LLM 非确定性输出 → 字节每次都变 → `[system][tools][旧摘要]` 前缀变成 `[system][tools][新摘要]` → 前缀缓存全部失效。

**一句话根因**：`startSeq` 用了 `surfaceNodes[0/1]`（最开头），而不是「上一条 checkpoint 之后」。改掉这一行就是本方案的核心。

---

## 2. 方案：三级压缩（归档→冻结→追加）

| 级别 | 触发 | 动作 | 性质 | 频率 |
|---|---|---|---|---|
| **Level-0 归档** | 存在「已完成的 tool exchange」 | 把 tool 结果**原样搬进 ArchiveStore**,surface 里 `replace` 成极短 ref 引用 | **无损**(按 ref 可精确读回) | 每轮 exchange 完成即归档 |
| **Level-1 增量** | 活跃尾部涨到阈值 | 只摘要「上一条 checkpoint 之后的新增段」，`append` 新块进冻结区，旧摘要字节不动 | 有损 | 高频 |
| **Level-2 合并** | 冻结块累积到阈值 | 把 N 个冻结 checkpoint 合并成一条 super-summary，`replace` 一次 | 有损 | 稀有（每 ~N 轮一次） |

- **Level-0 是新增的无损层**：tool 结果是上下文的膨胀大头，且是「数据」不是「语义」，可以原样搬走、按 ref 精确读回，零损耗；而对话语义才需要 L1/L2 的 LLM 摘要（必然有损）。
- 顺序必须是 **L0 → L2 → L1**：先把 tool 结果无损搬走，剩下的对话段再摘要，摘要成本大幅下降。
- Level-1 的 `append` 让前缀「变长」而不是「变」，旧冻结块字节不变 → 旧前缀照常命中，只有新增段 miss。
- Level-2 是「冻结块的自我合并」——因为冻结块线性累积（每块 ≤ maxTokens），迟早会撑爆窗口，必须偶尔合并一次，但分摊到 N 轮。

### 双阈值 + 不变量

```text
thresholdTokens  = floor(contextWindow × thresholdRatio)   // Level-1 触发线，建议 thresholdRatio=0.85
retainTokens     = 20000（绝对值，≈最近20条原文）           // 活跃尾部保留
frozenThreshold  = floor(contextWindow × frozenRatio)      // Level-2 触发线，建议 frozenRatio=0.5
frozenMaxBlocks  = 8                                       // Level-2 块数上限
```

**三条不变量**（违反则拒绝加载/运行期抛错）：

```text
① retainTokens < thresholdTokens                         （构造期：若用 retainRatio 形式；运行期：绝对值形式）
② frozenRatio  < thresholdRatio                          （构造期）
③ frozenMaxBlocks × maxTokens ≤ frozenThreshold          （运行期，用真实 contextWindow）
```

粗校验（构造期，窗口无关）：`frozenMaxBlocks / frozenRatio ≤ saneBlockRatio`（默认 128，可覆盖；`null` 关闭）。

---

## 3. 插件源码（准代码级，可直接落文件）

### 3.1 `desktop/dsh-plugins/compaction-incremental/package.json`

```json
{
  "name": "@biodsh/compaction-incremental",
  "version": "0.1.0",
  "type": "module",
  "main": "index.js"
}
```

### 3.2 `desktop/dsh-plugins/compaction-incremental/index.js`

```js
import {
  CompactionEngine,
  compactCheckpointSource,
  isCompactCheckpointSource,
  toolPairingBalancedBefore,
  CompactionId,
  ManualCompactionError,
} from "@deepseek-ai/dsh-compaction";
import { BlockAssembler, createUserMessage } from "@deepseek-ai/dsh-llm";
import { randomUUID } from "node:crypto";

export const name = "biodsh-compaction-incremental";

const COMPACTION_INSTRUCTION = [
  "You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.",
  "",
  "Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write \"(none)\" for an empty section — never drop a section.",
  "",
  "## Primary Request and Intent",
  "- [the user's original and evolving goals; quote verbatim where the exact wording matters]",
  "",
  "## Key Technical Concepts",
  "- [technologies, frameworks, patterns, and conventions in play]",
  "",
  "## Files and Code",
  "- [exact path: why it matters, key changes or snippets]",
  "",
  "## Errors and Fixes",
  "- [error: how it was resolved, plus any related user feedback]",
  "",
  "## Pending Jobs",
  "- [explicitly requested work not yet completed]",
  "",
  "## Current Work",
  "- [precisely what was in progress at this checkpoint]",
  "",
  "## Next Step",
  "- [the single next action, directly in line with the most recent request, or \"(none)\"]",
  "",
  "## Critical Context",
  "- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]",
  "",
  "Rules:",
  "- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.",
  "- Capture user feedback and explicit instructions faithfully, especially corrections.",
  "- Do NOT mention this summarization request or that the context was compacted.",
  "- Output only the checkpoint text: do not call any tool or take any other action.",
].join("\n");

// Level-2 专用：把 N 个历史 checkpoint 合并成一条（措辞与 Level-1 不同）
const MERGE_INSTRUCTION = [
  "You are now acting as a compaction engine. Below are N historical checkpoints from the same conversation. Merge them into a SINGLE consolidated checkpoint that preserves still-true facts, drops stale ones, and removes duplication.",
  "",
  "Output EXACTLY the Markdown structure below (same sections as the original checkpoints).",
  "",
  "## Primary Request and Intent",
  "## Key Technical Concepts",
  "## Files and Code",
  "## Errors and Fixes",
  "## Pending Jobs",
  "## Current Work",
  "## Next Step",
  "## Critical Context",
  "",
  "Rules:",
  "- Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.",
  "- Do NOT mention this summarization request or that the context was compacted.",
  "- Output only the checkpoint text.",
].join("\n");

const CHECKPOINT_PREAMBLE = "This is an automatically generated checkpoint condensing an earlier span of the conversation to free up context. Treat the captured context as established background and build on it without restating it. Continue the task directly from the messages that follow, without acknowledging this checkpoint.";

export class IncrementalCompactionEngine extends CompactionEngine {
  static inject = ["tokenMeter", "sessions", "llm"];

  static DEFAULTS = Object.freeze({
    thresholdRatio: 0.85,
    retainTokens: 20000,
    frozenRatio: 0.5,
    frozenMaxBlocks: 8,
    maxTokens: 8192,
    saneBlockRatio: 128,
    auto: true,
  });

  constructor(ctx, config = {}) {
    super(ctx, "compaction");
    const d = IncrementalCompactionEngine.DEFAULTS;
    this.thresholdRatio = config.thresholdRatio ?? d.thresholdRatio;
    this.retainTokens = config.retainTokens ?? d.retainTokens;
    this.retainRatio = config.retainRatio ?? undefined;
    this.frozenRatio = config.frozenRatio ?? d.frozenRatio;
    this.frozenMaxBlocks = config.frozenMaxBlocks ?? d.frozenMaxBlocks;
    this.maxTokens = config.maxTokens ?? d.maxTokens;
    this.saneBlockRatio = config.saneBlockRatio ?? d.saneBlockRatio;
    this.auto = config.auto ?? d.auto;
    this.validateStatic();
  }

  validateStatic() {
    const ratio = (n, v) => {
      if (typeof v !== "number" || !(v > 0) || !(v < 1)) throw new Error(`${n} must be a number in (0,1), got ${v}`);
    };
    ratio("thresholdRatio", this.thresholdRatio);
    ratio("frozenRatio", this.frozenRatio);
    if (this.frozenRatio >= this.thresholdRatio) throw new Error(`frozenRatio (${this.frozenRatio}) must be < thresholdRatio (${this.thresholdRatio})`);
    if (this.retainTokens !== undefined && this.retainRatio !== undefined) throw new Error("retainTokens and retainRatio are mutually exclusive");
    if (this.retainRatio !== undefined) {
      ratio("retainRatio", this.retainRatio);
      if (this.retainRatio >= this.thresholdRatio) throw new Error(`retainRatio (${this.retainRatio}) must be < thresholdRatio`);
    } else if (!Number.isInteger(this.retainTokens) || this.retainTokens <= 0) {
      throw new Error(`retainTokens must be a positive integer, got ${this.retainTokens}`);
    }
    if (!Number.isInteger(this.frozenMaxBlocks) || this.frozenMaxBlocks < 2) throw new Error(`frozenMaxBlocks must be an integer >= 2, got ${this.frozenMaxBlocks}`);
    if (!Number.isInteger(this.maxTokens) || this.maxTokens <= 0) throw new Error(`maxTokens must be a positive integer, got ${this.maxTokens}`);
    if (this.saneBlockRatio !== null) {
      if (!Number.isInteger(this.saneBlockRatio) || this.saneBlockRatio <= 0) throw new Error(`saneBlockRatio must be a positive integer or null, got ${this.saneBlockRatio}`);
      const required = this.frozenMaxBlocks / this.frozenRatio;
      if (required > this.saneBlockRatio) throw new Error(`frozenMaxBlocks/frozenRatio = ${required} implies a context window >= ${Math.round(required * this.maxTokens)} tokens, exceeding saneBlockRatio (${this.saneBlockRatio})`);
    }
  }

  async apply(ctx) {
    if (!this.auto) return;
    ctx.on("agent/pre-step", async ({ agent, signal }, next) => {
      const decision = await next();
      if (signal.aborted) return decision;
      try {
        const r = await this.compactIfNeeded(agent, signal);
        if (r) ctx.logger.info(`incremental-compaction (${r.level}): shadowed ${r.shadowedCount} nodes (${r.startSeq}-${r.endSeq})`);
      } catch (e) {
        ctx.logger.warn(`incremental compaction failed: ${e?.message ?? e}; continuing the turn`);
      }
      return decision;
    });
    ctx.on("agent/request-error", async ({ agent, failure, signal }, next) => {
      if (failure.code !== "context_window_exceeded" || signal.aborted) return next();
      try {
        const r = await this.compactIfNeeded(agent, signal, { overflow: true });
        if (r) return { kind: "retry" };
      } catch {}
      return next();
    });
  }

  async compactIfNeeded(agent, signal, { overflow = false } = {}) {
    const session = agent.session;
    let m = this.ctx.tokenMeter.measure(session);
    const spec = await this.resolveSpec(session, signal);
    const checkpoints = this.checkpointNodes(session);
    const frozenTokens = this.sumTokens(session, checkpoints.map((c) => c.seq));

    // ---- Level-2：冻结块自我合并（稀有，先做）----
    if (!overflow && (checkpoints.length >= spec.frozenMaxBlocks || frozenTokens >= spec.frozenThreshold)) {
      if (checkpoints.length < 2) return null;
      const r = await this.mergeFrozen(session, checkpoints, agent, signal);
      return { level: "merge", ...r };
    }

    // ---- Level-1：只压活跃尾部 ----
    if (!overflow && m.totalTokens < spec.thresholdTokens) return null;
    const prune = this.ctx.get("toolResultPruner");
    if (prune) { prune.pruneSession(session); m = this.ctx.tokenMeter.measure(session); }
    if (!overflow && m.totalTokens < spec.thresholdTokens) return null;

    const last = this.lastCheckpointSeq(session);
    const startSeq = last === null ? this.firstNonSystemSeq(session) : last + 1;
    const endSeq = this.resolveTailStart(session, overflow ? 0 : spec.retainTokens);
    if (startSeq >= endSeq) return null;

    const r = await this.compactRegion(session, startSeq, endSeq, agent, signal);
    return { level: "incremental", ...r };
  }

  async compactRegion(session, startSeq, endSeq, agent, signal) {
    const summary = await this.summarizeSpan(session, startSeq, endSeq, agent, signal);
    this.commitReplacement(session, startSeq, endSeq, summary);
    return { startSeq, endSeq, shadowedCount: endSeq - startSeq + 1 };
  }

  async mergeFrozen(session, checkpoints, agent, signal) {
    const startSeq = checkpoints[0].seq;
    const endSeq = checkpoints[checkpoints.length - 1].seq;
    const summary = await this.summarizeCheckpoints(session, checkpoints, agent, signal);
    this.commitReplacement(session, startSeq, endSeq, summary);
    return { startSeq, endSeq, shadowedCount: checkpoints.length };
  }

  commitReplacement(session, startSeq, endSeq, summary) {
    const compactionId = CompactionId(randomUUID());
    const lifecycle = { compactionId, turn: "current-turn" };
    const startEv = session.append("compaction/start", lifecycle);
    const summaryEv = session.append("compaction/summary", {
      compactionId,
      summary,
      shadowedRange: { start: startSeq, end: endSeq },
      shadowedSeqs: this.surfaceSeqRange(session, startSeq, endSeq),
    });
    session.append("user/message", this.frameSummary(summary), {
      surfaceOp: { op: "replace", startSeq, endSeq },
      source: compactCheckpointSource(compactionId),
      sourceEventSeqs: [startEv.seq, summaryEv.seq],
    });
    session.append("compaction/end", lifecycle);
  }

  async summarizeSpan(session, startSeq, endSeq, agent, signal) {
    const shadowedSeqs = this.surfaceSeqRange(session, startSeq, endSeq);
    const input = buildSummarizationInput(session, shadowedSeqs);
    return this.summarizeWithLlm(input, agent, signal, COMPACTION_INSTRUCTION);
  }

  async summarizeCheckpoints(session, checkpoints, agent, signal) {
    const head = systemHead(session, session.surface.nodes[0]);
    const system = head === undefined ? null : session.deriveEventMessage(head);
    const cpMessages = checkpoints.map((c) => session.deriveEventMessage(c.ev)).filter((m) => m !== null);
    return this.summarizeWithLlm(
      { messages: system === null ? cpMessages : [system, ...cpMessages] },
      agent, signal, MERGE_INSTRUCTION,
    );
  }

  async summarizeWithLlm(input, agent, signal, instruction) {
    const target = routedTarget(agent.session) ?? conversationTarget(agent);
    if (target === undefined) throw new ManualCompactionError("config", "no summarization target");
    const assembler = new BlockAssembler();
    const messages = [...input.messages, createUserMessage({
      content: [{ type: "text", text: instruction }],
      source: { kind: "plugin", plugin: "biodsh-compaction-incremental" },
    })];
    const options = {
      provider: target.provider,
      model: target.model,
      messages,
      ...(input.tools === undefined ? {} : { tools: [...input.tools] }),
      maxTokens: this.maxTokens,
      sessionId: agent.session.id,
      purpose: "compaction",
      ...(signal === undefined ? {} : { signal }),
    };
    for await (const chunk of this.ctx.llm.stream(options)) assembler.push(chunk);
    if (assembler.finish.kind !== "done") throw new ManualCompactionError("summary", `summarization finished as ${assembler.finish.kind}`);
    const text = assembler.blocks().filter((b) => b.type === "text").map((b) => b.text).join("\n").trim();
    if (!text) throw new ManualCompactionError("summary", "summarization produced no text");
    return text;
  }

  async resolveSpec(session, signal) {
    const header = session.requestHeader()?.config;
    if (header === undefined || header.provider.length === 0 || header.model.length === 0) throw new ManualCompactionError("config", "no routed target");
    const info = await this.ctx.llm.resolveModelInfo(header.provider, header.model, signal);
    const contextWindow = info.context?.contextWindow;
    if (contextWindow === undefined || contextWindow <= 0) throw new ManualCompactionError("config", `no contextWindow for ${header.provider}/${header.model}`);
    const thresholdTokens = Math.floor(contextWindow * this.thresholdRatio);
    const retainTokens = this.retainTokens ?? Math.floor(contextWindow * this.retainRatio);
    const frozenThreshold = Math.floor(contextWindow * this.frozenRatio);
    if (retainTokens >= thresholdTokens) throw new ManualCompactionError("config", `retainTokens (${retainTokens}) must be < thresholdTokens (${thresholdTokens})`);
    if (this.frozenMaxBlocks * this.maxTokens > frozenThreshold) throw new ManualCompactionError("config", `frozenMaxBlocks × maxTokens (${this.frozenMaxBlocks * this.maxTokens}) exceeds frozenThreshold (${frozenThreshold})`);
    return { contextWindow, thresholdTokens, retainTokens, frozenThreshold, frozenMaxBlocks: this.frozenMaxBlocks };
  }

  checkpointNodes(session) {
    const out = [];
    for (const seq of session.surface.nodes) {
      const ev = session.eventAt(seq);
      if (ev?.type === "user/message" && isCompactCheckpointSource(ev.source)) out.push({ seq, ev });
    }
    return out;
  }

  lastCheckpointSeq(session) {
    const cps = this.checkpointNodes(session);
    return cps.length ? cps[cps.length - 1].seq : null;
  }

  firstNonSystemSeq(session) {
    const nodes = session.surface.nodes;
    const head = session.eventAt(nodes[0]);
    return head?.type === "system/message" ? nodes[1] : nodes[0];
  }

  resolveTailStart(session, retainTokens) {
    const nodes = session.surface.nodes;
    const m = this.ctx.tokenMeter.measure(session);
    let acc = 0, idx = nodes.length;
    for (let i = nodes.length - 1; i >= 0; i--) {
      acc += m.nodes[i]?.tokens ?? 0;
      idx = i;
      if (acc >= retainTokens) break;
    }
    while (idx < nodes.length && !toolPairingBalancedBefore(session, nodes[idx])) idx += 1;
    return nodes[idx];
  }

  surfaceSeqRange(session, startSeq, endSeq) {
    const nodes = session.surface.nodes;
    return nodes.slice(nodes.indexOf(startSeq), nodes.indexOf(endSeq) + 1);
  }

  sumTokens(session, seqs) {
    const m = this.ctx.tokenMeter.measure(session);
    return seqs.reduce((n, seq) => n + (m.nodes.find((x) => x.seq === seq)?.tokens ?? 0), 0);
  }

  frameSummary(summary) {
    return { role: "user", content: `${CHECKPOINT_PREAMBLE}\n\n<compacted-summary>\n${summary}\n</compacted-summary>` };
  }
}

// ---- 无状态辅助函数（从 dsh-compaction-basic 移植，未导出）----
function systemHead(session, headSeq) {
  const head = session.eventAt(headSeq);
  return head?.type === "system/message" ? head : undefined;
}

function buildSummarizationInput(session, shadowedSeqs) {
  const header = session.requestHeader();
  const head = systemHead(session, session.surface.nodes[0]);
  const system = head === undefined ? null : session.deriveEventMessage(head);
  const regionMessages = shadowedSeqs.map((seq) => session.deriveEventMessage(session.eventAt(seq))).filter((m) => m !== null);
  return {
    ...(header?.tools === undefined ? {} : { tools: header.tools }),
    messages: system === null ? regionMessages : [system, ...regionMessages],
  };
}

function routedTarget(session) {
  const c = session.requestHeader()?.config;
  if (c === undefined || c.provider.length === 0 || c.model.length === 0) return undefined;
  return { provider: c.provider, model: c.model };
}

function conversationTarget(agent) {
  const r = routedTarget(agent.session);
  if (r !== undefined) return r;
  return (agent.options.provider?.length && agent.options.model?.length)
    ? { provider: agent.options.provider, model: agent.options.model }
    : undefined;
}

export default IncrementalCompactionEngine;
```

### 3.3 Level-0 无损归档 tool 结果（补充进 index.js）

> **实测依据（2026-11-11）**：命中率 49.7% 的客户端（带 `@omaax/local-memory 2.3.1`）走「归档已完成 tool exchange + 按 ref 读回」；命中率 26% 的客户端（无该插件）走「整段 condense 重摘要」。归档是**无损**的（原文搬走、按 ref 精确读回），优先级高于 L1 摘要——**L0 必须在 L1 之前跑**：先把 tool 结果（膨胀大头）无损搬走，再对剩下的对话语义做有损摘要。

**① ArchiveStore（新增文件，可复用 dsh storage 域或本地文件）**

```js
// desktop/dsh-plugins/compaction-incremental/archive-store.js
export class ArchiveStore {
  constructor(dir) { /* 本地 jsonl / 文件目录 */ }
  // 把一整段已完成的 exchange 原样存下来，返回可寻址 ref（如 seq 范围或 uuid）
  async put(exchange) { /* ... */ return ref; }
  // 按 ref 读回原文；offset/query 可选，用于只取需要的片段而非全量
  async get(ref, { offset, query } = {}) { /* ... */ return text; }
}
```

**② 归档 pass（compactIfNeeded 里，Level-1 之前）**

```js
// 只归档「已完成」的 tool exchange（assistant 的 tool_calls 已配对上 tool 结果）
async archiveCompletedExchanges(session, endSeq) {
  let cursor = this.firstNonSystemSeq(session);
  let archived = 0;
  while (cursor < endSeq) {
    const span = nextCompletedExchange(session, cursor);  // {startSeq,endSeq,exchange} | null
    if (!span) break;
    const ref = await this.archiveStore.put(span.exchange);  // 原文进 store
    session.append("user/message", {
      content: `[archived tool results: ${ref}]`,  // surface 里只留极短 ref
      surfaceOp: { op: "replace", startSeq: span.startSeq, endSeq: span.endSeq },
    });
    archived += 1;
    cursor = span.endSeq + 1;
  }
  return archived;
}
```

关键点：`replace` 的对象是「整段已完成 exchange」，替换成**字节稳定的 ref 占位串**；旧 exchange 段定稿后不再动 → 前缀稳定。

**③ 读取工具（让 agent 按需读，不把原文堆回上下文）**

```js
// 在插件 apply() 里注册工具
ctx.registerTool({
  name: "omaax_read_tool_result",
  description: "Read an archived tool result by ref (optional offset/query) to recover exact scripts/args/results.",
  parameters: {
    type: "object",
    properties: {
      ref: { type: "string" },
      offset: { type: "number" },
      query: { type: "string" },
    },
    required: ["ref"],
  },
  async handler({ ref, offset, query }) {
    return this.archiveStore.get(ref, { offset, query });
  },
});
```

**④ compactIfNeeded 调用顺序改为：L0 归档 → L2 合并 → L1 增量**

```js
async compactIfNeeded(agent, signal, { overflow = false } = {}) {
  const session = agent.session;
  // ---- Level-0：无损归档（必须在 L1 之前）----
  if (!overflow) {
    const tailStart = this.resolveTailStart(session, this.retainTokens);
    const n = await this.archiveCompletedExchanges(session, tailStart);
    if (n) this.ctx.logger.info(`archived ${n} tool exchanges`);
  }
  // ... 原有 Level-2 / Level-1 逻辑（见 3.2）
}
```

**⑤ 两条归档专用不变量**

1. 只归档「completed」exchange——进行中的 tool 往返不动，否则 `replace` 会打碎正在生成的 surface；
2. ref 占位串**字节稳定**——同一 exchange 的 ref 永不改，否则还是会把前缀改掉（回到老问题）。

---

## 4. 挂载（两处：Electron + Tauri）

工程当前是 **Electron 默认 + Tauri 半成品** 的双栈。生产（main 分支）走 Electron 的 `dsh.ts`；迁移目标（feature 分支）走 Tauri 的 `dsh.rs`。两处都要挂，且都要「禁用默认 compaction-basic + 插入新引擎」。

### 4.1 Tauri（`desktop/src-tauri/src/dsh.rs` 的 `ensure_home()`）

照抄现有 `step-guard` 的挂载方式（同函数内已有 `inserts` 列表）：

```rust
// ① 禁用默认滚动重摘要（和禁用 UI 插件同一处 body）
body.push("- id: compaction-basic\n  disabled: true".into());

// ② 插入新引擎（照抄 step_guard_plugin 那段，同目录 @biodsh/compaction-incremental）
let compaction_plugin = skill_router_plugin.parent().and_then(Path::parent)
    .map(|d| d.join("compaction-incremental").join("index.js"));
if let Some(p) = compaction_plugin.filter(|p| p.exists()) {
    inserts.push(format!(
        "    - id: biodsh-compaction-incremental\n      name: {}\n      config:\n        thresholdRatio: 0.85\n        retainTokens: 20000\n        frozenRatio: 0.5\n        frozenMaxBlocks: 8\n        saneBlockRatio: 128",
        yaml_str(&file_url(&p))
    ));
}

// ③ 把 cordis.patch.yml 的版本标记从 v8 升到 v9，强制下次启动重建
```

### 4.2 Electron（`desktop/src/main/dsh.ts` 的 `ensureHome()`）

`ensureHome()` 目前只生成「禁 UI + persona」，没有自定义插件插入逻辑，需补三处（import、插件路径函数、body 追加 + 版本标记）。

**① 新增 import（`createRequire` 之后）：**

```ts
import { pathToFileURL } from 'node:url';
```

**② 新增插件路径函数（`dshBin()` 之后，对齐 `paths.ts` 的 `resourcePath()` 分支）：**

```ts
// 找到增量压缩插件入口（dev 在 dsh-runtime，打包后由 electron-builder 映射到 resources/dsh）。
function compactionPluginPath(): string {
  if (app.isPackaged) return path.join(process.resourcesPath, 'dsh', 'node_modules', '@biodsh', 'compaction-incremental', 'index.js');
  return path.join(app.getAppPath(), 'dsh-runtime', 'node_modules', '@biodsh', 'compaction-incremental', 'index.js');
}
```

**③ `ensureHome()` 内 body 追加两条 + 版本标记 v4→v5：**

```ts
const personaYaml = PERSONA.split('\n').map((l) => (l ? `      ${l}` : '')).join('\n');
body.push(`- id: system-prompt\n  config:\n    persona: >-\n${personaYaml}`);
if (presetOk) body.push('- id: agent-presets\n  config:\n    default: biodsh');
// 增量压缩：禁用 dsh 默认的滚动全量重摘要，挂自定义两级压缩引擎
body.push('- id: compaction-basic\n  disabled: true');
body.push(`- id: biodsh-compaction-incremental\n  name: ${pathToFileURL(compactionPluginPath()).href}\n  config:\n    thresholdRatio: 0.85\n    retainTokens: 20000\n    frozenRatio: 0.5\n    frozenMaxBlocks: 8\n    saneBlockRatio: 128`);
writeFileSync(patch, `# biodsh-minimal v5 — BioDSH 精简模式 + BioDSH 人设 + 增量压缩（删除本文件并重启即可恢复 dsh 默认）\n${body.join('\n')}\n`);
```

> **正则不用改**：第 100 行的 `/# biodsh-minimal v[1234]/` 匹配旧版 v1–v4，现有用户的 v4 文件会被触发重建一次、写成 v5；新写的 v5 不匹配 `[1234]`，之后不再重建。
>
> **前置条件**：`compactionPluginPath()` 依赖 `dsh-runtime/node_modules/@biodsh/compaction-incremental/` 存在，需先完成第 5.4 节的 `stage-dsh.mjs` 拷贝。

---

## 5. 编译打包

### 5.1 目录布局

```text
源码:    desktop/dsh-plugins/compaction-incremental/{index.js, package.json}
         │  stage-dsh.mjs（cpSync 拷贝）
         ▼
staging: desktop/dsh-runtime/node_modules/@biodsh/compaction-incremental/   ← dev 读这里（app.getAppPath()）
         │  electron-builder extraResources（已配：dsh-runtime → resources/dsh）
         ▼
打包:    resources/dsh/node_modules/@biodsh/compaction-incremental/         ← packaged 读这里（process.resourcesPath）
         │  dsh.ts / dsh.rs 生成 file:// URL 写进 cordis.patch.yml
         ▼
运行时:  dsh 加载 → ctx.compaction = IncrementalCompactionEngine
```

### 5.2 构建命令

```bash
cd desktop
npm ci                              # 装依赖（dsh 0.1.x）
node scripts/stage-dsh.mjs           # stage dsh 运行时 + @biodsh 插件
node scripts/sync-skills.mjs         # 官方技能 → resources

# 确认插件已被 stage 到位（关键验证点）：
ls dsh-runtime/node_modules/@biodsh/compaction-incremental/index.js

# Tauri 打包（feature 分支）：
npm run tauri build
# Electron 打包（main 分支）：
npm run dist          # = npm run build && electron-builder
```

### 5.3 `stage-dsh.mjs` 精确 diff（两处）

**改动 1：import 加 `cpSync`**

```diff
- import { mkdirSync, writeFileSync, readFileSync, rmSync, existsSync, readdirSync, statSync } from 'node:fs';
+ import { mkdirSync, writeFileSync, readFileSync, rmSync, existsSync, readdirSync, statSync, cpSync } from 'node:fs';
```

**改动 2：`rmSync(.bin)` 之后、`console.log('ok')` 之前，加插件拷贝循环**

```diff
   rmSync(path.join(dir, 'node_modules', '.bin'), { recursive: true, force: true });
+
+  // BioDSH 的 cordis 插件：compaction-incremental = 增量压缩引擎。
+  // 拷进运行时，由 cordis.patch.yml（ensureHome 生成）以 @biodsh/compaction-incremental 加载。
+  for (const p of ['compaction-incremental']) {
+    const src = path.join(root, 'dsh-plugins', p);
+    if (existsSync(src)) {
+      const dst = path.join(dir, 'node_modules', '@biodsh', p);
+      rmSync(dst, { recursive: true, force: true });
+      cpSync(src, dst, { recursive: true });
+      console.log(`[stage-dsh] 已装 BioDSH 插件 @biodsh/${p}`);
+    }
+  }
   console.log('ok');
```

> `electron-builder.yml` **无需改**：第 12–13 行已有 `from: dsh-runtime/node_modules → to: dsh/node_modules`，`dsh-runtime/node_modules/@biodsh/compaction-incremental` 会自动打包到 `resources/dsh/node_modules/@biodsh/compaction-incremental`。

### 5.4 打包前必须验证的两件事

1. **插件被 stage 到位**：`ls dsh-runtime/node_modules/@biodsh/compaction-incremental/index.js` 存在。
2. **file URL 正确**：dsh 启动日志里出现引擎的 `incremental-compaction (incremental|merge): shadowed ...`。

---

## 6. 部署

1. 构建产出安装包（`.exe` / `.dmg` / `.deb` 或 Tauri 对应产物）。
2. 发布到现有分发渠道（与当前版本发布流程一致）。
3. 用户更新后首次启动，`ensureHome()`/`ensure_home()` 会因版本标记升级（v5/v9）自动重建 `cordis.patch.yml`，插件即刻生效。

> 回滚：删掉用户机器上 `$DSH_HOME/cordis.patch.yml` 并降级版本标记，或直接回退插件构建包，dsh 恢复到默认 `compaction-basic` 行为。

---

## 7. 验证（上线前必做）

| 指标 | 当前 | 目标 |
|---|---|---|
| 缓存命中率 `prompt_cache_hit_tokens/(hit+miss)` | 24.3% | ≥ 60% |
| 单请求 prompt | 均值 97.6k / 峰值 320k | ≤ 6 万 token |
| checkpoint 次数 vs 命中率 | **负相关**（0→49.2%、60→20%） | **正相关**（有 checkpoint 的会话命中 ≥ 无 checkpoint 的） |

**验证步骤**：
1. 本地起 dsh，跑一个超长多轮 agent 任务，触发多次 Level-1。
2. 观察日志 `incremental-compaction (incremental|merge): shadowed ...`，确认 Level-1 频繁、Level-2 稀有（约 7:1）。
3. 用日报脚本 `tooling/llm_daily_report.py` 复核次日数据，看命中率与 checkpoint 相关性是否翻正。
4. 调优：命中上去了但出现大量 `context_window_exceeded` → 把 `thresholdRatio` 再往上抬；冻结块合并太频繁 → 把 `frozenRatio`/`frozenMaxBlocks` 调大。

---

## 8. 待验证项（开发 agent 落地前必须逐条确认）

> 本指南的插件代码基于 dsh **0.1.5-rc.1** 的 API（feature/Tauri 分支版本）。若目标是 **0.1.1-rc.2**（main/Electron 分支），已确认 `CompactionEngine`/`isCompactCheckpointSource`/`toolPairingBalancedBefore` 同签名存在，但以下方法需在目标版本的 node_modules 里逐一核对：

1. `session.deriveEventMessage(event)` —— 是否存在、返回 message 结构。
2. `session.requestHeader()` 返回 `{ config:{provider,model}, tools }` 结构。
3. `ctx.llm.resolveModelInfo(provider, model, signal)` 返回 `{ context: { contextWindow } }`。
4. `ctx.llm.stream(options)` 的 `GenerateOptions` 字段（`sessionId`、`purpose` 是否必填）。
5. `BlockAssembler` / `createUserMessage` 从 `@deepseek-ai/dsh-llm` 导出。
6. `@deepseek-ai/dsh-compaction` 在目标 dsh 版本里的导出（`CompactionEngine`、`CompactionId`、`compactCheckpointSource`、`isCompactCheckpointSource`、`toolPairingBalancedBefore`）。
7. cordis 插件 `inject` 里 `llm`/`tokenMeter`/`sessions` 的服务名是否一致。
8. Electron 侧：`stage-dsh.mjs` 如何把 `dsh-plugins/` stage 到 resources；`pathToFileURL` 的 file URL 能否被 dsh 的 cordis loader 接受。

**核对方法**：在目标分支 `cd desktop && npm ci`，然后 `grep -rn "<方法名>" node_modules/@deepseek-ai/dsh-*/lib`。

---

## 9. 参考

- dsh 源码（已下载核对）：`@deepseek-ai/dsh-compaction`、`@deepseek-ai/dsh-compaction-basic`、`@deepseek-ai/dsh-compaction-tool-result-pruner`、`@deepseek-ai/dsh-session`、`@deepseek-ai/dsh-llm`（均 0.1.5-rc.1）。
- 关键结论：`selectCompactableRange` 的 `firstIdx=0/1` 是命中率低的根因；`session.append("user/message", ..., { surfaceOp:{op:"replace", startSeq, endSeq} })` 是唯一需要改的 surface 写入点。
- 日报：`cloud.haiguji.com → 上下文治理 → LLM用量与缓存命中率日报-*.md`。
