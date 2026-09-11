import { randomUUID } from "node:crypto";
import { config } from "./config.ts";
import { estimateTokens } from "./tokens.ts";
import type {
  ContentBlock,
  MessageResponse,
  StopReason,
  StreamEvent,
  Usage,
} from "./types.ts";
import type { RunnerEvent } from "./runner.ts";

const CHARS_PER_TOKEN = 3.8;

export function newMessageId(): string {
  return `msg_proxy_${randomUUID().replace(/-/g, "")}`;
}

export interface Halt {
  reason: "stop_sequence" | "max_tokens";
  sequence?: string;
}

/**
 * Enforces `max_tokens` and `stop_sequences`, which the CLI does not implement. Token
 * budgets are applied against the estimator in tokens.ts, so the cutoff is approximate.
 */
export class Limiter {
  private buf = "";
  private emitted = 0;
  private stopped = false;
  private readonly stopSequences: string[];
  private readonly allowedChars: number;
  private readonly longestStop: number;

  constructor(stopSequences: string[], maxTokens: number) {
    this.stopSequences = stopSequences.filter(Boolean);
    this.allowedChars =
      config.enforceMaxTokens && maxTokens !== Number.MAX_SAFE_INTEGER
        ? Math.floor(maxTokens * CHARS_PER_TOKEN)
        : Number.MAX_SAFE_INTEGER;
    this.longestStop = this.stopSequences.reduce((max, s) => Math.max(max, s.length), 0);
  }

  /** Text actually forwarded to the client so far. */
  get accumulated(): string {
    return this.buf.slice(0, this.emitted);
  }

  /**
   * Length of the tail that is still a possible prefix of a stop sequence, and so must be
   * withheld until the next delta resolves it. Zero when no stop sequences are configured.
   */
  private holdback(): number {
    let held = 0;
    for (const seq of this.stopSequences) {
      const max = Math.min(seq.length - 1, this.buf.length);
      for (let k = max; k > held; k--) {
        if (this.buf.endsWith(seq.slice(0, k))) {
          held = k;
          break;
        }
      }
    }
    return held;
  }

  /** Feeds one text delta and returns the portion safe to forward plus any stop condition. */
  push(delta: string): { emit: string; halt: Halt | null } {
    if (this.stopped) return { emit: "", halt: null };

    const prevLen = this.buf.length;
    this.buf += delta;

    let stopAt = -1;
    let stopSeq: string | undefined;
    if (this.longestStop > 0) {
      // Start far enough back to catch a sequence straddling the delta boundary.
      const from = Math.max(0, prevLen - this.longestStop + 1);
      for (const seq of this.stopSequences) {
        const at = this.buf.indexOf(seq, from);
        if (at !== -1 && (stopAt === -1 || at < stopAt)) {
          stopAt = at;
          stopSeq = seq;
        }
      }
    }

    let halt: Halt | null = null;
    let end: number;
    if (stopAt !== -1 && stopAt <= this.allowedChars) {
      end = stopAt;
      halt = { reason: "stop_sequence", sequence: stopSeq };
    } else if (this.buf.length > this.allowedChars) {
      end = this.allowedChars;
      halt = { reason: "max_tokens" };
    } else {
      end = this.buf.length - this.holdback();
    }

    if (end < this.emitted) end = this.emitted;
    const emit = this.buf.slice(this.emitted, end);
    this.emitted = end;
    if (halt) {
      this.stopped = true;
      this.buf = this.buf.slice(0, end);
    }
    return { emit, halt };
  }

  /** Releases any withheld tail once no further deltas can arrive. */
  flush(): string {
    if (this.stopped) return "";
    const rest = this.buf.slice(this.emitted);
    this.emitted = this.buf.length;
    return rest;
  }
}

export interface RenderOptions {
  /** Thinking blocks are forwarded only when the caller asked for them. */
  includeThinking: boolean;
}

function keepBlock(type: unknown, opts: RenderOptions): boolean {
  if (type === "thinking" || type === "redacted_thinking") return opts.includeThinking;
  if (type === "tool_use" || type === "server_tool_use") return config.exposeToolBlocks;
  return true;
}

function mergeUsage(target: Usage, raw: unknown): void {
  if (typeof raw !== "object" || raw === null) return;
  const u = raw as Record<string, unknown>;
  const num = (k: string): number => (typeof u[k] === "number" ? (u[k] as number) : 0);
  // Later reports from the CLI supersede earlier partial ones.
  if (typeof u["input_tokens"] === "number") target.input_tokens = num("input_tokens");
  if (typeof u["output_tokens"] === "number") target.output_tokens = num("output_tokens");
  if (typeof u["cache_creation_input_tokens"] === "number") {
    target.cache_creation_input_tokens = num("cache_creation_input_tokens");
  }
  if (typeof u["cache_read_input_tokens"] === "number") {
    target.cache_read_input_tokens = num("cache_read_input_tokens");
  }
}

function emptyUsage(): Usage {
  return {
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  };
}

/* ----------------------------- non-streaming ----------------------------- */

export interface Collected {
  message: MessageResponse;
  sessionId: string;
}

export function collectMessage(
  events: Iterable<RunnerEvent>,
  requestedModel: string,
  limiterFactory: () => Limiter,
  opts: RenderOptions,
): Collected {
  const content: ContentBlock[] = [];
  const usage = emptyUsage();
  const limiter = limiterFactory();

  let id = newMessageId();
  let model = requestedModel;
  let sessionId = "";
  let stopReason: StopReason = "end_turn";
  let stopSequence: string | null = null;
  let halted = false;
  let structured: unknown;

  for (const ev of events) {
    if (ev.kind === "init") {
      sessionId = ev.sessionId || sessionId;
      model = ev.model || model;
      continue;
    }
    if (ev.kind === "assistant") {
      if (content.length === 0 && ev.message.id) id = ev.message.id;
      model = ev.message.model || model;
      mergeUsage(usage, ev.message.usage);
      if (halted) continue;
      for (const block of ev.message.content ?? []) {
        if (!keepBlock(block.type, opts)) continue;
        if (block.type === "text") {
          const { emit, halt } = limiter.push((block as { text: string }).text);
          // The whole block is known here, so nothing needs withholding past its end.
          const text = halt ? emit : emit + limiter.flush();
          if (text) content.push({ type: "text", text });
          if (halt) {
            stopReason = halt.reason;
            stopSequence = halt.sequence ?? null;
            halted = true;
            break;
          }
        } else {
          content.push(block);
        }
      }
      continue;
    }
    if (ev.kind === "result") {
      sessionId = ev.result.session_id || sessionId;
      mergeUsage(usage, ev.result.usage);
      if (ev.result.structured_output !== undefined) structured = ev.result.structured_output;
      if (!halted && ev.result.stop_reason) stopReason = ev.result.stop_reason;
      if (ev.result.is_error) {
        stopReason = stopReason === "end_turn" ? "end_turn" : stopReason;
      }
    }
  }

  // With --json-schema the model answers through an internal StructuredOutput tool call
  // rather than text, so the validated value is the reply.
  if (structured !== undefined && !content.some((b) => b.type === "text")) {
    content.length = 0;
    content.push({ type: "text", text: JSON.stringify(structured) });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  if (usage.output_tokens === 0) {
    usage.output_tokens = estimateTokens(limiter.accumulated);
  }

  return {
    message: {
      id,
      type: "message",
      role: "assistant",
      model,
      content,
      stop_reason: stopReason,
      stop_sequence: stopSequence,
      usage,
    },
    sessionId,
  };
}

/* ------------------------------- streaming ------------------------------- */

/**
 * Rewrites the CLI's per-turn SSE events into one logical Anthropic message. In agent mode
 * the CLI emits several assistant messages per request, so block indices are remapped onto a
 * single continuous sequence and the intermediate message_delta/message_stop are suppressed.
 */
export class StreamAssembler {
  private started = false;
  private outIndex = 0;
  private indexMap = new Map<number, number>();
  private openBlock: number | null = null;
  private readonly textBlocks = new Set<number>();
  private readonly usage = emptyUsage();
  private readonly limiter: Limiter;
  private readonly opts: RenderOptions;

  private id = newMessageId();
  private model: string;
  private stopReason: StopReason = "end_turn";
  private stopSequence: string | null = null;

  /** Set once a stop condition fires; the caller should stop reading and kill the CLI. */
  halted = false;
  sessionId = "";

  constructor(requestedModel: string, limiter: Limiter, opts: RenderOptions) {
    this.model = requestedModel;
    this.limiter = limiter;
    this.opts = opts;
  }

  handle(ev: RunnerEvent): StreamEvent[] {
    if (this.halted) return [];

    if (ev.kind === "init") {
      this.sessionId = ev.sessionId || this.sessionId;
      this.model = ev.model || this.model;
      return [];
    }
    if (ev.kind === "result") {
      this.sessionId = ev.result.session_id || this.sessionId;
      mergeUsage(this.usage, ev.result.usage);
      if (ev.result.stop_reason) this.stopReason = ev.result.stop_reason;
      return [];
    }
    if (ev.kind === "assistant") {
      mergeUsage(this.usage, ev.message.usage);
      return [];
    }
    return this.handleStream(ev.event);
  }

  private handleStream(event: StreamEvent): StreamEvent[] {
    switch (event["type"]) {
      case "message_start": {
        const msg = event["message"] as Record<string, unknown> | undefined;
        this.indexMap.clear();
        if (this.started) return [];
        this.started = true;
        if (typeof msg?.["id"] === "string") this.id = msg["id"] as string;
        if (typeof msg?.["model"] === "string") this.model = msg["model"] as string;
        mergeUsage(this.usage, msg?.["usage"]);
        return [
          {
            type: "message_start",
            message: {
              id: this.id,
              type: "message",
              role: "assistant",
              model: this.model,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: { ...this.usage },
            },
          },
        ];
      }

      case "content_block_start": {
        const src = event["index"] as number;
        const block = event["content_block"] as { type?: unknown } | undefined;
        if (!keepBlock(block?.type, this.opts)) return [];
        const out = this.outIndex++;
        this.indexMap.set(src, out);
        this.openBlock = out;
        if (block?.type === "text") this.textBlocks.add(out);
        return [{ ...event, index: out }];
      }

      case "content_block_delta": {
        const src = event["index"] as number;
        const out = this.indexMap.get(src);
        if (out === undefined) return [];
        const delta = event["delta"] as Record<string, unknown> | undefined;
        if (delta?.["type"] !== "text_delta") return [{ ...event, index: out }];

        const { emit, halt } = this.limiter.push(String(delta["text"] ?? ""));
        const out_events: StreamEvent[] = [];
        if (emit) {
          out_events.push({
            type: "content_block_delta",
            index: out,
            delta: { type: "text_delta", text: emit },
          });
        }
        if (halt) {
          this.stopReason = halt.reason;
          this.stopSequence = halt.sequence ?? null;
          this.halted = true;
          out_events.push(...this.finalize());
        }
        return out_events;
      }

      case "content_block_stop": {
        const src = event["index"] as number;
        const out = this.indexMap.get(src);
        if (out === undefined) return [];
        if (this.openBlock === out) this.openBlock = null;
        // No more deltas for this block, so any text held back for stop-sequence
        // matching has to go out before the block closes.
        return [...this.flushInto(out), { ...event, index: out }];
      }

      case "message_delta": {
        mergeUsage(this.usage, event["usage"]);
        const delta = event["delta"] as Record<string, unknown> | undefined;
        if (typeof delta?.["stop_reason"] === "string") {
          this.stopReason = delta["stop_reason"] as StopReason;
        }
        // Suppressed: the proxy emits one message_delta of its own at the end.
        return [];
      }

      case "message_stop":
        return [];

      default:
        return [];
    }
  }

  /** Closes the logical message. Safe to call once; repeated calls return nothing. */
  finalize(): StreamEvent[] {
    if (this.finalized) return [];
    this.finalized = true;

    const events: StreamEvent[] = [];
    if (!this.started) {
      events.push({
        type: "message_start",
        message: {
          id: this.id,
          type: "message",
          role: "assistant",
          model: this.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { ...this.usage },
        },
      });
      this.started = true;
    }
    if (this.openBlock !== null) {
      events.push(...this.flushInto(this.openBlock));
      events.push({ type: "content_block_stop", index: this.openBlock });
      this.openBlock = null;
    }
    if (this.usage.output_tokens === 0) {
      this.usage.output_tokens = estimateTokens(this.limiter.accumulated);
    }
    events.push({
      type: "message_delta",
      delta: { stop_reason: this.stopReason, stop_sequence: this.stopSequence },
      usage: { ...this.usage },
    });
    events.push({ type: "message_stop" });
    return events;
  }

  private flushInto(out: number): StreamEvent[] {
    if (!this.textBlocks.has(out)) return [];
    const rest = this.limiter.flush();
    if (!rest) return [];
    return [
      { type: "content_block_delta", index: out, delta: { type: "text_delta", text: rest } },
    ];
  }

  private finalized = false;
}

/**
 * Replays a finished message as an Anthropic event stream. Needed when the reply had to be
 * buffered (tool calls) but the caller still asked for `stream: true`.
 */
export function synthesizeAnthropicStream(msg: MessageResponse): StreamEvent[] {
  const events: StreamEvent[] = [
    {
      type: "message_start",
      message: { ...msg, content: [], stop_reason: null, stop_sequence: null },
    },
  ];

  msg.content.forEach((block, index) => {
    if (block.type === "text") {
      events.push({ type: "content_block_start", index, content_block: { type: "text", text: "" } });
      events.push({
        type: "content_block_delta",
        index,
        delta: { type: "text_delta", text: (block as { text: string }).text },
      });
    } else if (block.type === "tool_use") {
      const t = block as { id: string; name: string; input: unknown };
      events.push({
        type: "content_block_start",
        index,
        content_block: { type: "tool_use", id: t.id, name: t.name, input: {} },
      });
      events.push({
        type: "content_block_delta",
        index,
        delta: { type: "input_json_delta", partial_json: JSON.stringify(t.input ?? {}) },
      });
    } else {
      events.push({ type: "content_block_start", index, content_block: block });
    }
    events.push({ type: "content_block_stop", index });
  });

  events.push({
    type: "message_delta",
    delta: { stop_reason: msg.stop_reason, stop_sequence: msg.stop_sequence },
    usage: msg.usage,
  });
  events.push({ type: "message_stop" });
  return events;
}
