import { randomUUID } from "node:crypto";
import { config } from "./config.ts";
import { badRequest } from "./errors.ts";
import type { ApiError } from "./errors.ts";
import { resolveModel } from "./translate.ts";
import type { Validated } from "./translate.ts";
import { normalizeToolChoice, normalizeTools } from "./tools.ts";
import type {
  ContentBlock,
  InputMessage,
  MessageResponse,
  StopReason,
  StreamEvent,
} from "./types.ts";

/** OpenAI Chat Completions compatibility layer over the same engine. */

interface ChatRequest {
  model?: string;
  messages?: unknown[];
  max_tokens?: number;
  max_completion_tokens?: number;
  stream?: boolean;
  stop?: string | string[];
  n?: number;
  tools?: unknown;
  functions?: unknown;
  tool_choice?: unknown;
  function_call?: unknown;
  response_format?: { type?: string; json_schema?: { schema?: unknown } };
  stream_options?: { include_usage?: boolean };
}

export interface ParsedChat {
  v: Validated;
  includeUsage: boolean;
  n: number;
}

export function chatCompletionId(): string {
  return `chatcmpl-${randomUUID().replace(/-/g, "")}`;
}

function convertContent(content: unknown, index: number): ContentBlock[] {
  if (content === null || content === undefined) return [];
  if (typeof content === "string") return content ? [{ type: "text", text: content }] : [];
  if (!Array.isArray(content)) {
    throw badRequest(`messages[${index}].content must be a string or an array of parts.`);
  }
  const blocks: ContentBlock[] = [];
  for (const part of content) {
    if (typeof part !== "object" || part === null) continue;
    const p = part as Record<string, unknown>;
    if (p["type"] === "text" || typeof p["text"] === "string") {
      blocks.push({ type: "text", text: String(p["text"] ?? "") });
      continue;
    }
    if (p["type"] === "image_url") {
      const url = String((p["image_url"] as { url?: unknown })?.url ?? "");
      const dataUrl = /^data:([^;]+);base64,(.*)$/.exec(url);
      if (dataUrl) {
        blocks.push({
          type: "image",
          source: { type: "base64", media_type: dataUrl[1]!, data: dataUrl[2]! },
        });
      } else if (url) {
        blocks.push({ type: "image", source: { type: "url", url } });
      }
    }
  }
  return blocks;
}

function toolCallsToBlocks(raw: unknown): ContentBlock[] {
  if (!Array.isArray(raw)) return [];
  const blocks: ContentBlock[] = [];
  for (const call of raw) {
    if (typeof call !== "object" || call === null) continue;
    const c = call as Record<string, unknown>;
    const fn = (c["function"] ?? {}) as Record<string, unknown>;
    const name = String(fn["name"] ?? "");
    if (!name) continue;
    let input: unknown = {};
    const args = fn["arguments"];
    if (typeof args === "string") {
      try {
        input = JSON.parse(args || "{}");
      } catch {
        input = { _raw: args };
      }
    } else if (args && typeof args === "object") {
      input = args;
    }
    blocks.push({ type: "tool_use", id: String(c["id"] ?? ""), name, input });
  }
  return blocks;
}

/** Maps an OpenAI chat request onto the same internal shape the Anthropic route produces. */
export function parseChatRequest(body: unknown): ParsedChat {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw badRequest("Request body must be a JSON object.");
  }
  const req = body as ChatRequest;

  if (!Array.isArray(req.messages) || req.messages.length === 0) {
    throw badRequest("`messages` must be a non-empty array.");
  }

  const systemParts: string[] = [];
  const messages: InputMessage[] = [];
  // Consecutive tool results belong to one user turn, the way Anthropic models them.
  let pendingResults: ContentBlock[] = [];

  const flushResults = (): void => {
    if (pendingResults.length === 0) return;
    messages.push({ role: "user", content: pendingResults });
    pendingResults = [];
  };

  for (const [i, raw] of req.messages.entries()) {
    if (typeof raw !== "object" || raw === null) throw badRequest(`messages[${i}] must be an object.`);
    const m = raw as Record<string, unknown>;
    const role = m["role"];

    if (role === "system" || role === "developer") {
      flushResults();
      systemParts.push(
        convertContent(m["content"], i)
          .map((b) => (b.type === "text" ? (b as { text: string }).text : ""))
          .join("\n"),
      );
      continue;
    }

    if (role === "tool" || role === "function") {
      const content = m["content"];
      pendingResults.push({
        type: "tool_result",
        tool_use_id: String(m["tool_call_id"] ?? m["name"] ?? ""),
        content: typeof content === "string" ? content : JSON.stringify(content ?? ""),
      });
      continue;
    }

    flushResults();

    if (role === "assistant") {
      const blocks = convertContent(m["content"], i);
      blocks.push(...toolCallsToBlocks(m["tool_calls"]));
      const legacy = m["function_call"];
      if (legacy && typeof legacy === "object") {
        blocks.push(...toolCallsToBlocks([{ id: "", function: legacy }]));
      }
      messages.push({ role: "assistant", content: blocks });
      continue;
    }

    if (role !== "user") {
      throw badRequest(`messages[${i}].role "${String(role)}" is not a supported role.`);
    }
    messages.push({ role: "user", content: convertContent(m["content"], i) });
  }
  flushResults();

  if (messages.length === 0) {
    throw badRequest("`messages` must contain at least one user or assistant turn.");
  }

  const maxTokens = req.max_completion_tokens ?? req.max_tokens;
  if (maxTokens !== undefined && (!Number.isInteger(maxTokens) || maxTokens < 1)) {
    throw badRequest("`max_tokens` must be a positive integer.");
  }

  const stop = req.stop === undefined || req.stop === null ? [] : Array.isArray(req.stop) ? req.stop : [req.stop];
  if (stop.some((s) => typeof s !== "string")) {
    throw badRequest("`stop` must be a string or an array of strings.");
  }

  const n = req.n ?? 1;
  if (!Number.isInteger(n) || n < 1) throw badRequest("`n` must be a positive integer.");
  if (n > 1 && req.stream === true) {
    throw badRequest("`n` greater than 1 is not supported together with `stream`.");
  }
  if (n > config.maxChoices) {
    throw badRequest(
      `\`n\` may not exceed ${config.maxChoices}; each choice is a separate CLI run. ` +
        "Raise CLAUDE_PROXY_MAX_CHOICES to allow more.",
    );
  }

  // `functions` is the pre-2023 spelling of `tools`.
  const toolSource = req.tools ?? (req.functions === undefined ? undefined : req.functions);
  const tools = normalizeTools(toolSource, "openai");
  const choiceSource = req.tool_choice ?? req.function_call;

  const format = req.response_format?.type;
  const jsonSchema = format === "json_schema" ? (req.response_format?.json_schema?.schema ?? null) : null;

  return {
    v: {
      model: resolveModel(req.model),
      system: systemParts.filter(Boolean).join("\n\n"),
      messages,
      maxTokens: maxTokens ?? Number.MAX_SAFE_INTEGER,
      stopSequences: stop,
      stream: req.stream === true,
      // OpenAI responses have nowhere to put a thinking block.
      includeThinking: false,
      tools,
      toolChoice: normalizeToolChoice(choiceSource, "openai"),
      jsonSchema,
      jsonMode: format === "json_object" || format === "json_schema",
    },
    includeUsage: req.stream_options?.include_usage === true,
    n,
  };
}

export function finishReason(stop: StopReason): string {
  switch (stop) {
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    default:
      return "stop";
  }
}

function textOf(content: ContentBlock[]): string {
  return content
    .filter((b) => b.type === "text")
    .map((b) => (b as { text: string }).text)
    .join("");
}

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export function toolCallsOf(content: ContentBlock[]): OpenAIToolCall[] {
  return content
    .filter((b) => b.type === "tool_use")
    .map((b) => {
      const t = b as { id: string; name: string; input: unknown };
      return {
        id: t.id,
        type: "function" as const,
        function: { name: t.name, arguments: JSON.stringify(t.input ?? {}) },
      };
    });
}

function choiceOf(msg: MessageResponse, index: number): unknown {
  const calls = toolCallsOf(msg.content);
  const text = textOf(msg.content);
  const message: Record<string, unknown> = { role: "assistant", content: text || null };
  if (calls.length > 0) message["tool_calls"] = calls;
  return {
    index,
    message,
    finish_reason: calls.length > 0 ? "tool_calls" : finishReason(msg.stop_reason),
    logprobs: null,
  };
}

export function toChatCompletion(messages: MessageResponse[], id: string): unknown {
  const first = messages[0]!;
  const prompt = messages.reduce((n, m) => n + m.usage.input_tokens, 0);
  const completion = messages.reduce((n, m) => n + m.usage.output_tokens, 0);
  return {
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: first.model,
    choices: messages.map(choiceOf),
    usage: {
      prompt_tokens: prompt,
      completion_tokens: completion,
      total_tokens: prompt + completion,
    },
  };
}

export function toOpenAIError(err: ApiError): unknown {
  return {
    error: {
      message: err.message,
      type: err.type,
      param: null,
      code: err.status === 404 ? "not_found" : null,
    },
  };
}

function chunk(id: string, model: string, choices: unknown[], usage: unknown = null): string {
  const body: Record<string, unknown> = {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices,
  };
  if (usage !== null) body["usage"] = usage;
  return `data: ${JSON.stringify(body)}\n\n`;
}

/**
 * Synthesises a stream from an already-complete message. Used when tools are in play: the
 * reply has to be buffered to detect a tool-call block, so there is nothing to stream live.
 */
export function synthesizeChatStream(
  msg: MessageResponse,
  id: string,
  includeUsage: boolean,
): string[] {
  const model = msg.model;
  const out = [chunk(id, model, [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }])];

  const text = textOf(msg.content);
  if (text) out.push(chunk(id, model, [{ index: 0, delta: { content: text }, finish_reason: null }]));

  const calls = toolCallsOf(msg.content);
  calls.forEach((call, i) => {
    out.push(
      chunk(id, model, [
        {
          index: 0,
          delta: { tool_calls: [{ index: i, id: call.id, type: "function", function: call.function }] },
          finish_reason: null,
        },
      ]),
    );
  });

  const finish = calls.length > 0 ? "tool_calls" : finishReason(msg.stop_reason);
  out.push(chunk(id, model, [{ index: 0, delta: {}, finish_reason: finish }]));
  if (includeUsage) {
    out.push(
      chunk(id, model, [], {
        prompt_tokens: msg.usage.input_tokens,
        completion_tokens: msg.usage.output_tokens,
        total_tokens: msg.usage.input_tokens + msg.usage.output_tokens,
      }),
    );
  }
  out.push("data: [DONE]\n\n");
  return out;
}

/**
 * Rewrites the Anthropic SSE events the engine produces into OpenAI chat.completion.chunk
 * frames. OpenAI streams carry no event names and terminate with a literal [DONE].
 */
export class ChatChunkWriter {
  private readonly id: string;
  private readonly includeUsage: boolean;
  private model: string;
  private roleSent = false;
  private finish = "stop";
  private prompt = 0;
  private completion = 0;

  constructor(id: string, model: string, includeUsage: boolean) {
    this.id = id;
    this.model = model;
    this.includeUsage = includeUsage;
  }

  handle(event: StreamEvent): string[] {
    switch (event["type"]) {
      case "message_start": {
        const msg = event["message"] as Record<string, unknown> | undefined;
        if (typeof msg?.["model"] === "string") this.model = msg["model"] as string;
        const usage = msg?.["usage"] as Record<string, number> | undefined;
        if (usage) this.prompt = usage["input_tokens"] ?? 0;
        this.roleSent = true;
        return [chunk(this.id, this.model, [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }])];
      }
      case "content_block_delta": {
        const delta = event["delta"] as Record<string, unknown> | undefined;
        if (delta?.["type"] !== "text_delta") return [];
        const out: string[] = [];
        if (!this.roleSent) {
          this.roleSent = true;
          out.push(chunk(this.id, this.model, [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }]));
        }
        out.push(chunk(this.id, this.model, [{ index: 0, delta: { content: String(delta["text"] ?? "") }, finish_reason: null }]));
        return out;
      }
      case "message_delta": {
        const delta = event["delta"] as Record<string, unknown> | undefined;
        this.finish = finishReason((delta?.["stop_reason"] ?? null) as StopReason);
        const usage = event["usage"] as Record<string, number> | undefined;
        if (usage) {
          this.prompt = usage["input_tokens"] ?? this.prompt;
          this.completion = usage["output_tokens"] ?? this.completion;
        }
        return [];
      }
      default:
        return [];
    }
  }

  /** Final frame plus the [DONE] sentinel every OpenAI client waits for. */
  done(): string[] {
    const out = [chunk(this.id, this.model, [{ index: 0, delta: {}, finish_reason: this.finish }])];
    if (this.includeUsage) {
      out.push(
        chunk(this.id, this.model, [], {
          prompt_tokens: this.prompt,
          completion_tokens: this.completion,
          total_tokens: this.prompt + this.completion,
        }),
      );
    }
    out.push("data: [DONE]\n\n");
    return out;
  }
}

/** Legacy /v1/completions: a prompt string instead of a message list. */
export function parseLegacyCompletion(body: unknown): ParsedChat {
  if (typeof body !== "object" || body === null) throw badRequest("Request body must be a JSON object.");
  const req = body as Record<string, unknown> & ChatRequest;
  const prompt = req["prompt"];
  const text = Array.isArray(prompt) ? prompt.join("\n") : String(prompt ?? "");
  if (!text) throw badRequest("`prompt` is required.");
  const { prompt: _drop, ...rest } = req;
  return parseChatRequest({ ...rest, messages: [{ role: "user", content: text }] });
}

export function toLegacyCompletion(messages: MessageResponse[], id: string): unknown {
  const first = messages[0]!;
  const prompt = messages.reduce((n, m) => n + m.usage.input_tokens, 0);
  const completion = messages.reduce((n, m) => n + m.usage.output_tokens, 0);
  return {
    id: id.replace("chatcmpl-", "cmpl-"),
    object: "text_completion",
    created: Math.floor(Date.now() / 1000),
    model: first.model,
    choices: messages.map((m, index) => ({
      index,
      text: textOf(m.content),
      logprobs: null,
      finish_reason: finishReason(m.stop_reason),
    })),
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/**
 * Model list shaped to satisfy both dialects at one path: OpenAI clients read
 * `object`/`owned_by`, Anthropic clients read `type`/`display_name`.
 */
export function modelList(ids: string[]): unknown {
  const created = Math.floor(Date.now() / 1000);
  return {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      type: "model",
      created,
      owned_by: "anthropic",
      display_name: id,
    })),
    has_more: false,
  };
}
