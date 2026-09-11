import { randomUUID } from "node:crypto";
import { badRequest } from "./errors.ts";
import type { ContentBlock, MessageResponse } from "./types.ts";

/**
 * Client-side tool calling on top of a CLI that will not surrender its own tool loop.
 *
 * `claude -p` never pauses to hand a `tool_use` back to the caller, so the tools a client
 * defines are described in the system prompt instead. The model emits a `<tool_call>` block,
 * which is parsed back into native `tool_calls` / `tool_use` before the response is returned.
 * Results arrive on the next request as ordinary transcript turns. This is prompt-level
 * emulation, not the provider's native tool use — see README "Fidelity".
 */

export interface ToolDef {
  name: string;
  description?: string;
  parameters: unknown;
}

export type ToolChoice =
  | { mode: "auto" }
  | { mode: "none" }
  | { mode: "required" }
  | { mode: "function"; name: string };

export interface ParsedCall {
  id: string;
  name: string;
  /** JSON-encoded, matching OpenAI's `function.arguments`. */
  arguments: string;
}

const TOOL_CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;

export function newCallId(): string {
  return `call_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

/** Accepts either dialect's tool declaration and produces one shape for the prompt. */
export function normalizeTools(raw: unknown, dialect: "openai" | "anthropic"): ToolDef[] {
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) throw badRequest("`tools` must be an array.");
  const out: ToolDef[] = [];
  for (const [i, item] of raw.entries()) {
    if (typeof item !== "object" || item === null) throw badRequest(`tools[${i}] must be an object.`);
    const t = item as Record<string, unknown>;
    if (dialect === "openai") {
      // {type:"function", function:{name, description, parameters}} or the legacy flat form.
      const fn = (t["function"] ?? t) as Record<string, unknown>;
      const name = fn["name"];
      if (typeof name !== "string" || !name) throw badRequest(`tools[${i}] is missing a function name.`);
      out.push({ name, description: String(fn["description"] ?? ""), parameters: fn["parameters"] ?? {} });
    } else {
      const name = t["name"];
      if (typeof name !== "string" || !name) throw badRequest(`tools[${i}].name is required.`);
      // Anthropic server-side tools have no schema for the model to fill in; skip them.
      if (t["type"] !== undefined && t["input_schema"] === undefined) continue;
      out.push({ name, description: String(t["description"] ?? ""), parameters: t["input_schema"] ?? {} });
    }
  }
  return out;
}

export function normalizeToolChoice(raw: unknown, dialect: "openai" | "anthropic"): ToolChoice {
  if (raw === undefined || raw === null) return { mode: "auto" };
  if (typeof raw === "string") {
    if (raw === "none") return { mode: "none" };
    if (raw === "required") return { mode: "required" };
    return { mode: "auto" };
  }
  if (typeof raw !== "object") return { mode: "auto" };
  const c = raw as Record<string, unknown>;
  if (dialect === "openai") {
    const name = (c["function"] as { name?: unknown } | undefined)?.name;
    if (typeof name === "string") return { mode: "function", name };
    return { mode: "auto" };
  }
  const type = c["type"];
  if (type === "any") return { mode: "required" };
  if (type === "none") return { mode: "none" };
  if (type === "tool" && typeof c["name"] === "string") return { mode: "function", name: c["name"] as string };
  return { mode: "auto" };
}

/** The instructions appended to the system prompt when a caller declares tools. */
export function buildToolPrompt(tools: ToolDef[], choice: ToolChoice): string {
  if (tools.length === 0 || choice.mode === "none") return "";

  const lines = [
    "# Tool calling",
    "",
    "You can call tools that the caller will run for you. To call one, emit exactly:",
    "",
    '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>',
    "",
    "Rules:",
    "- A reply containing a tool call must contain nothing else: no prose, no explanation,",
    "  no markdown fences around the block.",
    "- Emit several blocks in one reply to request several calls at once.",
    '- "arguments" must be a JSON object that matches the tool\'s parameter schema.',
    "- A <tool_result> turn in the conversation is the output of a call you already made.",
    "  Use it to answer; do not call the same tool again with the same arguments.",
  ];

  if (choice.mode === "required") {
    lines.push("- You MUST emit a tool call in this reply.");
  } else if (choice.mode === "function") {
    lines.push(`- You MUST call the tool "${choice.name}" in this reply.`);
  } else {
    lines.push("- If no tool is needed, answer normally and emit no tool call block.");
  }

  lines.push("", "Available tools:", "", JSON.stringify(tools, null, 2));
  return lines.join("\n");
}

function stripFences(raw: string): string {
  const fenced = /^```(?:json)?\s*([\s\S]*?)\s*```$/.exec(raw.trim());
  return fenced ? fenced[1]!.trim() : raw.trim();
}

/**
 * Extracts tool calls from a model reply. Malformed blocks are left as literal text rather
 * than silently dropped, so a caller can see what the model actually produced.
 */
export function parseToolCalls(text: string): { calls: ParsedCall[]; text: string } {
  const calls: ParsedCall[] = [];
  let residual = "";
  let last = 0;

  for (const m of text.matchAll(TOOL_CALL_RE)) {
    const at = m.index ?? 0;
    residual += text.slice(last, at);
    last = at + m[0].length;

    let parsed: unknown;
    try {
      parsed = JSON.parse(stripFences(m[1] ?? ""));
    } catch {
      residual += m[0];
      continue;
    }
    const obj = parsed as Record<string, unknown> | null;
    if (!obj || typeof obj["name"] !== "string") {
      residual += m[0];
      continue;
    }
    const args = obj["arguments"] ?? obj["parameters"] ?? obj["input"] ?? {};
    calls.push({
      id: newCallId(),
      name: obj["name"] as string,
      arguments: typeof args === "string" ? args : JSON.stringify(args),
    });
  }
  residual += text.slice(last);
  return { calls, text: residual.trim() };
}

/** Rewrites a completed message so emulated calls appear as real `tool_use` blocks. */
export function applyToolCalls(message: MessageResponse): MessageResponse {
  const text = message.content
    .filter((b) => b.type === "text")
    .map((b) => (b as { text: string }).text)
    .join("");

  const { calls, text: residual } = parseToolCalls(text);
  if (calls.length === 0) return message;

  const content: ContentBlock[] = [];
  if (residual) content.push({ type: "text", text: residual });
  for (const call of calls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: call.id, name: call.name, input });
  }

  const others = message.content.filter((b) => b.type !== "text");
  return { ...message, content: [...content, ...others], stop_reason: "tool_use" };
}

/** Renders a tool_use / tool_result block back into the transcript wire form. */
export function renderToolBlock(block: ContentBlock): string | null {
  if (block.type === "tool_use") {
    const b = block as { name: string; input: unknown };
    return `<tool_call>${JSON.stringify({ name: b.name, arguments: b.input ?? {} })}</tool_call>`;
  }
  if (block.type === "tool_result") {
    const b = block as { tool_use_id?: string; content?: unknown; is_error?: boolean };
    let body: string;
    if (typeof b.content === "string") body = b.content;
    else if (Array.isArray(b.content)) {
      body = b.content
        .map((p) => (typeof p === "object" && p !== null && (p as { text?: unknown }).text !== undefined
          ? String((p as { text: unknown }).text)
          : ""))
        .join("\n");
    } else body = b.content === undefined ? "" : JSON.stringify(b.content);
    const err = b.is_error ? ' is_error="true"' : "";
    return `<tool_result${err}>${body}</tool_result>`;
  }
  return null;
}
