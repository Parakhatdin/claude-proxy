import { config, log } from "./config.ts";
import { badRequest } from "./errors.ts";
import { systemToText } from "./tokens.ts";
import {
  buildToolPrompt,
  normalizeToolChoice,
  normalizeTools,
  renderToolBlock,
  type ToolChoice,
  type ToolDef,
} from "./tools.ts";
import type { ContentBlock, InputMessage, MessagesRequest } from "./types.ts";

export interface Validated {
  model: string;
  system: string;
  messages: InputMessage[];
  maxTokens: number;
  stopSequences: string[];
  stream: boolean;
  includeThinking: boolean;
  tools: ToolDef[];
  toolChoice: ToolChoice;
  /** JSON Schema the reply must satisfy (OpenAI `response_format: json_schema`). */
  jsonSchema: unknown | null;
  /** Free-form JSON mode (OpenAI `response_format: json_object`). */
  jsonMode: boolean;
}

const ROLE_LABEL: Record<string, string> = { user: "Human", assistant: "Assistant" };

const MODEL_ALIASES = new Set(["opus", "sonnet", "haiku", "fable", "default", "opusplan"]);

/**
 * Clients configured for another provider (Hermes defaults to "gpt-5.4", for instance) would
 * otherwise make the CLI fail on an unknown `--model`. Fall back instead, and report the model
 * the CLI actually used in the response.
 */
export function resolveModel(requested: string | undefined): string {
  const m = (requested ?? "").trim();
  if (!m) return config.defaultModel;
  if (m.startsWith("claude") || MODEL_ALIASES.has(m.toLowerCase())) return m;
  log("warn", `unknown model ${JSON.stringify(m)}; using ${config.defaultModel}`);
  return config.defaultModel;
}

export function validate(body: unknown): Validated {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw badRequest("Request body must be a JSON object.");
  }
  const req = body as MessagesRequest;

  if (!Array.isArray(req.messages) || req.messages.length === 0) {
    throw badRequest("`messages` must be a non-empty array.");
  }
  for (const [i, m] of req.messages.entries()) {
    if (typeof m !== "object" || m === null) throw badRequest(`messages[${i}] must be an object.`);
    if (m.role !== "user" && m.role !== "assistant") {
      throw badRequest(`messages[${i}].role must be "user" or "assistant".`);
    }
    if (typeof m.content !== "string" && !Array.isArray(m.content)) {
      throw badRequest(`messages[${i}].content must be a string or an array of content blocks.`);
    }
  }

  const maxTokens = req.max_tokens;
  if (maxTokens !== undefined && (!Number.isInteger(maxTokens) || maxTokens < 1)) {
    throw badRequest("`max_tokens` must be a positive integer.");
  }

  const stopSequences = req.stop_sequences ?? [];
  if (!Array.isArray(stopSequences) || stopSequences.some((s) => typeof s !== "string")) {
    throw badRequest("`stop_sequences` must be an array of strings.");
  }

  return {
    model: resolveModel(req.model),
    system: systemToText(req.system),
    messages: req.messages,
    maxTokens: maxTokens ?? Number.MAX_SAFE_INTEGER,
    stopSequences,
    stream: req.stream === true,
    // The CLI cannot stop the model from thinking, but the API only surfaces the blocks
    // when the caller opted in.
    includeThinking: config.includeThinking || req.thinking?.type === "enabled",
    tools: normalizeTools(req.tools, "anthropic"),
    toolChoice: normalizeToolChoice(req.tool_choice, "anthropic"),
    jsonSchema: null,
    jsonMode: false,
  };
}

/** The system prompt actually handed to the CLI, including any tool or JSON instructions. */
export function effectiveSystem(v: Validated): string {
  const parts = [v.system];
  const toolPrompt = buildToolPrompt(v.tools, v.toolChoice);
  if (toolPrompt) parts.push(toolPrompt);
  if (v.jsonMode && !v.jsonSchema) {
    parts.push(
      "Respond with a single valid JSON value and nothing else: no prose, no markdown fences.",
    );
  }
  return parts.filter(Boolean).join("\n\n");
}

export function buildArgs(v: Validated, resumeSessionId: string | null): string[] {
  const args = [
    "--print",
    "--verbose",
    "--output-format",
    "stream-json",
    "--input-format",
    "stream-json",
    "--model",
    v.model,
    "--strict-mcp-config",
  ];

  if (v.stream) args.push("--include-partial-messages");

  // A schema constrains the whole reply, so it cannot coexist with tool-call blocks.
  if (v.jsonSchema && v.tools.length === 0) {
    args.push("--json-schema", JSON.stringify(v.jsonSchema));
  }

  const system = effectiveSystem(v);
  if (config.mode === "clean") {
    // Strip CLAUDE.md, skills, plugins, hooks and MCP servers so the endpoint behaves like a
    // plain model rather than like this machine's Claude Code install.
    args.push("--safe-mode", "--disable-slash-commands", "--tools", "");
    args.push("--system-prompt", system || config.defaultSystem);
  } else {
    args.push("--permission-mode", config.permissionMode);
    if (config.agentTools.length > 0) args.push("--tools", ...config.agentTools);
    if (system) args.push("--append-system-prompt", system);
  }

  if (resumeSessionId) args.push("--resume", resumeSessionId);
  else if (!config.sessionReuse) args.push("--no-session-persistence");

  args.push(...config.extraArgs);
  return args;
}

function normalizeBlocks(content: string | ContentBlock[]): ContentBlock[] {
  if (typeof content === "string") return content ? [{ type: "text", text: content }] : [];
  return content;
}

function blockToText(b: ContentBlock): string {
  if (b.type === "text") return (b as { text: string }).text;
  if (b.type === "image") return "[image]";
  return renderToolBlock(b) ?? "";
}

/** Blocks the CLI accepts on the wire: text and images only, tools rendered into text. */
function toWireBlocks(content: string | ContentBlock[]): ContentBlock[] {
  const out: ContentBlock[] = [];
  let pending = "";
  for (const b of normalizeBlocks(content)) {
    if (b.type === "image") {
      if (pending) {
        out.push({ type: "text", text: pending });
        pending = "";
      }
      out.push(b);
      continue;
    }
    const text = blockToText(b);
    if (text) pending = pending ? `${pending}\n${text}` : text;
  }
  if (pending) out.push({ type: "text", text: pending });
  return out;
}

function renderTranscript(messages: InputMessage[]): string {
  const parts: string[] = [];
  for (const m of messages) {
    const text = normalizeBlocks(m.content).map(blockToText).filter(Boolean).join("\n");
    parts.push(`${ROLE_LABEL[m.role] ?? m.role}: ${text}`);
  }
  return parts.join("\n\n");
}

/**
 * Builds the single stream-json user turn to feed the CLI.
 *
 * On a resumed session the CLI already holds the history, so only the newest turn is sent.
 * On a cold start the earlier turns are replayed as a transcript inside one user message —
 * feeding them as separate stream-json turns would make the model answer each one again.
 */
export function buildStdinPayload(v: Validated, resumed: boolean): string {
  const last = v.messages[v.messages.length - 1]!;
  const history = v.messages.slice(0, -1);

  let blocks: ContentBlock[];
  if (resumed) {
    blocks = toWireBlocks(last.content);
  } else if (history.length === 0 && last.role === "user") {
    blocks = toWireBlocks(last.content);
  } else {
    const shown = last.role === "user" ? history : v.messages;
    const preamble =
      `Here is the conversation so far:\n\n<transcript>\n${renderTranscript(shown)}\n</transcript>\n\n` +
      (last.role === "user"
        ? "Reply to the final Human turn below. Do not repeat the transcript."
        : "Continue the final Assistant turn directly, without preamble.");
    blocks = [{ type: "text", text: preamble }];
    if (last.role === "user") blocks.push(...toWireBlocks(last.content));
  }

  if (blocks.length === 0) blocks = [{ type: "text", text: "" }];

  return `${JSON.stringify({
    type: "user",
    message: { role: "user", content: blocks },
  })}\n`;
}

/** The conversation prefix a resumed session must already have seen. */
export function prefixMessages(messages: InputMessage[]): InputMessage[] {
  return messages.slice(0, -1);
}
