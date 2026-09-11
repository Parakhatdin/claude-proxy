/** Subset of the Anthropic Messages API surface that this proxy speaks. */

export interface TextBlock {
  type: "text";
  text: string;
}

export interface ImageBlock {
  type: "image";
  source:
    | { type: "base64"; media_type: string; data: string }
    | { type: "url"; url: string };
}

export interface ThinkingBlock {
  type: "thinking";
  thinking: string;
  signature?: string;
}

export interface ToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
}

export interface ToolResultBlock {
  type: "tool_result";
  tool_use_id: string;
  content?: string | ContentBlock[];
  is_error?: boolean;
}

export type ContentBlock =
  | TextBlock
  | ImageBlock
  | ThinkingBlock
  | ToolUseBlock
  | ToolResultBlock
  | { type: string; [k: string]: unknown };

export interface InputMessage {
  role: "user" | "assistant";
  content: string | ContentBlock[];
}

export type SystemPrompt = string | TextBlock[];

export interface MessagesRequest {
  model?: string;
  messages: InputMessage[];
  system?: SystemPrompt;
  max_tokens?: number;
  stream?: boolean;
  stop_sequences?: string[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  metadata?: { user_id?: string };
  tools?: unknown[];
  tool_choice?: unknown;
  thinking?: { type: "enabled" | "disabled"; budget_tokens?: number };
}

export type StopReason =
  | "end_turn"
  | "max_tokens"
  | "stop_sequence"
  | "tool_use"
  | "pause_turn"
  | "refusal"
  | null;

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

export interface MessageResponse {
  id: string;
  type: "message";
  role: "assistant";
  model: string;
  content: ContentBlock[];
  stop_reason: StopReason;
  stop_sequence: string | null;
  usage: Usage;
}

/** Anthropic SSE event, passed through from the CLI with minimal rewriting. */
export interface StreamEvent {
  type: string;
  [k: string]: unknown;
}

/* ---------- Shapes emitted by `claude --output-format stream-json` ---------- */

export interface CliInitMessage {
  type: "system";
  subtype: "init";
  session_id: string;
  model: string;
  tools?: string[];
}

export interface CliAssistantMessage {
  type: "assistant";
  message: {
    id: string;
    model: string;
    role: "assistant";
    content: ContentBlock[];
    stop_reason: StopReason;
    stop_sequence: string | null;
    usage?: Partial<Usage> & Record<string, unknown>;
  };
  session_id: string;
  parent_tool_use_id: string | null;
}

export interface CliStreamEventMessage {
  type: "stream_event";
  event: StreamEvent;
  session_id: string;
  parent_tool_use_id: string | null;
}

export interface CliResultMessage {
  type: "result";
  subtype: "success" | "error_max_turns" | "error_during_execution" | string;
  is_error: boolean;
  result?: string;
  /** Present when `--json-schema` was used: the validated value the model produced. */
  structured_output?: unknown;
  stop_reason?: StopReason;
  session_id: string;
  total_cost_usd?: number;
  duration_ms?: number;
  usage?: Partial<Usage> & Record<string, unknown>;
}

export type CliMessage =
  | CliInitMessage
  | CliAssistantMessage
  | CliStreamEventMessage
  | CliResultMessage
  | { type: string; subtype?: string; [k: string]: unknown };
