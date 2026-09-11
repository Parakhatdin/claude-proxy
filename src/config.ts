import { tmpdir } from "node:os";

function str(name: string, dflt: string): string {
  const v = process.env[name];
  return v === undefined || v === "" ? dflt : v;
}

function optStr(name: string): string | undefined {
  const v = process.env[name];
  return v === undefined || v === "" ? undefined : v;
}

function int(name: string, dflt: number): number {
  const v = process.env[name];
  if (v === undefined || v === "") return dflt;
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) throw new Error(`${name} must be an integer, got ${JSON.stringify(v)}`);
  return n;
}

function bool(name: string, dflt: boolean): boolean {
  const v = process.env[name];
  if (v === undefined || v === "") return dflt;
  return ["1", "true", "yes", "on"].includes(v.toLowerCase());
}

function list(name: string): string[] {
  const v = process.env[name];
  if (v === undefined || v === "") return [];
  return v.split(/[,\s]+/).filter(Boolean);
}

/**
 * `agent` mode keeps Claude Code's own system prompt, tools and project context, so the
 * endpoint behaves like a coding agent. `clean` mode replaces the system prompt and turns
 * every tool off, so the endpoint behaves like a plain model API.
 */
export type Mode = "clean" | "agent";

export const config = {
  host: str("CLAUDE_PROXY_HOST", "127.0.0.1"),
  port: int("CLAUDE_PROXY_PORT", 8787),

  /** When set, callers must present it via `x-api-key` or `Authorization: Bearer`. */
  apiKey: optStr("CLAUDE_PROXY_API_KEY"),

  claudeBin: str("CLAUDE_PROXY_CLAUDE_BIN", "claude"),

  /** Working directory for the spawned CLI. Matters in agent mode; harmless in clean mode. */
  cwd: str("CLAUDE_PROXY_CWD", tmpdir()),

  mode: str("CLAUDE_PROXY_MODE", "clean") as Mode,

  defaultModel: str("CLAUDE_PROXY_DEFAULT_MODEL", "sonnet"),

  defaultSystem: str(
    "CLAUDE_PROXY_DEFAULT_SYSTEM",
    "You are Claude, a helpful AI assistant.",
  ),

  /** Tools passed to `--tools` in agent mode. Empty means Claude Code's default set. */
  agentTools: list("CLAUDE_PROXY_TOOLS"),
  permissionMode: str("CLAUDE_PROXY_PERMISSION_MODE", "default"),

  maxConcurrency: int("CLAUDE_PROXY_MAX_CONCURRENCY", 4),
  /** Ceiling on OpenAI `n`; each extra choice is another full CLI run. */
  maxChoices: int("CLAUDE_PROXY_MAX_CHOICES", 4),
  queueLimit: int("CLAUDE_PROXY_QUEUE_LIMIT", 32),
  requestTimeoutMs: int("CLAUDE_PROXY_TIMEOUT_MS", 10 * 60_000),

  /** The CLI has no max_tokens knob, so the proxy stops the stream itself. */
  enforceMaxTokens: bool("CLAUDE_PROXY_ENFORCE_MAX_TOKENS", true),

  /** Reuse a CLI session when a request continues a conversation we already ran. */
  sessionReuse: bool("CLAUDE_PROXY_SESSION_REUSE", true),
  sessionTtlMs: int("CLAUDE_PROXY_SESSION_TTL_MS", 30 * 60_000),
  sessionCacheMax: int("CLAUDE_PROXY_SESSION_CACHE_MAX", 200),

  /** Forward thinking blocks even when the request did not ask for them. */
  includeThinking: bool("CLAUDE_PROXY_INCLUDE_THINKING", false),
  /** Surface Claude Code's internal tool calls as `tool_use` blocks (agent mode only). */
  exposeToolBlocks: bool("CLAUDE_PROXY_EXPOSE_TOOL_BLOCKS", false),

  maxBodyBytes: int("CLAUDE_PROXY_MAX_BODY_BYTES", 32 * 1024 * 1024),
  logLevel: str("CLAUDE_PROXY_LOG_LEVEL", "info") as "debug" | "info" | "warn" | "error",

  /** Escape hatch: extra raw flags appended to every CLI invocation. */
  extraArgs: list("CLAUDE_PROXY_EXTRA_ARGS"),
};

export type Config = typeof config;

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };

export function log(level: keyof typeof LEVELS, msg: string, extra?: unknown): void {
  if (LEVELS[level] < LEVELS[config.logLevel]) return;
  const line = `${new Date().toISOString()} ${level.toUpperCase().padEnd(5)} ${msg}`;
  const sink = level === "error" || level === "warn" ? console.error : console.log;
  if (extra === undefined) sink(line);
  else sink(line, typeof extra === "string" ? extra : JSON.stringify(extra));
}
