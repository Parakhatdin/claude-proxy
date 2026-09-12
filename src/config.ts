import { readFileSync } from "node:fs";
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

interface ResolvedToken {
  token: string;
  /** Where it came from, for the startup log. Never the token itself. */
  source: string;
}

/**
 * The long-lived CLI credential minted by `claude setup-token` — an `sk-ant-oat01-…` value
 * that lasts a year. Prefer the file form: a launchd or systemd unit can point at a secrets
 * file instead of keeping the token in a `.env` that is easy to read or to commit. Resolved
 * once at startup, so rotating the token means restarting the proxy.
 */
function resolveOauthToken(): ResolvedToken | undefined {
  const inline = optStr("CLAUDE_CODE_OAUTH_TOKEN");
  const file = optStr("CLAUDE_PROXY_OAUTH_TOKEN_FILE");

  if (inline !== undefined && file !== undefined) {
    throw new Error(
      "Set CLAUDE_CODE_OAUTH_TOKEN or CLAUDE_PROXY_OAUTH_TOKEN_FILE, not both.",
    );
  }
  if (inline !== undefined) return { token: inline.trim(), source: "CLAUDE_CODE_OAUTH_TOKEN" };
  if (file === undefined) return undefined;

  let raw: string;
  try {
    raw = readFileSync(file, "utf8");
  } catch (err) {
    throw new Error(
      `CLAUDE_PROXY_OAUTH_TOKEN_FILE (${file}) could not be read: ${(err as Error).message}`,
    );
  }
  const token = raw.trim();
  if (!token) throw new Error(`CLAUDE_PROXY_OAUTH_TOKEN_FILE (${file}) is empty.`);
  return { token, source: `CLAUDE_PROXY_OAUTH_TOKEN_FILE (${file})` };
}

const oauth = resolveOauthToken();

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

  /** Long-lived `claude setup-token` credential handed to every spawned CLI, if configured. */
  oauthToken: oauth?.token,
  /** Human-readable description of which credential the CLI will use. Safe to log. */
  authSource:
    oauth?.source ??
    (process.env.ANTHROPIC_API_KEY ? "ANTHROPIC_API_KEY" : "the claude CLI's own login"),

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

let cachedEnv: NodeJS.ProcessEnv | null = null;

/**
 * Environment for the spawned CLI. When the proxy holds a token of its own, the API-key
 * variables are dropped: the CLI would otherwise have two credentials to pick between, and
 * which one wins is not something to leave to chance in a long-running service.
 */
export function claudeEnv(): NodeJS.ProcessEnv {
  if (cachedEnv !== null) return cachedEnv;
  const env: NodeJS.ProcessEnv = { ...process.env };
  if (config.oauthToken !== undefined) {
    env.CLAUDE_CODE_OAUTH_TOKEN = config.oauthToken;
    delete env.ANTHROPIC_API_KEY;
    delete env.ANTHROPIC_AUTH_TOKEN;
  }
  cachedEnv = env;
  return env;
}

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };

export function log(level: keyof typeof LEVELS, msg: string, extra?: unknown): void {
  if (LEVELS[level] < LEVELS[config.logLevel]) return;
  const line = `${new Date().toISOString()} ${level.toUpperCase().padEnd(5)} ${msg}`;
  const sink = level === "error" || level === "warn" ? console.error : console.log;
  if (extra === undefined) sink(line);
  else sink(line, typeof extra === "string" ? extra : JSON.stringify(extra));
}
