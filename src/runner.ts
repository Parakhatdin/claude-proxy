import { spawn } from "node:child_process";
import { claudeEnv, config, log } from "./config.ts";
import { ApiError } from "./errors.ts";
import type {
  CliAssistantMessage,
  CliResultMessage,
  CliMessage,
  StreamEvent,
} from "./types.ts";

export type RunnerEvent =
  | { kind: "init"; sessionId: string; model: string }
  | { kind: "stream"; event: StreamEvent }
  | { kind: "assistant"; message: CliAssistantMessage["message"] }
  | { kind: "result"; result: CliResultMessage };

export interface RunOptions {
  args: string[];
  stdin: string;
  signal: AbortSignal;
}

const STDERR_CAP = 16 * 1024;

/**
 * The CLI reports failures as a `result` with `is_error: true` and a human-readable string
 * rather than a non-zero exit, so they must be turned into real HTTP errors. Otherwise a
 * logged-out CLI looks to the caller like a successful reply saying "Not logged in".
 */
function classifyCliError(text: string): ApiError {
  const detail = text.trim() || "no detail provided";
  if (/not logged in|\/login|invalid api key|authentication|unauthorized|oauth/i.test(detail)) {
    return new ApiError(
      "authentication_error",
      `The claude CLI is not authenticated (${detail}). Mint a long-lived token with ` +
        "`claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN (or CLAUDE_PROXY_OAUTH_TOKEN_FILE), " +
        "or run `claude` and sign in with /login, then restart the proxy.",
    );
  }
  if (/rate.?limit|usage limit|quota|too many requests/i.test(detail)) {
    return new ApiError("rate_limit_error", `The claude CLI hit a usage limit: ${detail}`);
  }
  if (/overloaded|capacity/i.test(detail)) {
    return new ApiError("overloaded_error", `The upstream model is overloaded: ${detail}`);
  }
  return new ApiError("api_error", `The claude CLI reported an error: ${detail}`);
}

/** Minimal pushable async queue: the reader awaits, the writer never blocks. */
class Pushable<T> {
  private readonly items: T[] = [];
  private wake: (() => void) | null = null;
  private done = false;
  private failure: unknown = null;

  push(item: T): void {
    this.items.push(item);
    this.wake?.();
  }

  end(err?: unknown): void {
    if (err !== undefined) this.failure = err;
    this.done = true;
    this.wake?.();
  }

  async *drain(): AsyncGenerator<T> {
    for (;;) {
      while (this.items.length > 0) yield this.items.shift()!;
      if (this.failure !== null) throw this.failure;
      if (this.done) return;
      await new Promise<void>((resolve) => {
        this.wake = () => {
          this.wake = null;
          resolve();
        };
      });
    }
  }
}

/**
 * Spawns `claude -p`, feeds it one stream-json turn and yields the decoded CLI messages.
 * Unrecognised message types (hook lifecycle, rate-limit notices, turn summaries) are dropped.
 */
export async function* runClaude(opts: RunOptions): AsyncGenerator<RunnerEvent> {
  const queue = new Pushable<RunnerEvent>();

  log("debug", "spawning claude", opts.args);
  const child = spawn(config.claudeBin, opts.args, {
    cwd: config.cwd,
    env: claudeEnv(),
    stdio: ["pipe", "pipe", "pipe"],
  });

  let settled = false;
  let sawResult = false;
  let stderrText = "";
  let timedOut = false;

  const finish = (err?: unknown): void => {
    if (settled) return;
    settled = true;
    queue.end(err);
  };

  const kill = (): void => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    child.kill("SIGTERM");
    setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    }, 2_000).unref();
  };

  const timer = setTimeout(() => {
    timedOut = true;
    kill();
  }, config.requestTimeoutMs);
  timer.unref();

  const onAbort = (): void => kill();
  opts.signal.addEventListener("abort", onAbort, { once: true });

  child.on("error", (err: NodeJS.ErrnoException) => {
    clearTimeout(timer);
    if (err.code === "ENOENT") {
      finish(
        new ApiError(
          "api_error",
          `Could not run "${config.claudeBin}". Install the Claude Code CLI or set CLAUDE_PROXY_CLAUDE_BIN to its path.`,
        ),
      );
    } else {
      finish(new ApiError("api_error", `Failed to start the claude CLI: ${err.message}`));
    }
  });

  let buffer = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buffer += chunk;
    let nl: number;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;

      let msg: CliMessage;
      try {
        msg = JSON.parse(line) as CliMessage;
      } catch {
        log("debug", "skipping unparseable CLI line", line.slice(0, 200));
        continue;
      }

      switch (msg.type) {
        case "system":
          if (msg.subtype === "init") {
            const init = msg as { session_id?: string; model?: string };
            queue.push({
              kind: "init",
              sessionId: init.session_id ?? "",
              model: init.model ?? "",
            });
          }
          break;
        case "stream_event":
          queue.push({ kind: "stream", event: (msg as { event: StreamEvent }).event });
          break;
        case "assistant":
          queue.push({ kind: "assistant", message: (msg as CliAssistantMessage).message });
          break;
        case "result": {
          sawResult = true;
          const result = msg as CliResultMessage;
          if (result.is_error) {
            finish(classifyCliError(result.result ?? result.subtype));
            return;
          }
          queue.push({ kind: "result", result });
          break;
        }
        default:
          break;
      }
    }
  });

  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => {
    if (stderrText.length < STDERR_CAP) stderrText += chunk;
  });

  child.on("close", (code, signal) => {
    clearTimeout(timer);
    if (timedOut) {
      finish(
        new ApiError(
          "timeout_error",
          `The claude CLI did not finish within ${config.requestTimeoutMs} ms.`,
        ),
      );
      return;
    }
    if (opts.signal.aborted) {
      finish(new ApiError("api_error", "Request aborted by the client."));
      return;
    }
    if (!sawResult) {
      const detail = stderrText.trim() || `exit code ${code ?? "null"}, signal ${signal ?? "none"}`;
      finish(new ApiError("api_error", `The claude CLI exited without a result: ${detail}`));
      return;
    }
    finish();
  });

  child.stdin.on("error", () => {
    /* The CLI may close stdin first; the close handler reports the real failure. */
  });
  child.stdin.end(opts.stdin);

  try {
    yield* queue.drain();
  } finally {
    opts.signal.removeEventListener("abort", onAbort);
    clearTimeout(timer);
    kill();
  }
}
