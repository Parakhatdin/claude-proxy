import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { timingSafeEqual } from "node:crypto";
import { config, log } from "./config.ts";
import { ApiError, badRequest, toApiError } from "./errors.ts";
import { runClaude, type RunnerEvent } from "./runner.ts";
import { SessionCache, sessions } from "./sessions.ts";
import {
  Limiter,
  StreamAssembler,
  collectMessage,
  synthesizeAnthropicStream,
  type RenderOptions,
} from "./assemble.ts";
import { applyToolCalls } from "./tools.ts";
import { estimateRequestTokens } from "./tokens.ts";
import {
  buildArgs,
  buildStdinPayload,
  prefixMessages,
  validate,
  type Validated,
} from "./translate.ts";
import {
  ChatChunkWriter,
  chatCompletionId,
  modelList,
  parseChatRequest,
  parseLegacyCompletion,
  synthesizeChatStream,
  toChatCompletion,
  toLegacyCompletion,
  toOpenAIError,
} from "./openai.ts";
import type { ContentBlock, MessageResponse, StreamEvent } from "./types.ts";

const MODELS = [
  "claude-opus-5",
  "claude-sonnet-5",
  "claude-fable-5-1",
  "claude-haiku-4-5-20251001",
  "opus",
  "sonnet",
  "haiku",
  "fable",
];

/** Which API dialect a route speaks; decides request parsing and error shape. */
type Dialect = "anthropic" | "openai";

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, OPTIONS",
  "access-control-allow-headers": "*",
};

/** Bounded concurrency: each in-flight request owns one `claude` process. */
class Semaphore {
  private active = 0;
  private readonly waiters: Array<() => void> = [];

  async acquire(): Promise<() => void> {
    if (this.active >= config.maxConcurrency) {
      if (this.waiters.length >= config.queueLimit) {
        throw new ApiError(
          "overloaded_error",
          `Proxy is at capacity (${config.maxConcurrency} concurrent, ${config.queueLimit} queued).`,
        );
      }
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
    this.active++;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.active--;
      this.waiters.shift()?.();
    };
  }
}

const semaphore = new Semaphore();

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    ...CORS,
    "content-type": "application/json",
    "content-length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

function sendError(res: ServerResponse, err: unknown, dialect: Dialect): void {
  const apiErr = toApiError(err);
  if (apiErr.status >= 500) log("error", `request failed: ${apiErr.message}`);
  else log("debug", `request rejected: ${apiErr.message}`);
  if (res.headersSent) {
    res.end();
    return;
  }
  sendJson(res, apiErr.status, dialect === "openai" ? toOpenAIError(apiErr) : apiErr.toJSON());
}

function readBody(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > config.maxBodyBytes) {
        reject(new ApiError("request_too_large", `Request body exceeds ${config.maxBodyBytes} bytes.`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw.trim()) {
        reject(badRequest("Request body is empty."));
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch (e) {
        reject(badRequest(`Request body is not valid JSON: ${(e as Error).message}`));
      }
    });
    req.on("error", reject);
  });
}

function constantTimeEquals(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

function authorize(req: IncomingMessage): void {
  if (!config.apiKey) return;
  const header = req.headers["x-api-key"];
  const auth = req.headers["authorization"];
  const presented =
    (typeof header === "string" ? header : undefined) ??
    (typeof auth === "string" && auth.toLowerCase().startsWith("bearer ")
      ? auth.slice(7).trim()
      : undefined);
  if (!presented || !constantTimeEquals(presented, config.apiKey)) {
    throw new ApiError("authentication_error", "Invalid API key.");
  }
}

/** Claims a resumable CLI session for this conversation prefix, if one is cached. */
function claimSession(v: Validated): { prefixKey: string | null; resumeId: string | null } {
  if (!config.sessionReuse || v.messages.length < 2) return { prefixKey: null, resumeId: null };
  const prefixKey = SessionCache.key(v.model, v.system, prefixMessages(v.messages));
  const resumeId = sessions.acquire(prefixKey);
  return { prefixKey: resumeId ? prefixKey : null, resumeId };
}

function rememberSession(
  v: Validated,
  prefixKey: string | null,
  sessionId: string,
  reply: ContentBlock[],
): void {
  if (!config.sessionReuse || !sessionId) {
    if (prefixKey) sessions.release(prefixKey);
    return;
  }
  const nextKey = SessionCache.key(v.model, v.system, [
    ...v.messages,
    { role: "assistant", content: reply },
  ]);
  sessions.commit(prefixKey, nextKey, sessionId);
}

type Shape = "chat" | "legacy" | "messages";

/** One CLI run: spawn, collect, and fold emulated tool calls back into the message. */
async function runOnce(
  v: Validated,
  args: string[],
  stdin: string,
  render: RenderOptions,
  signal: AbortSignal,
): Promise<{ message: MessageResponse; sessionId: string }> {
  const events: RunnerEvent[] = [];
  for await (const ev of runClaude({ args, stdin, signal })) events.push(ev);
  const { message, sessionId } = collectMessage(
    events,
    v.model,
    () => new Limiter(v.stopSequences, v.maxTokens),
    render,
  );
  return { message: v.tools.length > 0 ? applyToolCalls(message) : message, sessionId };
}

async function handleChat(req: IncomingMessage, res: ServerResponse, shape: Shape): Promise<void> {
  const body = await readBody(req);

  let v: Validated;
  let includeUsage = false;
  let n = 1;
  const chatId = chatCompletionId();
  if (shape === "messages") {
    v = validate(body);
  } else {
    const parsed = shape === "legacy" ? parseLegacyCompletion(body) : parseChatRequest(body);
    v = parsed.v;
    includeUsage = parsed.includeUsage;
    n = parsed.n;
  }

  // A tool call is only detectable once the whole reply is in hand, and a schema-constrained
  // reply arrives as a StructuredOutput tool call rather than as text. Neither can be streamed
  // live, so the reply is buffered and replayed as a synthetic stream instead.
  const buffered = v.tools.length > 0 || v.jsonSchema !== null;
  const live = v.stream && !buffered;

  const release = await semaphore.acquire();
  const controller = new AbortController();
  const onClose = (): void => controller.abort();
  res.on("close", onClose);

  // Resuming a session only makes sense for a single continuation of one conversation.
  const { prefixKey, resumeId } = n > 1 ? { prefixKey: null, resumeId: null } : claimSession(v);
  const args = buildArgs({ ...v, stream: live }, resumeId);
  const stdin = buildStdinPayload(v, resumeId !== null);
  const render: RenderOptions = { includeThinking: v.includeThinking };

  log(
    "info",
    `${shape} model=${v.model} stream=${v.stream}${buffered ? " (buffered)" : ""} tools=${v.tools.length} n=${n} resume=${resumeId ?? "none"}`,
  );

  try {
    if (live) {
      await streamResponse(res, v, args, stdin, new Limiter(v.stopSequences, v.maxTokens), render, prefixKey, controller.signal, shape, chatId, includeUsage);
      return;
    }

    const results: MessageResponse[] = [];
    let sessionId = "";
    for (let i = 0; i < n; i++) {
      const once = await runOnce(v, args, stdin, render, controller.signal);
      results.push(once.message);
      if (i === 0) sessionId = once.sessionId;
    }
    if (n === 1) rememberSession(v, prefixKey, sessionId, results[0]!.content);
    else if (prefixKey) sessions.release(prefixKey);

    if (v.stream) {
      writeBufferedStream(res, results[0]!, shape, chatId, includeUsage);
      return;
    }
    if (shape === "messages") sendJson(res, 200, results[0]);
    else if (shape === "legacy") sendJson(res, 200, toLegacyCompletion(results, chatId));
    else sendJson(res, 200, toChatCompletion(results, chatId));
  } catch (err) {
    if (prefixKey) sessions.release(prefixKey);
    throw err;
  } finally {
    res.off("close", onClose);
    release();
  }
}

/** Replays an already-complete message as SSE, for the buffered (tool-calling) path. */
function writeBufferedStream(
  res: ServerResponse,
  msg: MessageResponse,
  shape: Shape,
  chatId: string,
  includeUsage: boolean,
): void {
  res.writeHead(200, {
    ...CORS,
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache, no-store",
    connection: "keep-alive",
    "x-accel-buffering": "no",
  });
  if (shape === "messages") {
    for (const ev of synthesizeAnthropicStream(msg)) {
      res.write(`event: ${ev["type"]}\ndata: ${JSON.stringify(ev)}\n\n`);
    }
  } else {
    for (const frame of synthesizeChatStream(msg, chatId, includeUsage)) res.write(frame);
  }
  res.end();
}

async function streamResponse(
  res: ServerResponse,
  v: Validated,
  args: string[],
  stdin: string,
  limiter: Limiter,
  render: RenderOptions,
  prefixKey: string | null,
  signal: AbortSignal,
  shape: Shape,
  chatId: string,
  includeUsage: boolean,
): Promise<void> {
  res.writeHead(200, {
    ...CORS,
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache, no-store",
    connection: "keep-alive",
    "x-accel-buffering": "no",
  });

  const chunker = shape === "messages" ? null : new ChatChunkWriter(chatId, v.model, includeUsage);

  const write = (event: StreamEvent): void => {
    if (chunker) {
      for (const frame of chunker.handle(event)) res.write(frame);
      return;
    }
    res.write(`event: ${event["type"]}\ndata: ${JSON.stringify(event)}\n\n`);
  };

  // Some clients and reverse proxies drop an idle connection. Anthropic streams allow a
  // ping event; OpenAI streams have no such frame, so send an SSE comment instead.
  const ping = setInterval(() => {
    if (res.writableEnded) return;
    if (chunker) res.write(": keepalive\n\n");
    else res.write(`event: ping\ndata: ${JSON.stringify({ type: "ping" })}\n\n`);
  }, 15_000);
  ping.unref();

  const assembler = new StreamAssembler(v.model, limiter, render);
  const text: string[] = [];

  try {
    for await (const ev of runClaude({ args, stdin, signal })) {
      for (const out of assembler.handle(ev)) {
        if (out["type"] === "content_block_delta") {
          const delta = out["delta"] as { type?: string; text?: string };
          if (delta?.type === "text_delta") text.push(delta.text ?? "");
        }
        write(out);
      }
      // A stop sequence or the token budget ended the message early; stop reading the CLI.
      if (assembler.halted) break;
    }
    for (const out of assembler.finalize()) write(out);
    if (chunker) for (const frame of chunker.done()) res.write(frame);
    rememberSession(v, prefixKey, assembler.sessionId, [{ type: "text", text: text.join("") }]);
  } catch (err) {
    const apiErr = toApiError(err);
    log("error", `stream failed: ${apiErr.message}`);
    if (chunker) {
      res.write(`data: ${JSON.stringify(toOpenAIError(apiErr))}\n\n`);
      res.write("data: [DONE]\n\n");
    } else {
      res.write(`event: error\ndata: ${JSON.stringify({ type: "error", error: apiErr.toJSON().error })}\n\n`);
    }
    if (prefixKey) sessions.release(prefixKey);
  } finally {
    clearInterval(ping);
    res.end();
  }
}

async function handleCountTokens(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  const v = validate(body);
  sendJson(res, 200, { input_tokens: estimateRequestTokens(v.messages, v.system) });
}

export function createProxyServer() {
  return createServer((req, res) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    // Accept the path with or without the /v1 prefix: clients differ on whether the base
    // URL already includes it, and a missing prefix is otherwise a silent 404.
    const path = url.pathname.replace(/\/+$/, "").replace(/^\/v1(?=\/|$)/, "") || "/";
    const method = req.method ?? "GET";
    // Errors must come back in the dialect the caller speaks, not the route's shape.
    const dialect: Dialect =
      path === "/chat/completions" || path === "/completions" || path === "/embeddings"
        ? "openai"
        : "anthropic";

    void (async () => {
      try {
        if (method === "OPTIONS") {
          res.writeHead(204, CORS);
          res.end();
          return;
        }
        if (method === "GET" && (path === "/health" || path === "/")) {
          sendJson(res, 200, { status: "ok", mode: config.mode, sessions: sessions.size });
          return;
        }
        authorize(req);

        if (method === "GET" && path === "/models") {
          sendJson(res, 200, modelList(MODELS));
          return;
        }
        if (method === "POST" && path === "/chat/completions") {
          await handleChat(req, res, "chat");
          return;
        }
        if (method === "POST" && path === "/completions") {
          await handleChat(req, res, "legacy");
          return;
        }
        if (method === "POST" && path === "/messages") {
          await handleChat(req, res, "messages");
          return;
        }
        if (path === "/embeddings") {
          throw badRequest(
            "Embeddings are not available: the claude CLI exposes no embedding model. " +
              "Point your client at a dedicated embeddings provider for this call.",
          );
        }
        if (method === "POST" && path === "/messages/count_tokens") {
          await handleCountTokens(req, res);
          return;
        }
        throw new ApiError("not_found_error", `Unknown route: ${method} ${url.pathname}`);
      } catch (err) {
        sendError(res, err, dialect);
      }
    })();
  });
}
