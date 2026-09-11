import { test } from "node:test";
import assert from "node:assert/strict";

import { Limiter, StreamAssembler, collectMessage, type RenderOptions } from "../src/assemble.ts";
import { buildArgs, buildStdinPayload, validate } from "../src/translate.ts";
import { SessionCache } from "../src/sessions.ts";
import { estimateRequestTokens } from "../src/tokens.ts";
import type { RunnerEvent } from "../src/runner.ts";
import type { StreamEvent } from "../src/types.ts";

const NO_THINKING: RenderOptions = { includeThinking: false };
const UNLIMITED = Number.MAX_SAFE_INTEGER;

/* ------------------------------- Limiter ------------------------------- */

test("Limiter passes text through when no limit is hit", () => {
  const l = new Limiter([], UNLIMITED);
  assert.deepEqual(l.push("hello "), { emit: "hello ", halt: null });
  assert.deepEqual(l.push("world"), { emit: "world", halt: null });
  assert.equal(l.accumulated, "hello world");
});

test("Limiter cuts at a stop sequence and reports it", () => {
  const l = new Limiter(["STOP"], UNLIMITED);
  const r = l.push("keep this STOP drop this");
  assert.equal(r.emit, "keep this ");
  assert.deepEqual(r.halt, { reason: "stop_sequence", sequence: "STOP" });
});

test("Limiter finds a stop sequence split across two deltas", () => {
  const l = new Limiter(["<<END>>"], UNLIMITED);
  assert.equal(l.push("abc<<EN").halt, null);
  const r = l.push("D>>tail");
  assert.equal(r.emit, "");
  assert.deepEqual(r.halt, { reason: "stop_sequence", sequence: "<<END>>" });
  assert.equal(l.accumulated, "abc");
});

test("Limiter withholds a possible stop-sequence prefix, then releases it on flush", () => {
  const l = new Limiter(["<<END>>"], UNLIMITED);
  // "<<EN" could still become "<<END>>", so it must not reach the client yet.
  assert.deepEqual(l.push("abc<<EN"), { emit: "abc", halt: null });
  assert.equal(l.flush(), "<<EN", "the tail is released once no more deltas can arrive");
  assert.equal(l.accumulated, "abc<<EN");
});

test("Limiter flush yields nothing after a stop sequence halted the message", () => {
  const l = new Limiter(["STOP"], UNLIMITED);
  l.push("keep STOP rest");
  assert.equal(l.flush(), "");
  assert.equal(l.accumulated, "keep ");
});

test("Limiter ignores stop-sequence prefixes that turn out not to match", () => {
  const l = new Limiter(["FOO"], UNLIMITED);
  assert.equal(l.push("bar F").emit, "bar ");
  assert.equal(l.push("izz").emit, "Fizz");
  assert.equal(l.flush(), "");
  assert.equal(l.accumulated, "bar Fizz");
});

test("Limiter picks the earliest of several stop sequences", () => {
  const l = new Limiter(["ZZ", "B"], UNLIMITED);
  const r = l.push("aaBccZZ");
  assert.equal(r.emit, "aa");
  assert.equal(r.halt?.sequence, "B");
});

test("Limiter truncates on the max_tokens budget", () => {
  const l = new Limiter([], 2); // 2 * 3.8 => 7 characters
  const r = l.push("0123456789");
  assert.equal(r.emit, "0123456");
  assert.deepEqual(r.halt, { reason: "max_tokens" });
});

test("Limiter prefers a stop sequence that precedes the token budget", () => {
  const l = new Limiter(["X"], 100);
  const r = l.push("abXcd");
  assert.equal(r.halt?.reason, "stop_sequence");
});

/* ------------------------------- validate ------------------------------ */

test("validate accepts Anthropic tool declarations and their choice", () => {
  const v = validate({
    messages: [{ role: "user", content: "hi" }],
    tools: [{ name: "get_weather", description: "Look up weather", input_schema: { type: "object" } }],
    tool_choice: { type: "any" },
  });
  assert.equal(v.tools.length, 1);
  assert.equal(v.tools[0]!.name, "get_weather");
  assert.deepEqual(v.tools[0]!.parameters, { type: "object" });
  assert.deepEqual(v.toolChoice, { mode: "required" });
});

test("validate accepts an explicitly empty tools array", () => {
  const v = validate({ messages: [{ role: "user", content: "hi" }], tools: [] });
  assert.equal(v.messages.length, 1);
});

test("validate rejects malformed requests", () => {
  assert.throws(() => validate({ messages: [] }), /non-empty array/);
  assert.throws(() => validate({ messages: [{ role: "system", content: "x" }] }), /role must be/);
  assert.throws(() => validate({ messages: [{ role: "user", content: 7 }] }), /content must be/);
  assert.throws(
    () => validate({ messages: [{ role: "user", content: "x" }], max_tokens: 0 }),
    /positive integer/,
  );
  assert.throws(() => validate("nope"), /must be a JSON object/);
});

test("validate flattens a block-form system prompt", () => {
  const v = validate({
    messages: [{ role: "user", content: "hi" }],
    system: [{ type: "text", text: "line one" }, { type: "text", text: "line two" }],
  });
  assert.equal(v.system, "line one\nline two");
});

test("validate keeps thinking blocks hidden unless requested", () => {
  assert.equal(validate({ messages: [{ role: "user", content: "hi" }] }).includeThinking, false);
  const v = validate({
    messages: [{ role: "user", content: "hi" }],
    thinking: { type: "enabled", budget_tokens: 1024 },
  });
  assert.equal(v.includeThinking, true);
});

/* ------------------------------- buildArgs ----------------------------- */

test("buildArgs isolates the CLI from local Claude Code config in clean mode", () => {
  const v = validate({ model: "sonnet", messages: [{ role: "user", content: "hi" }], system: "S" });
  const args = buildArgs(v, null);
  for (const flag of ["--print", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands"]) {
    assert.ok(args.includes(flag), `expected ${flag}`);
  }
  assert.equal(args[args.indexOf("--system-prompt") + 1], "S");
  assert.equal(args[args.indexOf("--tools") + 1], "");
  assert.ok(!args.includes("--include-partial-messages"), "no partial messages when not streaming");
});

test("buildArgs adds partial messages for streaming and resume for a cached session", () => {
  const v = validate({ messages: [{ role: "user", content: "hi" }], stream: true });
  const args = buildArgs(v, "sess-123");
  assert.ok(args.includes("--include-partial-messages"));
  assert.equal(args[args.indexOf("--resume") + 1], "sess-123");
});

/* ---------------------------- buildStdinPayload ------------------------ */

function parsePayload(raw: string): { role: string; content: Array<Record<string, unknown>> } {
  const lines = raw.trim().split("\n");
  assert.equal(lines.length, 1, "exactly one turn is fed to the CLI");
  return JSON.parse(lines[0]!).message;
}

test("buildStdinPayload sends a lone user turn verbatim", () => {
  const v = validate({ messages: [{ role: "user", content: "just this" }] });
  const msg = parsePayload(buildStdinPayload(v, false));
  assert.deepEqual(msg.content, [{ type: "text", text: "just this" }]);
});

test("buildStdinPayload replays history as one transcript on a cold start", () => {
  const v = validate({
    messages: [
      { role: "user", content: "first" },
      { role: "assistant", content: "reply" },
      { role: "user", content: "second" },
    ],
  });
  const msg = parsePayload(buildStdinPayload(v, false));
  const text = String(msg.content[0]!["text"]);
  assert.match(text, /<transcript>/);
  assert.match(text, /Human: first/);
  assert.match(text, /Assistant: reply/);
  // The newest turn stays a separate block so images in it survive.
  assert.deepEqual(msg.content[1], { type: "text", text: "second" });
  assert.ok(!text.includes("second"), "the newest turn is not duplicated into the transcript");
});

test("buildStdinPayload sends only the newest turn when resuming", () => {
  const v = validate({
    messages: [
      { role: "user", content: "first" },
      { role: "assistant", content: "reply" },
      { role: "user", content: "second" },
    ],
  });
  const msg = parsePayload(buildStdinPayload(v, true));
  assert.deepEqual(msg.content, [{ type: "text", text: "second" }]);
});

test("buildStdinPayload asks the model to continue an assistant prefill", () => {
  const v = validate({
    messages: [
      { role: "user", content: "write a poem" },
      { role: "assistant", content: "Roses are" },
    ],
  });
  const text = String(parsePayload(buildStdinPayload(v, false)).content[0]!["text"]);
  assert.match(text, /Continue the final Assistant turn/);
  assert.match(text, /Assistant: Roses are/);
});

test("buildStdinPayload preserves image blocks in the newest turn", () => {
  const image = { type: "image", source: { type: "base64", media_type: "image/png", data: "AAA" } };
  const v = validate({ messages: [{ role: "user", content: [image, { type: "text", text: "what?" }] }] });
  const msg = parsePayload(buildStdinPayload(v, false));
  assert.deepEqual(msg.content[0], image);
});

/* ----------------------------- SessionCache ---------------------------- */

test("SessionCache treats a bare string and a single text block as the same turn", () => {
  const a = SessionCache.key("sonnet", "sys", [{ role: "user", content: "hi" }]);
  const b = SessionCache.key("sonnet", "sys", [
    { role: "user", content: [{ type: "text", text: "hi" }] },
  ]);
  assert.equal(a, b);
});

test("SessionCache keys separate different models and system prompts", () => {
  const msgs = [{ role: "user" as const, content: "hi" }];
  assert.notEqual(SessionCache.key("sonnet", "a", msgs), SessionCache.key("opus", "a", msgs));
  assert.notEqual(SessionCache.key("sonnet", "a", msgs), SessionCache.key("sonnet", "b", msgs));
});

test("SessionCache hands a session to one caller at a time", () => {
  const cache = new SessionCache();
  cache.commit(null, "k1", "sess-1");
  assert.equal(cache.acquire("k1"), "sess-1");
  // A second concurrent turn must not resume the same CLI session.
  assert.equal(cache.acquire("k1"), null);
  cache.release("k1");
  assert.equal(cache.acquire("k1"), "sess-1");
});

test("SessionCache advances a session to the new prefix and drops the old one", () => {
  const cache = new SessionCache();
  cache.commit(null, "k1", "sess-1");
  cache.acquire("k1");
  cache.commit("k1", "k2", "sess-1");
  assert.equal(cache.acquire("k1"), null, "the stale prefix no longer matches");
  assert.equal(cache.acquire("k2"), "sess-1");
});

test("SessionCache misses on an unknown prefix", () => {
  assert.equal(new SessionCache().acquire("nope"), null);
});

/* ------------------------------- assembly ------------------------------ */

function streamEv(event: StreamEvent): RunnerEvent {
  return { kind: "stream", event };
}

function collectStream(events: RunnerEvent[], opts = NO_THINKING): StreamEvent[] {
  const a = new StreamAssembler("sonnet", new Limiter([], UNLIMITED), opts);
  const out: StreamEvent[] = [];
  for (const ev of events) {
    out.push(...a.handle(ev));
    if (a.halted) break;
  }
  out.push(...a.finalize());
  return out;
}

test("StreamAssembler merges two CLI turns into one message with continuous indices", () => {
  const out = collectStream([
    { kind: "init", sessionId: "s1", model: "claude-sonnet-4-6" },
    streamEv({ type: "message_start", message: { id: "msg_1", model: "claude-sonnet-4-6" } }),
    streamEv({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
    streamEv({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "one" } }),
    streamEv({ type: "content_block_stop", index: 0 }),
    streamEv({ type: "message_delta", delta: { stop_reason: "end_turn" } }),
    streamEv({ type: "message_stop" }),
    // Second CLI turn restarts its own indices at 0.
    streamEv({ type: "message_start", message: { id: "msg_2" } }),
    streamEv({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
    streamEv({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "two" } }),
    streamEv({ type: "content_block_stop", index: 0 }),
    { kind: "result", result: { type: "result", subtype: "success", is_error: false, session_id: "s1", stop_reason: "end_turn" } },
  ]);

  const types = out.map((e) => e["type"]);
  assert.equal(types.filter((t) => t === "message_start").length, 1);
  assert.equal(types.filter((t) => t === "message_stop").length, 1);
  assert.equal(types.filter((t) => t === "message_delta").length, 1);
  assert.equal(types[types.length - 1], "message_stop");

  const starts = out.filter((e) => e["type"] === "content_block_start");
  assert.deepEqual(starts.map((e) => e["index"]), [0, 1], "second turn is remapped to index 1");
  assert.equal((out[0]!["message"] as { id: string }).id, "msg_1");
});

test("StreamAssembler drops thinking blocks unless asked, and keeps them when asked", () => {
  const events = [
    streamEv({ type: "message_start", message: { id: "m" } }),
    streamEv({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } }),
    streamEv({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "hmm" } }),
    streamEv({ type: "content_block_stop", index: 0 }),
    streamEv({ type: "content_block_start", index: 1, content_block: { type: "text", text: "" } }),
    streamEv({ type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "answer" } }),
    streamEv({ type: "content_block_stop", index: 1 }),
  ];

  const hidden = collectStream(events);
  assert.equal(hidden.filter((e) => e["type"] === "content_block_start").length, 1);
  assert.deepEqual(
    hidden.filter((e) => e["type"] === "content_block_start")[0]!["content_block"],
    { type: "text", text: "" },
  );
  // The surviving text block is renumbered to 0 so the client sees a gapless sequence.
  assert.equal(hidden.filter((e) => e["type"] === "content_block_start")[0]!["index"], 0);

  const shown = collectStream(events, { includeThinking: true });
  assert.equal(shown.filter((e) => e["type"] === "content_block_start").length, 2);
});

test("StreamAssembler closes the message itself when a stop sequence fires", () => {
  const a = new StreamAssembler("sonnet", new Limiter(["HALT"], UNLIMITED), NO_THINKING);
  const out: StreamEvent[] = [];
  out.push(...a.handle(streamEv({ type: "message_start", message: { id: "m" } })));
  out.push(...a.handle(streamEv({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })));
  out.push(...a.handle(streamEv({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "abcHALTdef" } })));

  assert.equal(a.halted, true);
  const types = out.map((e) => e["type"]);
  assert.deepEqual(types.slice(-3), ["content_block_stop", "message_delta", "message_stop"]);
  const delta = out.find((e) => e["type"] === "message_delta")!;
  assert.deepEqual(delta["delta"], { stop_reason: "stop_sequence", stop_sequence: "HALT" });
  assert.equal(out.filter((e) => e["type"] === "content_block_delta")[0]!["delta"]["text"], "abc");
  assert.deepEqual(a.finalize(), [], "finalize is idempotent after an early halt");
});

test("StreamAssembler flushes withheld text before the block closes", () => {
  const a = new StreamAssembler("sonnet", new Limiter(["<<END>>"], UNLIMITED), NO_THINKING);
  const out: StreamEvent[] = [];
  out.push(...a.handle(streamEv({ type: "message_start", message: { id: "m" } })));
  out.push(...a.handle(streamEv({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })));
  out.push(...a.handle(streamEv({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi <<EN" } })));
  out.push(...a.handle(streamEv({ type: "content_block_stop", index: 0 })));
  out.push(...a.finalize());

  const text = out
    .filter((e) => e["type"] === "content_block_delta")
    .map((e) => (e["delta"] as { text: string }).text)
    .join("");
  assert.equal(text, "hi <<EN", "no characters are lost when the sequence never completes");
  const types = out.map((e) => e["type"]);
  assert.ok(
    types.lastIndexOf("content_block_delta") < types.indexOf("content_block_stop"),
    "the flushed delta precedes content_block_stop",
  );
});

test("StreamAssembler still emits a well-formed message when the CLI produced nothing", () => {
  const out = collectStream([{ kind: "init", sessionId: "s", model: "m" }]);
  assert.deepEqual(out.map((e) => e["type"]), ["message_start", "message_delta", "message_stop"]);
});

test("collectMessage merges turns, applies limits and reports usage", () => {
  const { message, sessionId } = collectMessage(
    [
      { kind: "init", sessionId: "s9", model: "claude-sonnet-4-6" },
      { kind: "assistant", message: { id: "msg_x", model: "claude-sonnet-4-6", role: "assistant", content: [{ type: "thinking", thinking: "quiet" }, { type: "text", text: "visible" }], stop_reason: null, stop_sequence: null, usage: { input_tokens: 11 } } },
      { kind: "result", result: { type: "result", subtype: "success", is_error: false, session_id: "s9", stop_reason: "end_turn", usage: { input_tokens: 11, output_tokens: 3 } } },
    ],
    "sonnet",
    () => new Limiter([], UNLIMITED),
    NO_THINKING,
  );

  assert.equal(sessionId, "s9");
  assert.equal(message.id, "msg_x");
  assert.equal(message.model, "claude-sonnet-4-6");
  assert.deepEqual(message.content, [{ type: "text", text: "visible" }]);
  assert.equal(message.stop_reason, "end_turn");
  assert.equal(message.usage.input_tokens, 11);
  assert.equal(message.usage.output_tokens, 3);
});

test("collectMessage never returns an empty content array", () => {
  const { message } = collectMessage([], "sonnet", () => new Limiter([], UNLIMITED), NO_THINKING);
  assert.deepEqual(message.content, [{ type: "text", text: "" }]);
});

/* -------------------------------- tokens ------------------------------- */

test("estimateRequestTokens grows with the conversation and counts images", () => {
  const one = estimateRequestTokens([{ role: "user", content: "hello" }], undefined);
  const two = estimateRequestTokens(
    [{ role: "user", content: "hello" }, { role: "assistant", content: "hi there" }],
    "be terse",
  );
  assert.ok(two > one);
  const withImage = estimateRequestTokens(
    [{ role: "user", content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: "x" } }] }],
    undefined,
  );
  assert.ok(withImage > 1000, "images carry a fixed cost");
});
