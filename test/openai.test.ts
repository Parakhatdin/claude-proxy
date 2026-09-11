import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ChatChunkWriter,
  finishReason,
  modelList,
  parseChatRequest,
  toChatCompletion,
} from "../src/openai.ts";
import type { MessageResponse, StreamEvent } from "../src/types.ts";

/* ---------------------------- request mapping ---------------------------- */

test("parseChatRequest hoists system and developer turns into the system prompt", () => {
  const { v } = parseChatRequest({
    model: "sonnet",
    messages: [
      { role: "system", content: "Be terse." },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "developer", content: "Prefer metric units." },
      { role: "user", content: "how tall?" },
    ],
  });
  assert.equal(v.system, "Be terse.\n\nPrefer metric units.");
  assert.deepEqual(v.messages.map((m) => m.role), ["user", "assistant", "user"]);
  assert.deepEqual(v.messages[0]!.content, [{ type: "text", text: "hi" }]);
});

test("parseChatRequest falls back to the default model for non-Claude ids", () => {
  // Hermes ships "gpt-5.4" as its placeholder default; `claude --model gpt-5.4` would fail.
  assert.equal(parseChatRequest({ model: "gpt-5.4", messages: [{ role: "user", content: "x" }] }).v.model, "sonnet");
  assert.equal(parseChatRequest({ messages: [{ role: "user", content: "x" }] }).v.model, "sonnet");
  // Real Claude ids and aliases are passed through untouched.
  assert.equal(parseChatRequest({ model: "opus", messages: [{ role: "user", content: "x" }] }).v.model, "opus");
  assert.equal(
    parseChatRequest({ model: "claude-sonnet-4-6", messages: [{ role: "user", content: "x" }] }).v.model,
    "claude-sonnet-4-6",
  );
});

test("parseChatRequest normalises stop and the two max_tokens spellings", () => {
  assert.deepEqual(
    parseChatRequest({ messages: [{ role: "user", content: "x" }], stop: "END" }).v.stopSequences,
    ["END"],
  );
  assert.deepEqual(
    parseChatRequest({ messages: [{ role: "user", content: "x" }], stop: ["A", "B"] }).v.stopSequences,
    ["A", "B"],
  );
  assert.equal(
    parseChatRequest({ messages: [{ role: "user", content: "x" }], max_completion_tokens: 42 }).v.maxTokens,
    42,
  );
  assert.equal(
    parseChatRequest({ messages: [{ role: "user", content: "x" }], max_tokens: 7 }).v.maxTokens,
    7,
  );
  // Absent means unlimited, since OpenAI does not require the field.
  assert.equal(
    parseChatRequest({ messages: [{ role: "user", content: "x" }] }).v.maxTokens,
    Number.MAX_SAFE_INTEGER,
  );
});

test("parseChatRequest converts multipart content including data-URL images", () => {
  const { v } = parseChatRequest({
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "what is this?" },
          { type: "image_url", image_url: { url: "data:image/png;base64,QUJD" } },
          { type: "image_url", image_url: { url: "https://example.com/a.png" } },
        ],
      },
    ],
  });
  assert.deepEqual(v.messages[0]!.content, [
    { type: "text", text: "what is this?" },
    { type: "image", source: { type: "base64", media_type: "image/png", data: "QUJD" } },
    { type: "image", source: { type: "url", url: "https://example.com/a.png" } },
  ]);
});

test("parseChatRequest rejects what it genuinely cannot do", () => {
  assert.throws(() => parseChatRequest({ messages: [] }), /non-empty array/);
  assert.throws(() => parseChatRequest({ messages: [{ role: "system", content: "only" }] }), /at least one user/);
  assert.throws(() => parseChatRequest({ messages: [{ role: "nope", content: "x" }] }), /not a supported role/);
  // n interacts badly with streaming, and each extra choice costs a whole CLI run.
  assert.throws(
    () => parseChatRequest({ messages: [{ role: "user", content: "x" }], n: 2, stream: true }),
    /not supported together with `stream`/,
  );
  assert.throws(
    () => parseChatRequest({ messages: [{ role: "user", content: "x" }], n: 99 }),
    /may not exceed/,
  );
});

test("parseChatRequest never asks for thinking blocks", () => {
  // OpenAI responses have nowhere to put them.
  assert.equal(parseChatRequest({ messages: [{ role: "user", content: "x" }] }).v.includeThinking, false);
});

/* --------------------------- response mapping ---------------------------- */

function message(overrides: Partial<MessageResponse> = {}): MessageResponse {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-6",
    content: [{ type: "text", text: "Paris" }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 10, output_tokens: 3 },
    ...overrides,
  };
}

test("toChatCompletion produces a well-formed chat.completion", () => {
  const out = toChatCompletion([message()], "chatcmpl-x") as any;
  assert.equal(out.object, "chat.completion");
  assert.equal(out.id, "chatcmpl-x");
  assert.equal(out.model, "claude-sonnet-4-6");
  assert.deepEqual(out.choices[0].message, { role: "assistant", content: "Paris" });
  assert.equal(out.choices[0].finish_reason, "stop");
  assert.deepEqual(out.usage, { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 });
});

test("toChatCompletion concatenates text blocks and drops non-text ones", () => {
  const out = toChatCompletion(
    [message({ content: [{ type: "text", text: "a" }, { type: "thinking", thinking: "x" }, { type: "text", text: "b" }] })],
    "id",
  ) as any;
  assert.equal(out.choices[0].message.content, "ab");
});

test("finishReason maps the Anthropic stop reasons", () => {
  assert.equal(finishReason("end_turn"), "stop");
  assert.equal(finishReason("stop_sequence"), "stop");
  assert.equal(finishReason("max_tokens"), "length");
  assert.equal(finishReason("tool_use"), "tool_calls");
  assert.equal(finishReason(null), "stop");
});

/* ------------------------------- streaming ------------------------------- */

function frames(writer: ChatChunkWriter, events: StreamEvent[]): string[] {
  const out: string[] = [];
  for (const e of events) out.push(...writer.handle(e));
  out.push(...writer.done());
  return out;
}

function parseFrames(raw: string[]): any[] {
  return raw
    .filter((f) => f.startsWith("data: ") && !f.includes("[DONE]"))
    .map((f) => JSON.parse(f.slice(6).trim()));
}

test("ChatChunkWriter emits role, content, finish and [DONE] in order", () => {
  const raw = frames(new ChatChunkWriter("chatcmpl-1", "sonnet", false), [
    { type: "message_start", message: { model: "claude-sonnet-4-6", usage: { input_tokens: 5 } } },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hel" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "lo" } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { input_tokens: 5, output_tokens: 2 } },
    { type: "message_stop" },
  ]);

  assert.equal(raw[raw.length - 1], "data: [DONE]\n\n", "OpenAI clients wait for the sentinel");
  const parsed = parseFrames(raw);
  assert.equal(parsed[0].choices[0].delta.role, "assistant");
  assert.equal(parsed.every((f) => f.object === "chat.completion.chunk"), true);
  assert.equal(parsed.map((f) => f.choices[0]?.delta?.content ?? "").join(""), "Hello");
  const last = parsed[parsed.length - 1];
  assert.equal(last.choices[0].finish_reason, "stop");
  assert.deepEqual(last.choices[0].delta, {});
  // The model name is corrected to the one the CLI actually used.
  assert.equal(parsed[0].model, "claude-sonnet-4-6");
});

test("ChatChunkWriter reports length when the token budget truncated the reply", () => {
  const parsed = parseFrames(
    frames(new ChatChunkWriter("id", "sonnet", false), [
      { type: "message_start", message: {} },
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } },
      { type: "message_delta", delta: { stop_reason: "max_tokens" } },
    ]),
  );
  assert.equal(parsed[parsed.length - 1].choices[0].finish_reason, "length");
});

test("ChatChunkWriter appends a usage frame only when asked", () => {
  const events: StreamEvent[] = [
    { type: "message_start", message: { usage: { input_tokens: 11 } } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "x" } },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { input_tokens: 11, output_tokens: 4 } },
  ];
  const without = parseFrames(frames(new ChatChunkWriter("id", "sonnet", false), events));
  assert.equal(without.some((f) => f.usage), false);

  const withUsage = parseFrames(frames(new ChatChunkWriter("id", "sonnet", true), events));
  const usageFrame = withUsage.find((f) => f.usage);
  assert.deepEqual(usageFrame.usage, { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 });
  assert.deepEqual(usageFrame.choices, [], "the usage frame carries no choices");
});

test("ChatChunkWriter still emits a role delta if the CLI sent no message_start", () => {
  const parsed = parseFrames(
    frames(new ChatChunkWriter("id", "sonnet", false), [
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "x" } },
    ]),
  );
  assert.equal(parsed[0].choices[0].delta.role, "assistant");
});

/* -------------------------------- models --------------------------------- */

test("modelList satisfies both dialects from one path", () => {
  const list = modelList(["sonnet", "opus"]) as any;
  assert.equal(list.object, "list");
  assert.equal(list.has_more, false);
  const first = list.data[0];
  // OpenAI clients read object/owned_by; Anthropic clients read type/display_name.
  assert.equal(first.object, "model");
  assert.equal(first.owned_by, "anthropic");
  assert.equal(first.type, "model");
  assert.equal(first.display_name, "sonnet");
  assert.equal(first.id, "sonnet");
  assert.equal(typeof first.created, "number");
});
