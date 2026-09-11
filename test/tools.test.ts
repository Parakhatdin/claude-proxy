import { test } from "node:test";
import assert from "node:assert/strict";

import {
  applyToolCalls,
  buildToolPrompt,
  normalizeToolChoice,
  normalizeTools,
  parseToolCalls,
  renderToolBlock,
} from "../src/tools.ts";
import {
  parseChatRequest,
  parseLegacyCompletion,
  synthesizeChatStream,
  toChatCompletion,
  toLegacyCompletion,
  toolCallsOf,
} from "../src/openai.ts";
import { buildArgs, buildStdinPayload, effectiveSystem, validate } from "../src/translate.ts";
import type { MessageResponse } from "../src/types.ts";

const WEATHER = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Current weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  },
};

/* ----------------------------- normalisation ----------------------------- */

test("normalizeTools reads both the nested and flat OpenAI forms", () => {
  const nested = normalizeTools([WEATHER], "openai");
  const flat = normalizeTools([WEATHER.function], "openai");
  assert.deepEqual(nested, flat);
  assert.equal(nested[0]!.name, "get_weather");
  assert.deepEqual(nested[0]!.parameters, WEATHER.function.parameters);
});

test("normalizeTools reads Anthropic input_schema and skips server-side tools", () => {
  const tools = normalizeTools(
    [
      { name: "lookup", description: "d", input_schema: { type: "object" } },
      // A server tool has a type but no schema for the model to fill in.
      { type: "web_search_20250305", name: "web_search" },
    ],
    "anthropic",
  );
  assert.equal(tools.length, 1);
  assert.equal(tools[0]!.name, "lookup");
});

test("normalizeTools rejects malformed declarations", () => {
  assert.throws(() => normalizeTools({}, "openai"), /must be an array/);
  assert.throws(() => normalizeTools([{ function: {} }], "openai"), /missing a function name/);
  assert.throws(() => normalizeTools([{ description: "d" }], "anthropic"), /name is required/);
});

test("normalizeToolChoice maps both dialects", () => {
  assert.deepEqual(normalizeToolChoice(undefined, "openai"), { mode: "auto" });
  assert.deepEqual(normalizeToolChoice("none", "openai"), { mode: "none" });
  assert.deepEqual(normalizeToolChoice("required", "openai"), { mode: "required" });
  assert.deepEqual(normalizeToolChoice({ type: "function", function: { name: "f" } }, "openai"), {
    mode: "function",
    name: "f",
  });
  assert.deepEqual(normalizeToolChoice({ type: "any" }, "anthropic"), { mode: "required" });
  assert.deepEqual(normalizeToolChoice({ type: "tool", name: "f" }, "anthropic"), {
    mode: "function",
    name: "f",
  });
});

/* ------------------------------- prompting ------------------------------- */

test("buildToolPrompt states the contract and embeds the schemas", () => {
  const tools = normalizeTools([WEATHER], "openai");
  const prompt = buildToolPrompt(tools, { mode: "auto" });
  assert.match(prompt, /<tool_call>/);
  assert.match(prompt, /get_weather/);
  assert.match(prompt, /answer normally and emit no tool call/);
  // The full parameter schema has to reach the model, not just the tool name.
  assert.match(prompt, /"city"/);
  assert.match(prompt, /"required"/);
  assert.match(prompt, /Current weather for a city/);
});

test("buildToolPrompt reflects the tool choice, and none means no prompt", () => {
  const tools = normalizeTools([WEATHER], "openai");
  assert.match(buildToolPrompt(tools, { mode: "required" }), /MUST emit a tool call/);
  assert.match(buildToolPrompt(tools, { mode: "function", name: "get_weather" }), /MUST call the tool "get_weather"/);
  assert.equal(buildToolPrompt(tools, { mode: "none" }), "");
  assert.equal(buildToolPrompt([], { mode: "auto" }), "");
});

test("effectiveSystem appends the tool contract to the caller's system prompt", () => {
  const v = validate({
    messages: [{ role: "user", content: "hi" }],
    system: "You are terse.",
    tools: [{ name: "f", input_schema: { type: "object" } }],
  });
  const sys = effectiveSystem(v);
  assert.match(sys, /^You are terse\./);
  assert.match(sys, /# Tool calling/);
});

/* -------------------------------- parsing -------------------------------- */

test("parseToolCalls extracts a single call", () => {
  const { calls, text } = parseToolCalls('<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>');
  assert.equal(calls.length, 1);
  assert.equal(calls[0]!.name, "get_weather");
  assert.equal(calls[0]!.arguments, '{"city":"Paris"}');
  assert.match(calls[0]!.id, /^call_/);
  assert.equal(text, "");
});

test("parseToolCalls extracts several parallel calls", () => {
  const { calls } = parseToolCalls(
    '<tool_call>{"name":"a","arguments":{}}</tool_call>\n<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>',
  );
  assert.deepEqual(calls.map((c) => c.name), ["a", "b"]);
  assert.notEqual(calls[0]!.id, calls[1]!.id, "each call gets its own id");
});

test("parseToolCalls tolerates markdown fences and stray prose", () => {
  const { calls, text } = parseToolCalls(
    'Let me check.\n<tool_call>```json\n{"name":"a","arguments":{"x":1}}\n```</tool_call>',
  );
  assert.equal(calls.length, 1);
  assert.equal(calls[0]!.arguments, '{"x":1}');
  assert.equal(text, "Let me check.");
});

test("parseToolCalls leaves malformed blocks as literal text", () => {
  const { calls, text } = parseToolCalls("<tool_call>not json</tool_call>");
  assert.equal(calls.length, 0);
  assert.match(text, /not json/, "the caller can see what the model actually produced");
});

test("parseToolCalls ignores a plain reply", () => {
  const { calls, text } = parseToolCalls("The weather in Paris is mild.");
  assert.equal(calls.length, 0);
  assert.equal(text, "The weather in Paris is mild.");
});

/* ------------------------------- application ----------------------------- */

function message(text: string): MessageResponse {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-6",
    content: [{ type: "text", text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 10, output_tokens: 5 },
  };
}

test("applyToolCalls turns an emulated call into a tool_use block", () => {
  const out = applyToolCalls(message('<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'));
  assert.equal(out.stop_reason, "tool_use");
  assert.equal(out.content.length, 1);
  assert.deepEqual(out.content[0], {
    type: "tool_use",
    id: (out.content[0] as any).id,
    name: "get_weather",
    input: { city: "Paris" },
  });
});

test("applyToolCalls keeps surrounding prose as a text block", () => {
  const out = applyToolCalls(message('Checking.\n<tool_call>{"name":"a","arguments":{}}</tool_call>'));
  assert.deepEqual(out.content[0], { type: "text", text: "Checking." });
  assert.equal(out.content[1]!.type, "tool_use");
});

test("applyToolCalls leaves an ordinary reply untouched", () => {
  const original = message("Just an answer.");
  const out = applyToolCalls(original);
  assert.equal(out.stop_reason, "end_turn");
  assert.deepEqual(out.content, original.content);
});

/* ------------------------- transcript round trip ------------------------- */

test("renderToolBlock renders calls and results for the transcript", () => {
  assert.equal(
    renderToolBlock({ type: "tool_use", id: "c1", name: "a", input: { x: 1 } }),
    '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>',
  );
  assert.equal(renderToolBlock({ type: "tool_result", tool_use_id: "c1", content: "18C" }), "<tool_result>18C</tool_result>");
  assert.equal(
    renderToolBlock({ type: "tool_result", tool_use_id: "c1", content: [{ type: "text", text: "ok" }] }),
    "<tool_result>ok</tool_result>",
  );
  assert.match(
    String(renderToolBlock({ type: "tool_result", tool_use_id: "c1", content: "boom", is_error: true })),
    /is_error="true"/,
  );
  assert.equal(renderToolBlock({ type: "text", text: "x" }), null);
});

test("an OpenAI tool round trip survives into the CLI transcript", () => {
  const { v } = parseChatRequest({
    model: "sonnet",
    tools: [WEATHER],
    messages: [
      { role: "user", content: "Weather in Paris?" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "18C and clear" },
    ],
  });

  assert.deepEqual(v.messages.map((m) => m.role), ["user", "assistant", "user"]);
  assert.deepEqual(v.messages[1]!.content, [
    { type: "tool_use", id: "call_1", name: "get_weather", input: { city: "Paris" } },
  ]);
  assert.deepEqual(v.messages[2]!.content, [
    { type: "tool_result", tool_use_id: "call_1", content: "18C and clear" },
  ]);

  // The CLI only accepts text and images, so tool turns are rendered into the transcript.
  const payload = JSON.parse(buildStdinPayload(v, false).trim()).message;
  // The prior turns become a transcript block; the newest turn (the results) follows it.
  const text = payload.content.map((b: any) => b.text ?? "").join("\n");
  assert.match(text, /<tool_call>\{"name":"get_weather"/);
  assert.match(text, /<tool_result>18C and clear<\/tool_result>/);
});

test("consecutive tool results merge into one user turn", () => {
  const { v } = parseChatRequest({
    messages: [
      { role: "user", content: "go" },
      { role: "assistant", tool_calls: [
        { id: "c1", type: "function", function: { name: "a", arguments: "{}" } },
        { id: "c2", type: "function", function: { name: "b", arguments: "{}" } },
      ] },
      { role: "tool", tool_call_id: "c1", content: "r1" },
      { role: "tool", tool_call_id: "c2", content: "r2" },
    ],
  });
  assert.equal(v.messages.length, 3);
  assert.equal(v.messages[2]!.content.length, 2, "both results land in a single turn");
});

test("malformed tool_call arguments are preserved rather than dropped", () => {
  const { v } = parseChatRequest({
    messages: [
      { role: "user", content: "go" },
      { role: "assistant", tool_calls: [{ id: "c1", type: "function", function: { name: "a", arguments: "not json" } }] },
    ],
  });
  assert.deepEqual((v.messages[1]!.content as any)[0].input, { _raw: "not json" });
});

/* ------------------------------- responses ------------------------------- */

function withCall(): MessageResponse {
  return applyToolCalls(message('<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'));
}

test("toChatCompletion emits tool_calls with a null content and the right finish_reason", () => {
  const out = toChatCompletion([withCall()], "chatcmpl-1") as any;
  const choice = out.choices[0];
  assert.equal(choice.finish_reason, "tool_calls");
  assert.equal(choice.message.content, null);
  assert.equal(choice.message.tool_calls.length, 1);
  assert.equal(choice.message.tool_calls[0].type, "function");
  assert.equal(choice.message.tool_calls[0].function.name, "get_weather");
  assert.equal(choice.message.tool_calls[0].function.arguments, '{"city":"Paris"}');
});

test("toChatCompletion returns one choice per n and sums their usage", () => {
  const out = toChatCompletion([message("a"), message("b")], "id") as any;
  assert.deepEqual(out.choices.map((c: any) => c.index), [0, 1]);
  assert.deepEqual(out.choices.map((c: any) => c.message.content), ["a", "b"]);
  assert.equal(out.usage.completion_tokens, 10);
  assert.equal(out.usage.prompt_tokens, 20);
});

test("toolCallsOf ignores non-tool blocks", () => {
  assert.deepEqual(toolCallsOf([{ type: "text", text: "x" }]), []);
});

test("synthesizeChatStream replays a buffered tool call as OpenAI frames", () => {
  const raw = synthesizeChatStream(withCall(), "chatcmpl-1", false);
  assert.equal(raw[raw.length - 1], "data: [DONE]\n\n");
  const frames = raw.filter((f) => !f.includes("[DONE]")).map((f) => JSON.parse(f.slice(6)));
  assert.equal(frames[0].choices[0].delta.role, "assistant");
  const callFrame = frames.find((f: any) => f.choices[0]?.delta?.tool_calls);
  assert.equal(callFrame.choices[0].delta.tool_calls[0].index, 0);
  assert.equal(callFrame.choices[0].delta.tool_calls[0].function.name, "get_weather");
  assert.equal(frames[frames.length - 1].choices[0].finish_reason, "tool_calls");
});

/* ---------------------------- response_format ---------------------------- */

test("response_format json_schema reaches the CLI as --json-schema", () => {
  const schema = { type: "object", properties: { city: { type: "string" } } };
  const { v } = parseChatRequest({
    messages: [{ role: "user", content: "x" }],
    response_format: { type: "json_schema", json_schema: { schema } },
  });
  assert.deepEqual(v.jsonSchema, schema);
  assert.equal(v.jsonMode, true);
  const args = buildArgs(v, null);
  assert.equal(args[args.indexOf("--json-schema") + 1], JSON.stringify(schema));
});

test("response_format json_object instructs the model instead", () => {
  const { v } = parseChatRequest({
    messages: [{ role: "user", content: "x" }],
    response_format: { type: "json_object" },
  });
  assert.equal(v.jsonSchema, null);
  assert.match(effectiveSystem(v), /single valid JSON value/);
});

test("a schema is dropped when tools are in play, since both claim the whole reply", () => {
  const { v } = parseChatRequest({
    messages: [{ role: "user", content: "x" }],
    tools: [WEATHER],
    response_format: { type: "json_schema", json_schema: { schema: { type: "object" } } },
  });
  assert.equal(buildArgs(v, null).includes("--json-schema"), false);
});

/* ------------------------------ legacy route ----------------------------- */

test("parseLegacyCompletion turns a prompt into a single user turn", () => {
  const { v } = parseLegacyCompletion({ model: "sonnet", prompt: "Once upon a time", max_tokens: 20 });
  assert.deepEqual(v.messages, [{ role: "user", content: [{ type: "text", text: "Once upon a time" }] }]);
  assert.equal(v.maxTokens, 20);
  assert.throws(() => parseLegacyCompletion({ model: "sonnet" }), /`prompt` is required/);
});

test("toLegacyCompletion uses the text_completion shape", () => {
  const out = toLegacyCompletion([message("hello")], "chatcmpl-abc") as any;
  assert.equal(out.object, "text_completion");
  assert.equal(out.id, "cmpl-abc");
  assert.equal(out.choices[0].text, "hello");
  assert.equal(out.choices[0].finish_reason, "stop");
});
