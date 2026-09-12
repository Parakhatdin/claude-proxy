# claude-proxy

An HTTP service that speaks both the **Anthropic Messages API** and the **OpenAI Chat
Completions API**, and answers either by driving your local `claude -p` CLI. Point any
Anthropic or OpenAI client at `http://localhost:8787` and it works — no provider API key,
because the CLI uses the credentials you are already signed in with.

```bash
npm install && npm run build && node dist/index.js
```

```js
import Anthropic from "@anthropic-ai/sdk";
const client = new Anthropic({ baseURL: "http://localhost:8787", apiKey: "unused" });

const msg = await client.messages.create({
  model: "sonnet",
  max_tokens: 256,
  messages: [{ role: "user", content: "Capital of Japan?" }],
});
```

## Endpoints

| Endpoint | Dialect | Notes |
| --- | --- | --- |
| `POST /v1/messages` | Anthropic | Streaming and non-streaming, with tool use. |
| `POST /v1/chat/completions` | OpenAI | Streaming and non-streaming, terminated with `[DONE]`. |
| `POST /v1/completions` | OpenAI | Legacy text completion, for older clients. |
| `GET /v1/models` | both | One payload shaped to satisfy either client. |
| `POST /v1/messages/count_tokens` | Anthropic | **Estimate only** — see [Fidelity](#fidelity). |
| `POST /v1/embeddings` | OpenAI | Always 400: the CLI has no embedding model. |
| `GET /health` | — | Liveness, current mode, live session count. No auth required. |

The `/v1` prefix is optional on every route, since clients disagree about whether the base
URL already includes it — `/models` and `/v1/models` both work. Errors come back in the
dialect of the route you called, and CORS plus `OPTIONS` preflight are handled so
browser- and Electron-based clients work.

Streaming returns a spec-correct SSE stream (`message_start`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, plus `ping`
keepalives), so the SDK's `client.messages.stream()` helper and `finalMessage()` work
unmodified.

## Connecting an OpenAI-format client

Anything that talks to Ollama's `/v1` or to OpenAI works unchanged — set the base URL to
`http://127.0.0.1:8787/v1` and any non-empty API key:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"Capital of France?"}]}'
```

### Supported parameters

| Parameter | Support |
| --- | --- |
| `tools`, `tool_choice`, `functions`, `function_call` | Yes — see [Tool calling](#tool-calling). |
| `response_format: json_object` | Yes, by instruction. |
| `response_format: json_schema` | Yes, enforced by the CLI's own schema validator. |
| `stream`, `stream_options.include_usage` | Yes. |
| `stop`, `max_tokens`, `max_completion_tokens` | Yes. |
| `n` | Yes, up to `CLAUDE_PROXY_MAX_CHOICES` (4). One CLI run per choice; not with `stream`. |
| `messages` with `image_url` (incl. `data:` URLs) | Yes, converted to Anthropic image blocks. |
| `system` / `developer` roles | Hoisted into the system prompt. |
| `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed`, `logit_bias` | Accepted and **ignored** — the CLI exposes no sampling controls. |
| `logprobs` | Always `null`. |

## Tool calling

Client-defined tools work in both dialects, but by emulation rather than native tool use.
`claude -p` never pauses to hand a `tool_use` back to its caller — it runs its own tool loop
internally — so the proxy describes your tools in the system prompt, has the model emit a
`<tool_call>` block, and parses that back into real `tool_calls` (OpenAI) or `tool_use`
blocks (Anthropic) with `finish_reason: "tool_calls"`. Results you send back on the next
request are replayed into the transcript. A normal agent loop works unchanged:

```js
const first = await client.chat.completions.create({ model: "sonnet", tools, messages });
const call = first.choices[0].message.tool_calls[0];          // get_weather({"city":"Paris"})
messages.push(first.choices[0].message);
messages.push({ role: "tool", tool_call_id: call.id, content: result });
const second = await client.chat.completions.create({ model: "sonnet", tools, messages });
```

`tool_choice` is honoured in all four forms (`auto`, `none`, `required`, a named function),
and several calls in one reply become parallel `tool_calls`.

What to know about the emulation:

- **Replies are buffered when tools are declared.** A tool call is only detectable once the
  whole reply is in hand, so `stream: true` returns a synthetic stream replayed after the
  fact rather than live tokens. The frames are well-formed either way.
- **It depends on instruction-following, not a trained tool-use channel.** It is reliable in
  practice but not identical to the provider's native tool use. A malformed `<tool_call>`
  block is passed through as visible text rather than silently dropped, so you can see what
  the model actually produced.
- **Tool schemas are sent on every request**, since the proxy is stateless.
- **Session reuse is skipped for `n > 1`**, where each choice is its own run.

**Model names are remapped.** A request for a non-Claude model (`gpt-4o`, or Hermes's
`gpt-5.4` placeholder) would make `claude --model` fail, so the proxy substitutes
`CLAUDE_PROXY_DEFAULT_MODEL`, logs a warning, and reports the model actually used in the
response. Real Claude ids and aliases pass through untouched.

## Two modes

**`clean` (default)** — the endpoint behaves like a plain model. The CLI runs with
`--safe-mode --strict-mcp-config --disable-slash-commands --tools ""` and your `system`
prompt replaces Claude Code's own, so no `CLAUDE.md`, skills, plugins, hooks or MCP servers
leak in. This cuts the prompt preamble from ~24k tokens to ~150.

**`agent`** (`CLAUDE_PROXY_MODE=agent`) — the endpoint is the full Claude Code agent. It can
read files and run tools in `CLAUDE_PROXY_CWD`; your `system` prompt is *appended* to
Claude Code's. The internal tool loop is collapsed into one Anthropic message, so a caller
still sees a single ordinary reply. Tool calls need a permission mode that does not prompt
(`CLAUDE_PROXY_PERMISSION_MODE=bypassPermissions`) since nothing is interactive.

> Agent mode gives any caller the tool access you grant it, inside `CLAUDE_PROXY_CWD`.
> Treat the port as trusted and set `CLAUDE_PROXY_API_KEY`.

## Multi-turn and session reuse

The Messages API is stateless — clients resend the whole transcript every turn — while the
CLI is stateful. The proxy bridges this by fingerprinting the conversation prefix each CLI
session has already seen:

- **Follow-up turn** — if the request continues a conversation this proxy already ran, it
  `--resume`s that session and sends only the newest user turn. The CLI keeps its prompt
  cache, so follow-ups are cheaper and faster.
- **Cold start** — otherwise the earlier turns are replayed as a `<transcript>` block inside
  one user message. Feeding them as separate turns would make the model answer each one
  again, so this is deliberate.

A session is handed to one request at a time; a concurrent turn on the same conversation
starts its own rather than corrupting the first. Cache entries expire after
`CLAUDE_PROXY_SESSION_TTL_MS` (30 min). Set `CLAUDE_PROXY_SESSION_REUSE=0` to always cold
start.

Reuse is keyed on the exact reply you echo back, so a client that rewrites assistant content
simply falls back to a cold start — correct either way, just slower.

## Authenticating the CLI

By default the spawned CLI uses whatever login lives in your keychain. That is fine when you
start the proxy from a terminal, and unreliable when launchd or systemd starts it — those
services often cannot reach the keychain, and every request comes back `401`.

For an unattended proxy, mint a long-lived token instead. It needs a Claude subscription and
is valid for a year:

```bash
claude setup-token          # prints an sk-ant-oat01-… token
```

Hand it to the proxy one of two ways:

```bash
# Straight from the environment (or from .env, which scripts/run.sh sources).
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... node dist/index.js

# Or keep it out of .env entirely and point at a secrets file holding only the token.
printf %s "$TOKEN" > ~/.config/claude-proxy/token && chmod 600 ~/.config/claude-proxy/token
CLAUDE_PROXY_OAUTH_TOKEN_FILE=~/.config/claude-proxy/token node dist/index.js
```

Set exactly one of the two — both together is a startup error. The file is read once at
startup, so rotating the token means restarting the proxy. On boot the log names the
credential in use (never the token itself):

```
INFO  CLI credential: CLAUDE_PROXY_OAUTH_TOKEN_FILE (/Users/me/.config/claude-proxy/token)
```

When the proxy holds a token, `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` are stripped
from the CLI's environment so there is exactly one credential in play. Note that this is the
proxy's credential for talking to Claude — it is unrelated to `CLAUDE_PROXY_API_KEY`, which is
the secret *your* callers must present.

## Configuration

Every option is an environment variable; see [`.env.example`](.env.example). The ones that
matter most:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLAUDE_PROXY_HOST` / `_PORT` | `127.0.0.1` / `8787` | Bind address. |
| `CLAUDE_PROXY_API_KEY` | *(unset)* | Require `x-api-key` / `Bearer`. |
| `CLAUDE_CODE_OAUTH_TOKEN` | *(unset)* | Long-lived CLI credential from `claude setup-token`. |
| `CLAUDE_PROXY_OAUTH_TOKEN_FILE` | *(unset)* | Same, read from a secrets file instead. |
| `CLAUDE_PROXY_MODE` | `clean` | `clean` or `agent`. |
| `CLAUDE_PROXY_DEFAULT_MODEL` | `sonnet` | Used when a request omits `model`. |
| `CLAUDE_PROXY_MAX_CONCURRENCY` | `4` | Concurrent CLI processes; excess queues, then 529. |
| `CLAUDE_PROXY_TIMEOUT_MS` | `600000` | Per-request ceiling before the CLI is killed. |
| `CLAUDE_PROXY_INCLUDE_THINKING` | `0` | Force thinking blocks on even when unrequested. |
| `CLAUDE_PROXY_EXTRA_ARGS` | *(none)* | Raw flags appended to every CLI invocation. |

## Fidelity

What the proxy reproduces faithfully, and where it can't:

- **`max_tokens` is enforced by the proxy, approximately.** The CLI has no such flag, so
  output is cut at `max_tokens × 3.8` characters and reported as
  `stop_reason: "max_tokens"`. The estimate comes from `src/tokens.ts`, not a real
  tokenizer. `usage.output_tokens` reports what the model actually generated, which can
  exceed the truncated text. Disable with `CLAUDE_PROXY_ENFORCE_MAX_TOKENS=0`.
- **`stop_sequences` are enforced exactly.** Text that could still become a stop sequence is
  withheld until the next delta resolves it, so a partial match never reaches the client.
- **`count_tokens` is an estimate** from the same heuristic — no local tokenizer exists.
- **Client-defined tools are emulated, not native** — see [Tool calling](#tool-calling).
- **`temperature`, `top_p`, `top_k` are accepted and ignored** — the CLI exposes no sampling
  controls. They are not rejected, because most clients always send them.
- **Thinking blocks are hidden unless requested** via `thinking: { type: "enabled" }`,
  matching the real API. The CLI cannot stop the model thinking, only stop reporting it.
- **An assistant prefill** (a trailing `assistant` message) is honoured as a best-effort
  instruction to continue, not as a true prefill.
- **Images** pass through on the newest turn; images in replayed history become `[image]`.
- **CLI failures become HTTP errors.** The CLI reports problems as a successful-looking
  result with `is_error: true`, so a logged-out CLI would otherwise look like a reply whose
  text is "Not logged in". These are mapped to `401`, `429` or `500` with the CLI's own
  message. A 401 means the CLI itself is not authenticated — see
  [Authenticating the CLI](#authenticating-the-cli).

## Development

```bash
npm run typecheck
npm test          # 76 unit tests, no CLI or network needed
npm run dev       # watch mode, runs the TypeScript directly
```

`src/` layout: `server.ts` (HTTP, auth, concurrency, routing) → `translate.ts` (request →
CLI flags and stdin) → `runner.ts` (spawn, NDJSON decode) → `assemble.ts` (CLI events → one
Anthropic message), with `sessions.ts` for resume bookkeeping, `openai.ts` translating the
OpenAI dialect on and off that same engine, and `tools.ts` handling tool-call emulation.

## Requirements

Node ≥ 20.10 and an authenticated Claude Code CLI (`claude --version`); see
[Authenticating the CLI](#authenticating-the-cli). Everything else is
stdlib; the only devDependencies are TypeScript and `@types/node`.
