# Kiro Gateway — TODO

Features we don't fully support because the Kiro upstream has no matching capability. Each section names the shim that's in place, why it's a shim, and the exact code to change when a real backend is wired up.

---

## `web_search` built-in tool (stub)

**What codex sends.** Codex advertises the OpenAI Responses built-in tool:
```json
{"type": "web_search"}
```
— an opaque tool that the server (normally OpenAI) is expected to execute and return a `WebSearchCall` output item with live search results (see `codex-rs/protocol/src/models.rs:303-` and the parser in `codex-rs/codex-api/src/sse/responses.rs`).

**What the gateway does today.** We translate the built-in into a function-tool stub (`kiro/converters_responses.py:58-71`, entry `_BUILTIN_TOOL_STUBS["web_search"]`). The model can *call* `web_search({query})` the same way it calls any custom function, and the client (codex) handles execution — typically by refusing or by using its own shell-based retrieval. No actual web results come from the gateway.

**Why it's a stub.** Kiro / Amazon Q Developer does not expose a web search API we can proxy. Without a search backend we cannot hydrate a real `WebSearchCall` output.

**How to replace the stub.** When a search backend (Brave / Tavily / Kagi / SerpAPI / custom) is available:

1. Remove `web_search` from `_BUILTIN_TOOL_STUBS` in `kiro/converters_responses.py`.
2. In `convert_responses_tools_to_unified` (same file), stop converting `web_search` to a function; instead track that it was requested and let Kiro handle the tool as-is OR intercept `web_search` function-calls on the streaming side.
3. In `kiro/streaming_responses.py`, add a branch to `stream_kiro_to_responses` and `stream_kiro_to_responses_ws` that, when the model emits a `tool_use` named `web_search`, calls the backend and emits a `response.output_item.added` with `{"type": "web_search_call", "action": {"type": "search", "query": …}, "status": "completed"}` followed by `response.output_item.done` carrying the results. Reference shape: `codex-rs/protocol/src/models.rs:295-` and the fixture at `codex-rs/exec/tests/fixtures/cli_responses_fixture.sse`.
4. Add tests in `tests/unit/test_streaming_responses.py` that assert the search backend is called and that `WebSearchCall` items pass `TestCodexSseContract` wire-shape.
5. Make the backend pluggable via env var (`WEB_SEARCH_PROVIDER=tavily|brave|…` + API key), default off, so users without a backend keep the current stub behavior.

## `file_search` built-in tool (stub)

Same story as `web_search` (`kiro/converters_responses.py:73-85`). Kiro has no file-search API. To replace: mirror the steps above, emit `response.output_item` with `{"type": "file_search_call", …}`, and wire a concrete backend (e.g., local RAG over a user directory). Until then the stub lets the model *pretend* to search and lets codex render the function-call.

## `code_interpreter` built-in tool (stub)

Same pattern (`kiro/converters_responses.py:87-101`). Kiro has no sandboxed-exec API. Codex typically runs code in its own shell anyway, so this stub is the right shape: the model invokes `code_interpreter(code)` as a function, and codex's own `local_shell` / apply_patch path handles execution. Do not attempt to execute arbitrary code server-side from this gateway.

---

## Realtime / voice APIs (not implemented)

Codex can open a WebSocket to `/v1/realtime` or POST to `/v1/realtime/calls` for voice sessions. Kiro has no audio/realtime API. We currently 404 these paths. If Kiro ever exposes a realtime endpoint, add a route under `kiro/routes_realtime.py` that proxies the WebSocket and translates the codex realtime v1/v2 protocol (see `codex-rs/codex-api/src/endpoint/realtime_websocket/protocol_v2.rs`).

## `/v1/memories/trace_summarize` (not implemented)

Codex's long-term-memory summarization endpoint (`codex-rs/codex-api/src/endpoint/memories.rs`). Kiro has no memory store. If users want this, it can be implemented exactly like `/v1/responses/compact`: synthesize a summarization prompt from the incoming `MemorySummarizeInput`, call `generateAssistantResponse`, reshape into `MemorySummarizeOutput`. Low effort; skipped because no user has asked yet.

## `ResponseItem::CustomToolCall` / `CustomToolCallOutput` (not implemented)

Codex's freeform text-in/text-out custom tools (`codex-rs/protocol/src/models.rs:264-287`). The gateway currently only emits `function_call` items. To support custom tools, extend `convert_responses_tools_to_unified` to pass through `type: "custom"` tool definitions and extend the streaming emitter to produce `custom_tool_call` output items when the model's tool schema indicates a freeform tool. Needed only if a user wires codex custom tools against Kiro.

## `ResponseItem::LocalShellCall` (not implemented)

The sandboxed-shell tool (`codex-rs/protocol/src/models.rs:217-226`). Kiro doesn't provide a shell-exec API, and codex runs shell commands in its own sandbox anyway, so exposing this from the gateway would be redundant. Left unimplemented on purpose.
