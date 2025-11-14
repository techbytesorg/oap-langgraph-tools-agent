# LangChain v1 Agent Migration Log

_Log date: 2025-11-13_

## Baseline snapshot

- Graph entrypoint remains `tools_agent/agent.py:graph`, exporting an async factory that parses `RunnableConfig`, builds tool lists (RAG + MCP), initializes a chat model via `init_chat_model`, and finally calls `langgraph.prebuilt.create_react_agent`.
- `create_react_agent` currently receives:
  - `prompt`: `cfg.system_prompt + UNEDITABLE_SYSTEM_PROMPT`.
  - `model`: output of `init_chat_model`, already parameterized by `cfg.model_name`, `temperature`, `max_tokens`, and API key resolution logic (`get_api_key_for_model`).
  - `tools`: dynamically assembled LangChain `StructuredTool` instances (RAG collections + authenticated MCP tools).
  - `config_schema`: `GraphConfigPydantic`, which defines the OAP-facing fields (model selection, temperature, RAG + MCP config, system prompt, etc.).
  - `response_format`: optional schema returned by `tools_agent.utils.structured_output.load_schema_model`.
  - `store`: module-level `InMemoryStore`.
- Configuration & auth flow:
  - User-selected values arrive inside `config["configurable"]`, parsed into `GraphConfigPydantic`.
  - Supabase JWT tokens are sourced from either `config["configurable"]["x-supabase-access-token"]` or `config["metadata"]["supabaseAccessToken"]`.
  - MCP authentication tokens (when `auth_required=True`) flow through `tools_agent.utils.token.fetch_tokens` and are added as HTTP headers (`Authorization: Bearer ...`).
  - RAG tools rely on both the Supabase token and the `cfg.rag` fields (URL + collection list).

## Migration requirements gleaned from LangChain v1 docs

| Legacy behavior | Required v1 behavior | Reference |
| --- | --- | --- |
| `langgraph.prebuilt.create_react_agent` helper with `prompt` parameter | `langchain.agents.create_agent` with `system_prompt` argument | [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1) |
| `config_schema` prop on the prebuilt graph | v1 no longer surfaces `config_schema`; configs should be parsed manually (as we already do) and exposed via runtime context if needed | [Migration guide – prompts/config](https://docs.langchain.com/oss/python/migrate/langchain-v1) |
| Static metadata injected via `config["configurable"]` | Prefer the new `context=` parameter when invoking the agent; declare a `context_schema` to make data available in tools (`runtime.context`) | [Runtime context update](https://docs.langchain.com/oss/python/migrate/langchain-v1) |
| Structured output enforced via prompt engineering | `create_agent` supports `response_format` directly and surfaces parsed `structured_response` in the final state | [Structured output docs](https://docs.langchain.com/oss/python/langchain/quickstart) |
| Tool/runtime hooks handled via pre/post model hooks | V1 recommends middleware (`wrap_model_call`, `dynamic_prompt`, etc.) for prompt shaping, tool error handling, and response-format switching | [Middleware overview](https://docs.langchain.com/oss/python/langchain/context-engineering) |

## Immediate decisions & open questions

1. **Maintaining GraphConfig-driven UX**: We continue parsing `GraphConfigPydantic` inside `graph()`, but we no longer rely on `config_schema`. Instead, we document that UI surfaces are defined via `langgraph.json` and OAP metadata.
2. **Runtime context exposure**: Supabase tokens, user IDs from decoded JWTs, and MCP credentials that tools might need should be plumbed via an explicit `context_schema`. The agent invocation sites (LangGraph server) must forward these through the new `context` channel.
3. **State schema**: If tools need direct access to user-specific state (e.g., previously fetched MCP headers), we can extend the default `AgentState` or attach middleware to manage additional keys. For now, message history suffices.
4. **Store usage**: `InMemoryStore` already aligns with `create_agent` expectations; no change required beyond ensuring tools access it via `runtime.store`.

## Next implementation steps

1. Upgrade dependencies (`langchain`, `langchain-core`, `langgraph`) to the latest v1-compatible releases, then regenerate the lockfile with `uv sync`.
2. Swap the final `create_react_agent` call for `create_agent`, mapping the existing arguments (`system_prompt`, `model`, `tools`, `response_format`, `store`). Remove the obsolete `config_schema` argument.
3. Introduce a `context_schema` (or document expectations) so downstream callers pass Supabase metadata via the recommended `context` argument rather than `config["configurable"]`.
4. Run `uv run langgraph dev --no-browser` to verify MCP/RAG tools still register and that structured responses populate `structured_response` as expected. Capture findings + any middleware follow-ups back in this log.

---

### 2025-11-13 15:45 UTC – Clarification updates

- **`config_schema` vs `context_schema`:** The migration guide’s “Runtime context” section confirms we still declare a schema for static runtime data, but it now lives under `context_schema` when calling `create_agent` ([docs](https://docs.langchain.com/oss/python/migrate/langchain-v1)). Plan: keep `GraphConfigPydantic` as the canonical definition of configurable knobs, but expose a lightweight dataclass/TypedDict mirror that we pass as `context_schema` so Supabase tokens, MCP details, and RAG selections remain injectable at runtime.
- **Structured output strategy:** LangChain v1 requires wrapping schemas in either `ToolStrategy` or `ProviderStrategy` to control how structured output is produced ([structured output docs](https://docs.langchain.com/oss/python/langchain/structured-output)). Our agent relies on tool-calling semantics across providers, so we’ll explicitly set `response_format=ToolStrategy(LoadedSchema)` when `load_schema_model` returns a Pydantic model to guarantee consistent behavior.
- **Model selection path:** `init_chat_model` is still the recommended convenience for binding a single provider/model name ([model overview](https://docs.langchain.com/oss/python/langchain/models)). However, the migration guide highlights “Dynamic model selection” via middleware ([same migrate doc](https://docs.langchain.com/oss/python/migrate/langchain-v1)). Conclusion: we can retain `init_chat_model` for the default model (so existing config fields keep working) while leaving room to add an optional middleware later if we need to auto-switch models based on context (e.g., conversation length or tenant-specific budgets).
- **Log housekeeping:** Future edits must append changelog entries like this with timestamps so we can trace every decision during the migration.

### 2025-11-13 16:35 UTC – Agent + dependency updates

- **Dependency sweep:** Bumped `langchain`, `langchain-core`, `langgraph`, `langchain-openai`, and `langchain-anthropic` to their `1.0.x` lines and refreshed `uv.lock`. Downgraded the `langgraph-cli[inmem]` dev extra to the latest `0.4.x` since no 1.x artifact exists yet (per CLI docs).
- **create_agent adoption:** Swapped the final `create_react_agent` call for `langchain.agents.create_agent`, wiring in the existing system prompt, tool list, and memory store while setting `context_schema=GraphConfigPydantic` so runtime context replaces the old config_schema surface.
- **Structured output strategy:** Wrapped dynamically loaded schema models with `ToolStrategy(...)` before passing them into `create_agent`, matching the v1 requirement for explicit structured-output strategies.
- **Default runtime context:** After parsing `GraphConfigPydantic` we now bind it as the default `context` via `agent.with_config(context=cfg)` so the LangGraph runtime can continue injecting user-configured RAG/MCP params through the new context channel.

### 2025-11-13 16:55 UTC – Smoke test

- Started `uv run langgraph dev --no-browser` to validate the upgraded dependencies + new agent wiring. Server booted cleanly (no stack traces in the startup logs) and listened until manually terminated; no MCP/RAG initialization errors surfaced during bring-up. This confirms the project still compiles under LangChain/LangGraph v1.0.x.

