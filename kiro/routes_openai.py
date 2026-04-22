# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
FastAPI routes for Kiro Gateway.

Contains all API endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions
- /v1/responses: Responses API (OpenAI Agents SDK compatible)
"""

import json
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import (
    PROXY_API_KEY,
    APP_VERSION,
)
from kiro.models_openai import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
)
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver
from kiro.converters_openai import build_kiro_payload
from kiro.converters_responses import build_kiro_payload as build_kiro_payload_responses
from kiro.response_store import (
    ResponseStore,
    StoredTurn,
    _sanitize_stored_input_for_replay,
)
from kiro.streaming_openai import (
    stream_kiro_to_openai,
    collect_stream_response,
    stream_with_first_token_retry,
)
from kiro.streaming_responses import (
    stream_kiro_to_responses,
    stream_kiro_to_responses_ws,
    collect_responses_response,
    _sse_event as _responses_sse_event,
    _ws_event as _responses_ws_event,
    _generate_response_id,
)
from kiro.responses_errors import (
    build_response_failed_event,
    classify_upstream_error,
)
from kiro.http_client import KiroHttpClient
from kiro.utils import generate_conversation_id
from kiro.config import WEB_SEARCH_ENABLED
from kiro.mcp_tools import handle_native_web_search

# Import debug_logger
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# --- Security scheme ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    """
    Verify API key in Authorization header.

    Expects format: "Bearer {PROXY_API_KEY}"

    Args:
        auth_header: Authorization header value

    Returns:
        True if key is valid

    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# --- Router ---
router = APIRouter()


async def _resume_from_prev_response(
    request_data,
    response_store: Optional[ResponseStore],
) -> Tuple[Optional[list], Optional[str]]:
    """
    Look up a prior turn by ``previous_response_id`` and merge it in.

    Codex's delta protocol sends only new input items on turn N>=2
    (typically ``function_call_output`` items) plus the prior turn's
    response id. This helper fetches the stored canonical conversation
    and returns a merged ``input`` list ready to replace
    ``request_data.input``.

    The stored ``input_items`` are sanitized to drop reasoning-type
    items before replay — reasoning is a server-side artifact and
    replaying it confuses the model.

    Args:
        request_data: Parsed ``ResponsesRequest``.
        response_store: App-wide store (may be ``None`` in tests).

    Returns:
        Tuple of ``(merged_input, merged_instructions)``. Either may be
        ``None`` when no merge is needed (no prev id, cache miss, or
        store disabled). The caller patches ``request_data`` with the
        non-None values.
    """
    prev_id = getattr(request_data, "previous_response_id", None)
    if not prev_id or response_store is None:
        return None, None

    stored = await response_store.get(prev_id)
    if stored is None:
        # Cache miss is non-fatal: codex may have restarted or this id
        # belongs to a different session. Log and let the request
        # proceed as-is — the model will see only the deltas, which is
        # the current (broken) behaviour without the store.
        logger.warning(
            f"/v1/responses: previous_response_id={prev_id!r} not in store "
            "(cache miss) — processing request as-is"
        )
        return None, None

    # Build the replayable prior-turn items. Reasoning is dropped.
    replay_items = _sanitize_stored_input_for_replay(stored.input_items)

    # Current turn input may be a raw string (non-codex client) — in
    # that case, wrap it as a user message so we can concatenate lists.
    current_input = request_data.input
    if isinstance(current_input, str):
        new_items: List[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": current_input,
            }
        ]
    elif current_input is None:
        new_items = []
    else:
        # Items may be Pydantic models — coerce to dicts for persistence.
        new_items = [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in current_input
        ]

    merged_input = replay_items + new_items
    merged_instructions = stored.instructions if not request_data.instructions else None

    logger.info(
        f"/v1/responses: Resume from previous_response_id={prev_id!r} "
        f"(+{len(new_items)} new items, merged total={len(merged_input)})"
    )
    return merged_input, merged_instructions


async def _persist_turn(
    response_store: Optional[ResponseStore],
    response_id: str,
    request_data,
    canonical_input: list,
    output_items: List[dict],
) -> None:
    """
    Store a completed turn for later ``previous_response_id`` resume.

    Called from both the streaming on_complete callback and the
    non-streaming JSON path, so the same canonical snapshot lands in
    the store regardless of mode.

    Args:
        response_store: App-wide store (may be ``None`` in tests).
        response_id: Id of the response this turn produced.
        request_data: The already-merged ``ResponsesRequest`` (so
            ``input`` is the full reconstructed conversation).
        canonical_input: Input items used for this turn (what codex
            would have sent non-delta).
        output_items: Output items this response produced.
    """
    if response_store is None:
        return
    try:
        turn = StoredTurn(
            input_items=list(canonical_input),
            output_items=list(output_items),
            model=request_data.model,
            instructions=request_data.instructions or None,
        )
        await response_store.put(response_id, turn)
        logger.debug(
            f"/v1/responses: stored turn {response_id} "
            f"(input_items={len(canonical_input)}, output_items={len(output_items)})"
        )
    except Exception as e:
        logger.warning(f"/v1/responses: failed to persist turn {response_id}: {e}")


def _coerce_input_to_list(request_input) -> List[dict]:
    """
    Normalize ``request_data.input`` to a list of dicts for storage.

    Accepts a string (wraps as a user message), a list of dicts or
    Pydantic models (model_dumps each), or None (returns empty list).
    """
    if request_input is None:
        return []
    if isinstance(request_input, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": request_input,
            }
        ]
    out: List[dict] = []
    for item in request_input:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
        else:
            out.append(item)
    return out


@router.get("/")
async def root():
    """
    Health check endpoint.

    Returns:
        Status and application version
    """
    return {
        "status": "ok",
        "message": "Kiro Gateway is running",
        "version": APP_VERSION,
    }


@router.get("/health")
async def health():
    """
    Detailed health check.

    Returns:
        Status, timestamp and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
    }


@router.get(
    "/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)]
)
async def get_models(request: Request):
    """
    Return list of available models.

    Models are loaded at startup (blocking) and cached.
    This endpoint returns the cached list.

    Args:
        request: FastAPI Request for accessing app.state

    Returns:
        ModelList with available models in consistent format (with dots)
    """
    logger.info("Request to /v1/models")

    model_resolver: ModelResolver = request.app.state.model_resolver

    # Get all available models from resolver (cache + hidden models)
    available_model_ids = model_resolver.get_available_models()

    # Build OpenAI-compatible model list
    openai_models = [
        OpenAIModel(
            id=model_id, owned_by="anthropic", description="Claude model via Kiro API"
        )
        for model_id in available_model_ids
    ]

    return ModelList(data=openai_models)


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    """
    Chat completions endpoint - compatible with OpenAI API.

    Accepts requests in OpenAI format and translates them to Kiro API.
    Supports streaming and non-streaming modes.

    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in OpenAI ChatCompletionRequest format

    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode

    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(
        f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})"
    )

    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache

    # Note: prepare_new_request() and log_request_body() are now called by DebugLoggerMiddleware
    # This ensures debug logging works even for requests that fail Pydantic validation (422 errors)

    # Check for truncation recovery opportunities
    from kiro.truncation_state import get_tool_truncation, get_content_truncation
    from kiro.truncation_recovery import (
        generate_truncation_tool_result,
        generate_truncation_user_message,
    )
    from kiro.models_openai import ChatMessage

    modified_messages = []
    tool_results_modified = 0
    content_notices_added = 0

    for msg in request_data.messages:
        # Check if this is a tool_result for a truncated tool call
        if msg.role == "tool" and msg.tool_call_id:
            truncation_info = get_tool_truncation(msg.tool_call_id)
            if truncation_info:
                # Modify tool_result content to include truncation notice
                synthetic = generate_truncation_tool_result(
                    tool_name=truncation_info.tool_name,
                    tool_use_id=msg.tool_call_id,
                    truncation_info=truncation_info.truncation_info,
                )
                # Prepend truncation notice to original content
                modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"

                # Create NEW ChatMessage object (Pydantic immutability)
                modified_msg = msg.model_copy(update={"content": modified_content})
                modified_messages.append(modified_msg)
                tool_results_modified += 1
                logger.debug(
                    f"Modified tool_result for {msg.tool_call_id} to include truncation notice"
                )
                continue  # Skip normal append since we already added modified version

        # Check if this is an assistant message with truncated content
        if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
            truncation_info = get_content_truncation(msg.content)
            if truncation_info:
                # Add this message first
                modified_messages.append(msg)
                # Then add synthetic user message about truncation
                synthetic_user_msg = ChatMessage(
                    role="user", content=generate_truncation_user_message()
                )
                modified_messages.append(synthetic_user_msg)
                content_notices_added += 1
                logger.debug(
                    f"Added truncation notice after assistant message (hash: {truncation_info.message_hash})"
                )
                continue  # Skip normal append since we already added it

        modified_messages.append(msg)

    if tool_results_modified > 0 or content_notices_added > 0:
        request_data.messages = modified_messages
        logger.info(
            f"Truncation recovery: modified {tool_results_modified} tool_result(s), "
            f"added {content_notices_added} content notice(s)"
        )

    # ==============================================================================
    # WebSearch Support - Path B: Auto-Injection (MCP Tool Emulation)
    # ==============================================================================

    # Auto-inject web_search tool if enabled (Path B - MCP emulation)
    if WEB_SEARCH_ENABLED:
        if request_data.tools is None:
            request_data.tools = []

        # Check if web_search already exists
        has_ws = any(
            getattr(tool, "type", None) == "function" and
            getattr(getattr(tool, "function", None), "name", None) == "web_search"
            for tool in request_data.tools
        )

        if not has_ws:
            from kiro.models_openai import Tool, ToolFunction
            web_search_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Search the web for current information. Use when you need up-to-date data from the internet.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                )
            )
            request_data.tools.append(web_search_tool)
            logger.debug("Auto-injected web_search tool for MCP emulation (Path B)")

    # ==============================================================================
    # WebSearch Support - Path A: Native Format Check (OpenAI doesn't have native server-side tools)
    # ==============================================================================

    # OpenAI API doesn't have native server-side tools like Anthropic
    # But we check for consistency - if someone sends web_search function, handle it
    # This is actually Path B for OpenAI (all web_search goes through MCP emulation)

    # Generate conversation ID for Kiro API (random UUID, not used for tracking)
    conversation_id = generate_conversation_id()

    # Build payload for Kiro
    # profileArn is only needed for Kiro Desktop auth
    # AWS SSO OIDC (Builder ID) users don't need profileArn and it causes 403 if sent
    profile_arn_for_payload = ""
    if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
        profile_arn_for_payload = auth_manager.profile_arn

    try:
        kiro_payload = build_kiro_payload(
            request_data, conversation_id, profile_arn_for_payload
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log Kiro payload
    try:
        kiro_request_body = json.dumps(
            kiro_payload, ensure_ascii=False, indent=2
        ).encode("utf-8")
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")

    # Create HTTP client with retry logic
    # For streaming: use per-request client to avoid CLOSE_WAIT leak on VPN disconnect (issue #54)
    # For non-streaming: use shared client for connection pooling
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    logger.debug(f"Kiro API URL: {url}")

    if request_data.stream:
        # Streaming mode: per-request client prevents orphaned connections
        # when network interface changes (VPN disconnect/reconnect)
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        # Non-streaming mode: shared client for efficient connection reuse
        shared_client = request.app.state.http_client
        http_client = KiroHttpClient(auth_manager, shared_client=shared_client)
    try:
        # Make request to Kiro API (for both streaming and non-streaming modes)
        # Important: we wait for Kiro response BEFORE returning StreamingResponse,
        # so that 200 OK means Kiro accepted the request and started responding
        response = await http_client.request_with_retry(
            "POST", url, kiro_payload, stream=True
        )

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"

            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")

            # Try to parse JSON response from Kiro to extract error message
            error_message = error_text
            try:
                error_json = json.loads(error_text)
                # Enhance Kiro API errors with user-friendly messages
                from kiro.kiro_errors import enhance_kiro_error

                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
                # Log original error for debugging
                logger.debug(
                    f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})"
                )
            except (json.JSONDecodeError, KeyError):
                pass

            # Log access log for error (before flush, so it gets into app_logs)
            logger.warning(
                f"HTTP {response.status_code} - POST /v1/chat/completions - {error_message[:100]}"
            )

            # Flush debug logs on error ("errors" mode)
            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)

            # Return error in OpenAI API format
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "error": {
                        "message": error_message,
                        "type": "kiro_api_error",
                        "code": response.status_code,
                    }
                },
            )

        # Prepare data for fallback token counting
        # Convert Pydantic models to dicts for tokenizer
        messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
        tools_for_tokenizer = (
            [tool.model_dump() for tool in request_data.tools]
            if request_data.tools
            else None
        )

        if request_data.stream:
            # Streaming mode with first token retry — collect per-model usage for stats
            from kiro.token_stats import UsageCollector
            collector = UsageCollector()

            async def stream_wrapper():
                streaming_error = None
                client_disconnected = False
                try:
                    # Create retry request function for retries
                    async def make_retry_request():
                        return await http_client.request_with_retry(
                            "POST", url, kiro_payload, stream=True
                        )
                    
                    # Use retry wrapper with initial response
                    async for chunk in stream_with_first_token_retry(
                        make_request=make_retry_request,
                        client=http_client.client,
                        model=request_data.model,
                        model_cache=model_cache,
                        auth_manager=auth_manager,
                        initial_response=response,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer,
                        usage_collector=collector,
                    ):
                        yield chunk
                except GeneratorExit:
                    # Client disconnected - this is normal
                    client_disconnected = True
                    logger.debug(
                        "Client disconnected during streaming (GeneratorExit in routes)"
                    )
                except Exception as e:
                    streaming_error = e
                    # Try to send [DONE] to client before finishing
                    # so client doesn't "hang" waiting for data
                    try:
                        yield "data: [DONE]\n\n"
                    except Exception:
                        pass  # Client already disconnected
                    raise
                finally:
                    await http_client.close()
                    # Log access log for streaming (success or error)
                    if streaming_error:
                        error_type = type(streaming_error).__name__
                        error_msg = (
                            str(streaming_error)
                            if str(streaming_error)
                            else "(empty message)"
                        )
                        logger.error(
                            f"HTTP 500 - POST /v1/chat/completions (streaming) - [{error_type}] {error_msg[:100]}"
                        )
                    elif client_disconnected:
                        logger.info(
                            f"HTTP 200 - POST /v1/chat/completions (streaming) - client disconnected"
                        )
                    else:
                        logger.info(
                            f"HTTP 200 - POST /v1/chat/completions (streaming) - completed"
                        )
                    # Write debug logs AFTER streaming completes
                    if debug_logger:
                        if streaming_error:
                            debug_logger.flush_on_error(500, str(streaming_error))
                        else:
                            debug_logger.discard_buffers()
                    # Record token usage
                    if collector.input_tokens or collector.output_tokens:
                        try:
                            request.app.state.token_stats.record(
                                collector.model, collector.input_tokens, collector.output_tokens
                            )
                        except Exception as stats_err:
                            logger.debug(f"Failed to record token stats: {stats_err}")

            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

        else:
            # Non-streaming mode - collect entire response
            openai_response = await collect_stream_response(
                http_client.client,
                response,
                request_data.model,
                model_cache,
                auth_manager,
                request_messages=messages_for_tokenizer,
                request_tools=tools_for_tokenizer,
            )

            await http_client.close()

            # Log access log for non-streaming success
            logger.info(
                f"HTTP 200 - POST /v1/chat/completions (non-streaming) - completed"
            )

            # Record token usage
            usage = openai_response.get("usage", {})
            if usage:
                try:
                    request.app.state.token_stats.record(
                        request_data.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )
                except Exception as stats_err:
                    logger.debug(f"Failed to record token stats: {stats_err}")

            # Write debug logs after non-streaming request completes
            if debug_logger:
                debug_logger.discard_buffers()

            return JSONResponse(content=openai_response)

    except HTTPException as e:
        await http_client.close()
        # Log access log for HTTP error
        logger.error(f"HTTP {e.status_code} - POST /v1/chat/completions - {e.detail}")
        # Flush debug logs on HTTP error ("errors" mode)
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        # Log access log for internal error
        logger.error(f"HTTP 500 - POST /v1/chat/completions - {str(e)[:100]}")
        # Flush debug logs on internal error ("errors" mode)
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def responses(request: Request):
    """
    Responses API endpoint - compatible with OpenAI Responses API.

    Accepts requests in OpenAI Responses API format and translates them to Kiro API.
    Supports streaming and non-streaming modes.

    This endpoint is used by the OpenAI Agents SDK and other frameworks
    that target the Responses API instead of Chat Completions.

    State for ``previous_response_id``: the gateway maintains an
    in-memory ``ResponseStore`` so codex's delta protocol works. On turn
    N>=2, codex sends only new input items (typically
    ``function_call_output`` items) plus a ``previous_response_id``. The
    handler looks up the stored canonical conversation, merges it with
    the new items, then runs the normal Kiro payload build so the model
    sees the full prior context.

    Args:
        request: FastAPI Request for accessing app.state

    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode

    Raises:
        HTTPException: On validation or API errors
    """
    from kiro.models_responses import ResponsesRequest

    # Parse request body manually to support both Pydantic models and raw dicts
    try:
        body = await request.json()
        request_data = ResponsesRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

    logger.info(
        f"Request to /v1/responses (model={request_data.model}, stream={request_data.stream}, "
        f"input_type={'str' if isinstance(request_data.input, str) else f'list[{len(request_data.input)}]'}, "
        f"tools={len(request_data.tools) if request_data.tools else 0})"
    )

    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache
    response_store: Optional[ResponseStore] = getattr(
        request.app.state, "response_store", None
    )

    # --- Resume from previous_response_id (codex delta protocol) ---
    merged_input_items, merged_instructions = await _resume_from_prev_response(
        request_data, response_store
    )
    # Patch request_data in place so the rest of the handler and the
    # payload builder see the full reconstructed conversation.
    if merged_input_items is not None:
        request_data.input = merged_input_items
    if merged_instructions is not None and not request_data.instructions:
        request_data.instructions = merged_instructions

    # Generate conversation ID for Kiro API
    conversation_id = generate_conversation_id()

    # Build payload for Kiro
    profile_arn_for_payload = ""
    if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
        profile_arn_for_payload = auth_manager.profile_arn

    try:
        kiro_payload = build_kiro_payload_responses(
            request_data,
            conversation_id,
            profile_arn_for_payload,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log Kiro payload
    try:
        kiro_request_body = json.dumps(
            kiro_payload, ensure_ascii=False, indent=2
        ).encode("utf-8")
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")

    # Create HTTP client with retry logic
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    logger.debug(f"Kiro API URL: {url}")

    if request_data.stream:
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        shared_client = request.app.state.http_client
        http_client = KiroHttpClient(auth_manager, shared_client=shared_client)

    try:
        response = await http_client.request_with_retry(
            "POST",
            url,
            kiro_payload,
            stream=True,
        )

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"

            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")

            error_message = error_text
            error_info = None
            try:
                error_json = json.loads(error_text)
                from kiro.kiro_errors import enhance_kiro_error

                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
                logger.debug(
                    f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})"
                )
            except (json.JSONDecodeError, KeyError):
                pass

            logger.warning(
                f"HTTP {response.status_code} - POST /v1/responses - {error_message[:100]}"
            )

            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)

            # Streaming clients (codex) cannot consume JSON HTTP errors — the
            # SSE parser treats a non-200 transport as a generic "stream
            # closed" fault and retry-storms. Emit a terminal response.failed
            # SSE event instead so codex maps the upstream failure onto its
            # ApiError taxonomy (context_length_exceeded → fatal,
            # rate_limit_exceeded → backoff, etc).
            if request_data.stream:
                classified = classify_upstream_error(
                    status_code=response.status_code,
                    error_info=error_info,
                    raw_message=error_text,
                )
                resp_id = _generate_response_id()
                created_at = int(datetime.now(timezone.utc).timestamp())
                failed_payload = build_response_failed_event(
                    response_id=resp_id,
                    created_at=created_at,
                    model=request_data.model,
                    error=classified,
                )

                async def _emit_failed():
                    yield _responses_sse_event("response.failed", failed_payload)

                return StreamingResponse(
                    _emit_failed(), media_type="text/event-stream"
                )

            return JSONResponse(
                status_code=response.status_code,
                content={
                    "error": {
                        "message": error_message,
                        "type": "kiro_api_error",
                        "code": response.status_code,
                    }
                },
            )

        # Prepare data for fallback token counting
        messages_for_tokenizer = None
        tools_for_tokenizer = None
        if request_data.tools:
            tools_for_tokenizer = [
                t.model_dump() if hasattr(t, "model_dump") else t
                for t in request_data.tools
            ]

        if request_data.stream:
            # Streaming mode
            from kiro.token_stats import UsageCollector
            resp_collector = UsageCollector()

            # Canonical input snapshot for the store — freeze it now so
            # even if request_data.input gets further mutated later we
            # persist the exact sequence the model saw on this turn.
            canonical_input = _coerce_input_to_list(request_data.input)

            async def _on_complete(resp_id: str, final_output: list) -> None:
                await _persist_turn(
                    response_store,
                    resp_id,
                    request_data,
                    canonical_input,
                    final_output,
                )

            async def stream_wrapper():
                streaming_error = None
                client_disconnected = False
                try:
                    chunk_count = 0
                    async for chunk in stream_kiro_to_responses(
                        http_client.client,
                        response,
                        request_data.model,
                        model_cache,
                        auth_manager,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer,
                        usage_collector=resp_collector,
                        on_complete=_on_complete,
                    ):
                        chunk_count += 1
                        yield chunk
                    logger.debug(
                        f"[Responses stream_wrapper] Generator exhausted normally, "
                        f"yielded {chunk_count} chunks"
                    )
                except GeneratorExit:
                    client_disconnected = True
                    logger.debug(
                        "Client disconnected during streaming (GeneratorExit in routes)"
                    )
                except Exception as e:
                    streaming_error = e
                    raise
                finally:
                    await http_client.close()
                    if streaming_error:
                        error_type = type(streaming_error).__name__
                        error_msg = (
                            str(streaming_error)
                            if str(streaming_error)
                            else "(empty message)"
                        )
                        logger.error(
                            f"HTTP 500 - POST /v1/responses (streaming) - [{error_type}] {error_msg[:100]}"
                        )
                    elif client_disconnected:
                        logger.info(
                            "HTTP 200 - POST /v1/responses (streaming) - client disconnected"
                        )
                    else:
                        logger.info(
                            "HTTP 200 - POST /v1/responses (streaming) - completed"
                        )
                    if debug_logger:
                        if streaming_error:
                            debug_logger.flush_on_error(500, str(streaming_error))
                        else:
                            debug_logger.discard_buffers()
                    # Record token usage
                    if resp_collector.input_tokens or resp_collector.output_tokens:
                        try:
                            request.app.state.token_stats.record(
                                resp_collector.model, resp_collector.input_tokens, resp_collector.output_tokens
                            )
                        except Exception as stats_err:
                            logger.debug(f"Failed to record token stats: {stats_err}")

            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

        else:
            # Non-streaming mode
            responses_response = await collect_responses_response(
                http_client.client,
                response,
                request_data.model,
                model_cache,
                auth_manager,
                request_messages=messages_for_tokenizer,
                request_tools=tools_for_tokenizer,
            )

            await http_client.close()

            logger.info("HTTP 200 - POST /v1/responses (non-streaming) - completed")

            # Persist into the response store so codex can resume later.
            try:
                await _persist_turn(
                    response_store,
                    responses_response.get("id", ""),
                    request_data,
                    _coerce_input_to_list(request_data.input),
                    responses_response.get("output", []) or [],
                )
            except Exception as store_err:
                logger.warning(f"Failed to store turn: {store_err}")

            # Record token usage
            usage = responses_response.get("usage", {})
            if usage:
                try:
                    request.app.state.token_stats.record(
                        request_data.model,
                        usage.get("input_tokens", 0),
                        usage.get("output_tokens", 0),
                    )
                except Exception as stats_err:
                    logger.debug(f"Failed to record token stats: {stats_err}")

            if debug_logger:
                debug_logger.discard_buffers()

            return JSONResponse(content=responses_response)

    except HTTPException as e:
        await http_client.close()
        logger.error(f"HTTP {e.status_code} - POST /v1/responses - {e.detail}")
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        logger.error(f"HTTP 500 - POST /v1/responses - {str(e)[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/v1/responses/compact", dependencies=[Depends(verify_api_key)])
async def responses_compact(request: Request):
    """
    Responses API compaction endpoint — codex "/compact" target.

    Codex calls this to summarize a long conversation into a handoff
    message it will reinject as history. The wire shape on the request
    side is codex's `CompactionInput`: same fields as a normal Responses
    request (model, input[], instructions, tools, ...), with one extra
    constraint — the response body must be:

        {"output": [ResponseItem, ...]}

    where each ResponseItem has the same structure as a non-streaming
    `/v1/responses` output item. Codex reads `parsed.output` directly, so
    the shape must match exactly.

    Implementation: Kiro has no native "summarize this conversation"
    endpoint. We synthesize a one-shot completion by prepending a
    compaction instruction to the original instructions, force
    stream=False, and reshape the collected text into a single assistant
    message output item.

    Args:
        request: FastAPI Request for accessing app.state

    Returns:
        JSONResponse with {"output": [...]} matching codex CompactHistoryResponse.

    Raises:
        HTTPException: 400 for validation/build errors, upstream status
        codes for Kiro API failures.
    """
    from kiro.models_responses import ResponsesRequest

    # Parse body tolerantly — codex sends CompactionInput which is a
    # superset of our ResponsesRequest (same field names for model/input/
    # instructions/tools; reasoning/text live under extra=allow).
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # Reject obviously empty payloads early so codex sees a clean 400
    # instead of passing an empty prompt to Kiro.
    input_field = body.get("input")
    if input_field is None or (isinstance(input_field, list) and len(input_field) == 0):
        raise HTTPException(
            status_code=400,
            detail="Compaction requires a non-empty 'input' field",
        )

    # Prepend a compaction instruction so the model produces a faithful
    # handoff summary instead of continuing the conversation. Preserve
    # the caller's original instructions (Kiro gets them as additional
    # system context).
    COMPACTION_INSTRUCTIONS = (
        "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a faithful "
        "handoff summary of the conversation so another LLM can resume the "
        "task. Include: current progress and key decisions, important "
        "constraints and user preferences, what remains to be done with clear "
        "next steps, and any critical data/examples/references needed to "
        "continue. Be concise and structured. Output only the summary."
    )
    original_instructions = body.get("instructions") or ""
    if original_instructions:
        merged_instructions = (
            f"{COMPACTION_INSTRUCTIONS}\n\n--- Original instructions ---\n"
            f"{original_instructions}"
        )
    else:
        merged_instructions = COMPACTION_INSTRUCTIONS

    body["instructions"] = merged_instructions
    # Force unary mode: compact is never streamed back to codex.
    body["stream"] = False
    # Strip tools — we're asking for a summary, not tool use.
    body["tools"] = []

    try:
        request_data = ResponsesRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid compact request: {e}")

    logger.info(
        f"Request to /v1/responses/compact (model={request_data.model}, "
        f"input_type={'str' if isinstance(request_data.input, str) else f'list[{len(request_data.input)}]'})"
    )

    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache

    conversation_id = generate_conversation_id()

    profile_arn_for_payload = ""
    if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
        profile_arn_for_payload = auth_manager.profile_arn

    try:
        kiro_payload = build_kiro_payload_responses(
            request_data,
            conversation_id,
            profile_arn_for_payload,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        kiro_request_body = json.dumps(
            kiro_payload, ensure_ascii=False, indent=2
        ).encode("utf-8")
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")

    # Non-streaming path: shared client for connection pooling.
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    shared_client = request.app.state.http_client
    http_client = KiroHttpClient(auth_manager, shared_client=shared_client)

    try:
        response = await http_client.request_with_retry(
            "POST",
            url,
            kiro_payload,
            stream=True,
        )

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"

            await http_client.close()
            error_text = error_content.decode("utf-8", errors="replace")

            error_message = error_text
            try:
                error_json = json.loads(error_text)
                from kiro.kiro_errors import enhance_kiro_error

                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
            except (json.JSONDecodeError, KeyError):
                pass

            logger.warning(
                f"HTTP {response.status_code} - POST /v1/responses/compact - {error_message[:100]}"
            )

            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)

            return JSONResponse(
                status_code=response.status_code,
                content={
                    "error": {
                        "message": error_message,
                        "type": "kiro_api_error",
                        "code": response.status_code,
                    }
                },
            )

        # Collect full response — same path as non-streaming /v1/responses.
        responses_response = await collect_responses_response(
            http_client.client,
            response,
            request_data.model,
            model_cache,
            auth_manager,
        )

        await http_client.close()

        # Extract the assistant summary text and reshape into a single
        # message output item. Codex only reads `output`, so that's what
        # we emit. Drop reasoning/function_call items — they're not
        # useful handoff material.
        summary_text = ""
        for item in responses_response.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        summary_text += block.get("text", "")

        if not summary_text:
            summary_text = "(compaction produced no summary)"

        compact_item = {
            "type": "message",
            "id": _generate_response_id().replace("resp_", "item_"),
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": summary_text,
                    "annotations": [],
                }
            ],
        }

        logger.info(
            f"HTTP 200 - POST /v1/responses/compact - completed "
            f"(summary_chars={len(summary_text)})"
        )

        # Record token usage so the dashboard stays accurate.
        usage = responses_response.get("usage", {}) or {}
        if usage:
            try:
                request.app.state.token_stats.record(
                    request_data.model,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )
            except Exception as stats_err:
                logger.debug(f"Failed to record token stats: {stats_err}")

        if debug_logger:
            debug_logger.discard_buffers()

        return JSONResponse(content={"output": [compact_item]})

    except HTTPException as e:
        await http_client.close()
        logger.error(f"HTTP {e.status_code} - POST /v1/responses/compact - {e.detail}")
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error in /v1/responses/compact: {e}", exc_info=True)
        logger.error(f"HTTP 500 - POST /v1/responses/compact - {str(e)[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.websocket("/v1/responses")
async def responses_ws(websocket: WebSocket):
    """
    WebSocket endpoint for Responses API — used by Codex CLI.

    Codex connects via WebSocket first, falls back to HTTP POST.
    Protocol:
    - Client sends JSON: {"type": "response.create", ...request fields}
    - Server sends JSON events: {"type": "response.created", "response": {...}}, etc.
    - Stream ends with {"type": "response.completed", "response": {...}}
    """
    # Auth: extract from upgrade headers
    auth_header = websocket.headers.get("authorization", "")
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("WebSocket /v1/responses: invalid API key")
        await websocket.close(code=1008, reason="Invalid or missing API Key")
        return

    await websocket.accept()
    logger.info("WebSocket /v1/responses: connection accepted")

    from kiro.models_responses import ResponsesRequest

    auth_manager: KiroAuthManager = websocket.app.state.auth_manager
    model_cache: ModelInfoCache = websocket.app.state.model_cache
    response_store: Optional[ResponseStore] = getattr(
        websocket.app.state, "response_store", None
    )
    http_client = None

    try:
        # Receive requests in a loop (Codex sends multiple turns on one connection)
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.debug("WebSocket /v1/responses: client disconnected")
                break

            try:
                body = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"WebSocket /v1/responses: invalid JSON: {e}")
                await websocket.send_text(json.dumps({
                    "type": "response.failed",
                    "response": {"error": {"message": f"Invalid JSON: {e}", "code": "invalid_json"}},
                }))
                continue

            # Strip the "response.create" wrapper if present
            msg_type = body.pop("type", None)
            if msg_type and msg_type != "response.create":
                logger.debug(f"WebSocket /v1/responses: unknown message type: {msg_type}")
                continue

            try:
                request_data = ResponsesRequest(**body)
                # Force streaming for WebSocket
                request_data.stream = True
            except Exception as e:
                logger.warning(f"WebSocket /v1/responses: invalid request: {e}")
                await websocket.send_text(json.dumps({
                    "type": "response.failed",
                    "response": {"error": {"message": f"Invalid request: {e}", "code": "invalid_request"}},
                }))
                continue

            # Resume from previous_response_id (codex delta protocol)
            merged_input_items, merged_instructions = await _resume_from_prev_response(
                request_data, response_store
            )
            if merged_input_items is not None:
                request_data.input = merged_input_items
            if merged_instructions is not None and not request_data.instructions:
                request_data.instructions = merged_instructions

            logger.info(
                f"WebSocket /v1/responses: request (model={request_data.model}, "
                f"input_type={'str' if isinstance(request_data.input, str) else f'list[{len(request_data.input)}]'}, "
                f"tools={len(request_data.tools) if request_data.tools else 0})"
            )

            conversation_id = generate_conversation_id()

            profile_arn_for_payload = ""
            if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
                profile_arn_for_payload = auth_manager.profile_arn

            try:
                kiro_payload = build_kiro_payload_responses(
                    request_data, conversation_id, profile_arn_for_payload,
                )
            except ValueError as e:
                await websocket.send_text(json.dumps({
                    "type": "response.failed",
                    "response": {"error": {"message": str(e), "code": "invalid_request"}},
                }))
                continue

            url = f"{auth_manager.api_host}/generateAssistantResponse"
            http_client = KiroHttpClient(auth_manager, shared_client=None)

            try:
                kiro_response = await http_client.request_with_retry(
                    "POST", url, kiro_payload, stream=True,
                )

                if kiro_response.status_code != 200:
                    try:
                        error_content = await kiro_response.aread()
                    except Exception:
                        error_content = b"Unknown error"

                    error_text = error_content.decode("utf-8", errors="replace")
                    error_message = error_text
                    error_info = None
                    try:
                        error_json = json.loads(error_text)
                        from kiro.kiro_errors import enhance_kiro_error
                        error_info = enhance_kiro_error(error_json)
                        error_message = error_info.user_message
                    except (json.JSONDecodeError, KeyError):
                        pass

                    logger.warning(
                        f"WebSocket /v1/responses: Kiro API error {kiro_response.status_code}: "
                        f"{error_message[:100]}"
                    )
                    # Use the shared classifier so WS codex clients see the
                    # same error.code taxonomy as SSE clients.
                    classified = classify_upstream_error(
                        status_code=kiro_response.status_code,
                        error_info=error_info,
                        raw_message=error_text,
                    )
                    resp_id = _generate_response_id()
                    created_at = int(datetime.now(timezone.utc).timestamp())
                    failed_payload = build_response_failed_event(
                        response_id=resp_id,
                        created_at=created_at,
                        model=request_data.model,
                        error=classified,
                    )
                    await websocket.send_text(
                        _responses_ws_event("response.failed", failed_payload)
                    )
                    await http_client.close()
                    http_client = None
                    continue

                # Stream events to WebSocket
                tools_for_tokenizer = None
                if request_data.tools:
                    tools_for_tokenizer = [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in request_data.tools
                    ]

                from kiro.token_stats import UsageCollector
                ws_collector = UsageCollector()

                ws_canonical_input = _coerce_input_to_list(request_data.input)

                async def _ws_on_complete(resp_id: str, final_output: list) -> None:
                    await _persist_turn(
                        response_store,
                        resp_id,
                        request_data,
                        ws_canonical_input,
                        final_output,
                    )

                async for ws_msg in stream_kiro_to_responses_ws(
                    http_client.client,
                    kiro_response,
                    request_data.model,
                    model_cache,
                    auth_manager,
                    request_messages=None,
                    request_tools=tools_for_tokenizer,
                    usage_collector=ws_collector,
                    on_complete=_ws_on_complete,
                ):
                    await websocket.send_text(ws_msg)

                # Record token usage
                if ws_collector.input_tokens or ws_collector.output_tokens:
                    try:
                        websocket.app.state.token_stats.record(
                            ws_collector.model, ws_collector.input_tokens, ws_collector.output_tokens
                        )
                    except Exception as stats_err:
                        logger.debug(f"Failed to record token stats: {stats_err}")

                logger.info("WebSocket /v1/responses: turn completed")

            except WebSocketDisconnect:
                logger.debug("WebSocket /v1/responses: client disconnected during streaming")
                break
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else "(empty message)"
                logger.error(
                    f"WebSocket /v1/responses: streaming error [{error_type}] {error_msg}",
                    exc_info=True,
                )
                try:
                    await websocket.send_text(json.dumps({
                        "type": "response.failed",
                        "response": {"error": {"message": error_msg, "code": "server_error"}},
                    }))
                except Exception:
                    pass
                break
            finally:
                if http_client:
                    await http_client.close()
                    http_client = None

    except WebSocketDisconnect:
        logger.debug("WebSocket /v1/responses: client disconnected")
    except Exception as e:
        logger.error(f"WebSocket /v1/responses: unexpected error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket /v1/responses: connection closed")
