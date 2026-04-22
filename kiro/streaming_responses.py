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
Streaming logic for converting Kiro stream to OpenAI Responses API format.

Contains generators for:
- Converting Kiro events to Responses API SSE events
- Forming streaming events (response.created, response.output_item.added, etc.)
- Processing tool calls in stream
- Collecting full non-streaming response

Uses streaming_core.py for parsing Kiro stream into unified KiroEvent objects.

Responses API streaming events (in order):
1. response.created
2. response.in_progress
3. response.output_item.added (for each output item)
4. response.content_part.added (for message content parts)
5. response.output_text.delta (text deltas)
6. response.content_part.done
7. response.output_item.done
8. response.completed
"""

import json
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Optional

import httpx
from loguru import logger

from kiro.parsers import parse_bracket_tool_calls, deduplicate_tool_calls
from kiro.config import (
    FIRST_TOKEN_TIMEOUT,
    FAKE_REASONING_HANDLING,
)
from kiro.tokenizer import count_tokens, count_message_tokens, count_tools_tokens

from kiro.streaming_core import (
    parse_kiro_stream,
    FirstTokenTimeoutError,
    calculate_tokens_from_context_usage,
)
from kiro.responses_errors import (
    ResponsesError,
    build_response_failed_event,
    classify_exception,
)

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

# Import debug_logger for logging
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


def _generate_response_id() -> str:
    """Generate a unique response ID with resp_ prefix."""
    return f"resp_{uuid.uuid4().hex}"


def _generate_item_id() -> str:
    """Generate a unique output item ID with item_ prefix."""
    return f"item_{uuid.uuid4().hex[:16]}"


def _generate_call_id() -> str:
    """Generate a unique call ID with call_ prefix."""
    return f"call_{uuid.uuid4().hex[:12]}"


_RESPONSE_LIFECYCLE_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.completed",
        "response.failed",
        "response.incomplete",
    }
)


def _sse_event(event_type: str, data: dict) -> str:
    """
    Format a Responses API SSE event.

    The JSON body carries the same shape the OpenAI Responses API emits:
    a top-level "type" field plus, for response-lifecycle events, the payload
    nested under "response". Other events spread the payload as top-level
    keys. Clients (for example the Codex CLI) deserialize the data line as
    a single struct and will silently drop events that don't match this
    shape, so keep the SSE body consistent with the WebSocket serializer.

    Args:
        event_type: The event type (e.g., "response.created")
        data: The event data dictionary

    Returns:
        Formatted SSE string
    """
    if event_type in _RESPONSE_LIFECYCLE_EVENTS:
        payload = {"type": event_type, "response": data}
    else:
        payload = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_kiro_to_responses(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    usage_collector=None,
) -> AsyncGenerator[str, None]:
    """
    Generator for converting Kiro stream to Responses API SSE format.

    Parses AWS SSE stream and converts events to Responses API streaming events.

    Args:
        client: HTTP client (for connection management)
        response: HTTP response with data stream
        model: Model name to include in response
        model_cache: Model cache for getting token limits
        auth_manager: Authentication manager
        first_token_timeout: First token wait timeout (seconds)
        request_messages: Original request messages (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)

    Yields:
        Strings in SSE format for Responses API streaming
    """
    response_id = _generate_response_id()
    created_at = int(time.time())
    message_item_id = _generate_item_id()
    reasoning_item_id = _generate_item_id()

    metering_data = None
    context_usage_percentage = None
    full_content = ""
    full_thinking_content = ""
    tool_calls_from_stream = []

    # Track state for SSE event sequencing
    header_sent = False
    message_item_started = False
    content_part_started = False
    reasoning_item_started = False
    reasoning_item_closed = False
    reasoning_output_index: Optional[int] = None
    output_item_index = 0
    content_part_index = 0
    # Emit reasoning unless the user explicitly configured it away.
    emit_reasoning = FAKE_REASONING_HANDLING != "remove"

    def _close_reasoning_item() -> Optional[str]:
        """Close the reasoning output item if it was started; return SSE chunk or None."""
        nonlocal reasoning_item_closed
        if not reasoning_item_started or reasoning_item_closed:
            return None
        reasoning_item_closed = True
        done_item = {
            "type": "reasoning",
            "id": reasoning_item_id,
            "summary": [],
            "content": [
                {"type": "reasoning_text", "text": full_thinking_content}
            ],
            "encrypted_content": None,
        }
        return _sse_event(
            "response.output_item.done",
            {"output_index": reasoning_output_index, "item": done_item},
        )

    # Base response object for events
    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": [],
        "status": "in_progress",
        "usage": None,
    }

    try:
        async for event in parse_kiro_stream(response, first_token_timeout):
            if event.type == "content" and event.content:
                full_content += event.content

                # Send header events on first content
                if not header_sent:
                    # response.created
                    yield _sse_event("response.created", base_response)
                    # response.in_progress
                    yield _sse_event("response.in_progress", base_response)
                    header_sent = True

                # Close any open reasoning item before starting the message.
                chunk = _close_reasoning_item()
                if chunk:
                    yield chunk
                    output_item_index += 1

                # Start message output item if not started
                if not message_item_started:
                    item_data = {
                        "type": "message",
                        "id": message_item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                    yield _sse_event(
                        "response.output_item.added",
                        {"output_index": output_item_index, "item": item_data},
                    )
                    message_item_started = True

                # Start content part if not started
                if not content_part_started:
                    part_data = {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    }
                    yield _sse_event(
                        "response.content_part.added",
                        {
                            "output_index": output_item_index,
                            "content_index": content_part_index,
                            "part": part_data,
                        },
                    )
                    content_part_started = True

                # Text delta
                yield _sse_event(
                    "response.output_text.delta",
                    {
                        "output_index": output_item_index,
                        "content_index": content_part_index,
                        "delta": event.content,
                    },
                )

                if debug_logger:
                    debug_logger.log_modified_chunk(event.content.encode("utf-8"))

            elif event.type == "thinking" and event.thinking_content:
                # Send header events on first thinking content
                if not header_sent:
                    yield _sse_event("response.created", base_response)
                    yield _sse_event("response.in_progress", base_response)
                    header_sent = True

                if not emit_reasoning:
                    # Config says drop thinking from output; still count tokens.
                    full_thinking_content += event.thinking_content
                    continue

                # Start reasoning output item on first thinking chunk.
                if not reasoning_item_started:
                    reasoning_output_index = output_item_index
                    item_data = {
                        "type": "reasoning",
                        "id": reasoning_item_id,
                        "summary": [],
                        "content": [],
                        "encrypted_content": None,
                    }
                    yield _sse_event(
                        "response.output_item.added",
                        {"output_index": reasoning_output_index, "item": item_data},
                    )
                    reasoning_item_started = True

                # Emit reasoning delta (spread at top level with content_index).
                yield _sse_event(
                    "response.reasoning_text.delta",
                    {
                        "output_index": reasoning_output_index,
                        "content_index": 0,
                        "delta": event.thinking_content,
                    },
                )
                full_thinking_content += event.thinking_content

            elif event.type == "tool_use" and event.tool_use:
                tool_calls_from_stream.append(event.tool_use)

            elif event.type == "usage" and event.usage:
                metering_data = event.usage

            elif (
                event.type == "context_usage"
                and event.context_usage_percentage is not None
            ):
                context_usage_percentage = event.context_usage_percentage

        logger.debug(
            f"[Responses stream] Kiro stream ended. "
            f"content_len={len(full_content)}, thinking_len={len(full_thinking_content)}, "
            f"tool_calls_from_stream={len(tool_calls_from_stream)}, "
            f"header_sent={header_sent}, message_started={message_item_started}, "
            f"content_part_started={content_part_started}"
        )

        # Ensure header was sent (edge case: empty response)
        if not header_sent:
            yield _sse_event("response.created", base_response)
            yield _sse_event("response.in_progress", base_response)
            header_sent = True

        # If thinking arrived but no content followed, close the reasoning item now.
        if reasoning_item_started and not reasoning_item_closed and not message_item_started:
            chunk = _close_reasoning_item()
            if chunk:
                yield chunk
                output_item_index += 1

        # Check bracket-style tool calls in full content
        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = tool_calls_from_stream + bracket_tool_calls
        all_tool_calls = deduplicate_tool_calls(all_tool_calls)
        logger.debug(
            f"[Responses stream] Tool calls: bracket={len(bracket_tool_calls)}, "
            f"total={len(all_tool_calls)}"
        )

        # Close content part and message item if they were started
        if content_part_started:
            yield _sse_event(
                "response.content_part.done",
                {
                    "output_index": output_item_index,
                    "content_index": content_part_index,
                    "part": {
                        "type": "output_text",
                        "text": full_content,
                        "annotations": [],
                    },
                },
            )

        if message_item_started:
            yield _sse_event(
                "response.output_item.done",
                {
                    "output_index": output_item_index,
                    "item": {
                        "type": "message",
                        "id": message_item_id,
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": full_content,
                                "annotations": [],
                            }
                        ],
                    },
                },
            )
            output_item_index += 1

        # Emit function call output items
        for tc in all_tool_calls:
            func = tc.get("function") or {}
            tool_name = func.get("name") or ""
            tool_args = func.get("arguments") or "{}"
            tool_id = tc.get("id") or _generate_item_id()
            call_id = tool_id  # In Responses API, call_id is used for matching

            fc_item = {
                "type": "function_call",
                "id": _generate_item_id(),
                "call_id": call_id,
                "name": tool_name,
                "arguments": tool_args,
                "status": "completed",
            }

            yield _sse_event(
                "response.output_item.added",
                {"output_index": output_item_index, "item": fc_item},
            )
            yield _sse_event(
                "response.output_item.done",
                {"output_index": output_item_index, "item": fc_item},
            )
            output_item_index += 1

        # Calculate usage
        logger.debug("[Responses stream] Calculating usage...")
        completion_tokens = count_tokens(full_content + full_thinking_content)
        prompt_tokens, total_tokens, prompt_source, total_source = (
            calculate_tokens_from_context_usage(
                context_usage_percentage, completion_tokens, model_cache, model
            )
        )

        if prompt_source == "unknown" and request_messages:
            prompt_tokens = count_message_tokens(
                request_messages, apply_claude_correction=False
            )
            if request_tools:
                prompt_tokens += count_tools_tokens(
                    request_tools, apply_claude_correction=False
                )
            total_tokens = prompt_tokens + completion_tokens
            prompt_source = "tiktoken"
            total_source = "tiktoken"

        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        logger.debug(
            f"[Usage] {model}: "
            f"input_tokens={prompt_tokens} ({prompt_source}), "
            f"output_tokens={completion_tokens} (tiktoken), "
            f"total_tokens={total_tokens} ({total_source})"
        )

        # Build final output list for the completed response
        final_output = []
        if reasoning_item_started and full_thinking_content:
            final_output.append(
                {
                    "type": "reasoning",
                    "id": reasoning_item_id,
                    "summary": [],
                    "content": [
                        {"type": "reasoning_text", "text": full_thinking_content}
                    ],
                    "encrypted_content": None,
                }
            )
        if full_content:
            final_output.append(
                {
                    "type": "message",
                    "id": message_item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": full_content,
                            "annotations": [],
                        }
                    ],
                }
            )

        for tc in all_tool_calls:
            func = tc.get("function") or {}
            tool_id = tc.get("id") or _generate_item_id()
            final_output.append(
                {
                    "type": "function_call",
                    "id": _generate_item_id(),
                    "call_id": tool_id,
                    "name": func.get("name") or "",
                    "arguments": func.get("arguments") or "{}",
                    "status": "completed",
                }
            )

        # response.completed
        completed_response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "output": final_output,
            "status": "completed",
            "usage": usage,
        }
        logger.debug(
            f"[Responses stream] Yielding response.completed "
            f"(output_items={len(final_output)}, usage={usage})"
        )
        if usage_collector:
            usage_collector.set(model, usage["input_tokens"], usage["output_tokens"])
        yield _sse_event("response.completed", completed_response)
        logger.debug("[Responses stream] response.completed sent successfully")

    except FirstTokenTimeoutError:
        # Preserve existing retry semantics for first-token timeout — the
        # caller (stream_with_first_token_retry wrapper pattern) relies on
        # this exception to trigger a retry before any event has been sent.
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit)")
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty message)"
        logger.error(
            f"Error during Responses API streaming: [{error_type}] {error_msg}",
            exc_info=True,
        )
        # If the header has already reached the client, emit a terminal
        # response.failed event so codex stops instead of treating the
        # truncated stream as "stream closed before response.completed"
        # (which it retries indefinitely).
        if header_sent:
            try:
                err = classify_exception(e)
                failed_payload = build_response_failed_event(
                    response_id=response_id,
                    created_at=created_at,
                    model=model,
                    error=err,
                )
                yield _sse_event("response.failed", failed_payload)
                logger.debug(
                    f"[Responses stream] emitted response.failed "
                    f"(code={err.code})"
                )
                # We handled it — do not re-raise so the StreamingResponse
                # finishes cleanly; routes_openai's stream_wrapper logs
                # this as a completed stream.
                return
            except Exception as emit_err:
                logger.debug(f"Failed to emit response.failed: {emit_err}")
        raise
    finally:
        try:
            await response.aclose()
        except Exception as close_error:
            logger.debug(f"Error closing response: {close_error}")


def _ws_event(event_type: str, data: dict) -> str:
    """
    Format a Responses API WebSocket event.

    WebSocket events are plain JSON with a "type" field.
    Response-level events nest data under "response",
    other events spread data as top-level keys.

    Args:
        event_type: The event type (e.g., "response.created")
        data: The event data dictionary

    Returns:
        JSON string for WebSocket text message
    """
    # Response-level events: wrap data under "response" key
    if event_type in _RESPONSE_LIFECYCLE_EVENTS:
        return json.dumps(
            {"type": event_type, "response": data}, ensure_ascii=False
        )
    # All other events: spread data as top-level keys
    return json.dumps({"type": event_type, **data}, ensure_ascii=False)


async def stream_kiro_to_responses_ws(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    usage_collector=None,
) -> AsyncGenerator[str, None]:
    """
    Generator for converting Kiro stream to Responses API WebSocket format.

    Same logic as stream_kiro_to_responses but yields JSON strings
    for WebSocket text messages instead of SSE-formatted strings.

    Args:
        client: HTTP client (for connection management)
        response: HTTP response with data stream
        model: Model name to include in response
        model_cache: Model cache for getting token limits
        auth_manager: Authentication manager
        first_token_timeout: First token wait timeout (seconds)
        request_messages: Original request messages (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)

    Yields:
        JSON strings for WebSocket text messages
    """
    response_id = _generate_response_id()
    created_at = int(time.time())
    message_item_id = _generate_item_id()
    reasoning_item_id = _generate_item_id()

    metering_data = None
    context_usage_percentage = None
    full_content = ""
    full_thinking_content = ""
    tool_calls_from_stream = []

    header_sent = False
    message_item_started = False
    content_part_started = False
    reasoning_item_started = False
    reasoning_item_closed = False
    reasoning_output_index: Optional[int] = None
    output_item_index = 0
    content_part_index = 0
    emit_reasoning = FAKE_REASONING_HANDLING != "remove"

    def _ws_close_reasoning_item() -> Optional[str]:
        nonlocal reasoning_item_closed
        if not reasoning_item_started or reasoning_item_closed:
            return None
        reasoning_item_closed = True
        done_item = {
            "type": "reasoning",
            "id": reasoning_item_id,
            "summary": [],
            "content": [
                {"type": "reasoning_text", "text": full_thinking_content}
            ],
            "encrypted_content": None,
        }
        return _ws_event(
            "response.output_item.done",
            {"output_index": reasoning_output_index, "item": done_item},
        )

    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": [],
        "status": "in_progress",
        "usage": None,
    }

    try:
        async for event in parse_kiro_stream(response, first_token_timeout):
            if event.type == "content" and event.content:
                full_content += event.content

                if not header_sent:
                    yield _ws_event("response.created", base_response)
                    header_sent = True

                chunk = _ws_close_reasoning_item()
                if chunk:
                    yield chunk
                    output_item_index += 1

                if not message_item_started:
                    item_data = {
                        "type": "message",
                        "id": message_item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                    yield _ws_event(
                        "response.output_item.added",
                        {"output_index": output_item_index, "item": item_data},
                    )
                    message_item_started = True

                yield _ws_event(
                    "response.output_text.delta",
                    {
                        "output_index": output_item_index,
                        "content_index": content_part_index,
                        "delta": event.content,
                    },
                )

                if debug_logger:
                    debug_logger.log_modified_chunk(event.content.encode("utf-8"))

            elif event.type == "thinking" and event.thinking_content:
                if not header_sent:
                    yield _ws_event("response.created", base_response)
                    header_sent = True

                if not emit_reasoning:
                    full_thinking_content += event.thinking_content
                    continue

                if not reasoning_item_started:
                    reasoning_output_index = output_item_index
                    item_data = {
                        "type": "reasoning",
                        "id": reasoning_item_id,
                        "summary": [],
                        "content": [],
                        "encrypted_content": None,
                    }
                    yield _ws_event(
                        "response.output_item.added",
                        {"output_index": reasoning_output_index, "item": item_data},
                    )
                    reasoning_item_started = True

                yield _ws_event(
                    "response.reasoning_text.delta",
                    {
                        "output_index": reasoning_output_index,
                        "content_index": 0,
                        "delta": event.thinking_content,
                    },
                )
                full_thinking_content += event.thinking_content

            elif event.type == "tool_use" and event.tool_use:
                tool_calls_from_stream.append(event.tool_use)

            elif event.type == "usage" and event.usage:
                metering_data = event.usage

            elif (
                event.type == "context_usage"
                and event.context_usage_percentage is not None
            ):
                context_usage_percentage = event.context_usage_percentage

        logger.debug(
            f"[Responses WS] Kiro stream ended. "
            f"content_len={len(full_content)}, thinking_len={len(full_thinking_content)}, "
            f"tool_calls={len(tool_calls_from_stream)}"
        )

        if not header_sent:
            yield _ws_event("response.created", base_response)
            header_sent = True

        # If thinking arrived but no content followed, close the reasoning item now.
        if reasoning_item_started and not reasoning_item_closed and not message_item_started:
            chunk = _ws_close_reasoning_item()
            if chunk:
                yield chunk
                output_item_index += 1

        # Check bracket-style tool calls
        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = tool_calls_from_stream + bracket_tool_calls
        all_tool_calls = deduplicate_tool_calls(all_tool_calls)

        # Close message item if started
        if message_item_started:
            yield _ws_event(
                "response.output_item.done",
                {
                    "output_index": output_item_index,
                    "item": {
                        "type": "message",
                        "id": message_item_id,
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": full_content,
                                "annotations": [],
                            }
                        ],
                    },
                },
            )
            output_item_index += 1

        # Emit function call output items
        for tc in all_tool_calls:
            func = tc.get("function") or {}
            tool_name = func.get("name") or ""
            tool_args = func.get("arguments") or "{}"
            tool_id = tc.get("id") or _generate_item_id()
            call_id = tool_id

            fc_item = {
                "type": "function_call",
                "id": _generate_item_id(),
                "call_id": call_id,
                "name": tool_name,
                "arguments": tool_args,
                "status": "completed",
            }

            yield _ws_event(
                "response.output_item.done",
                {"output_index": output_item_index, "item": fc_item},
            )
            output_item_index += 1

        # Calculate usage
        completion_tokens = count_tokens(full_content + full_thinking_content)
        prompt_tokens, total_tokens, prompt_source, total_source = (
            calculate_tokens_from_context_usage(
                context_usage_percentage, completion_tokens, model_cache, model
            )
        )

        if prompt_source == "unknown" and request_messages:
            prompt_tokens = count_message_tokens(
                request_messages, apply_claude_correction=False
            )
            if request_tools:
                prompt_tokens += count_tools_tokens(
                    request_tools, apply_claude_correction=False
                )
            total_tokens = prompt_tokens + completion_tokens

        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        logger.debug(
            f"[Responses WS] Usage: {model}: "
            f"input={prompt_tokens}, output={completion_tokens}, total={total_tokens}"
        )

        # Build final output for completed response
        final_output = []
        if reasoning_item_started and full_thinking_content:
            final_output.append(
                {
                    "type": "reasoning",
                    "id": reasoning_item_id,
                    "summary": [],
                    "content": [
                        {"type": "reasoning_text", "text": full_thinking_content}
                    ],
                    "encrypted_content": None,
                }
            )
        if full_content:
            final_output.append(
                {
                    "type": "message",
                    "id": message_item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": full_content,
                            "annotations": [],
                        }
                    ],
                }
            )

        for tc in all_tool_calls:
            func = tc.get("function") or {}
            tool_id = tc.get("id") or _generate_item_id()
            final_output.append(
                {
                    "type": "function_call",
                    "id": _generate_item_id(),
                    "call_id": tool_id,
                    "name": func.get("name") or "",
                    "arguments": func.get("arguments") or "{}",
                    "status": "completed",
                }
            )

        completed_response = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "output": final_output,
            "status": "completed",
            "usage": usage,
        }
        if usage_collector:
            usage_collector.set(model, usage["input_tokens"], usage["output_tokens"])
        yield _ws_event("response.completed", completed_response)
        logger.debug("[Responses WS] response.completed sent")

    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("[Responses WS] Client disconnected (GeneratorExit)")
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty message)"
        logger.error(
            f"Error during Responses WS streaming: [{error_type}] {error_msg}",
            exc_info=True,
        )
        # Mid-stream failure after the header was sent: emit a terminal
        # response.failed WS frame so codex stops retrying.
        if header_sent:
            try:
                err = classify_exception(e)
                failed_payload = build_response_failed_event(
                    response_id=response_id,
                    created_at=created_at,
                    model=model,
                    error=err,
                )
                yield _ws_event("response.failed", failed_payload)
                logger.debug(
                    f"[Responses WS] emitted response.failed (code={err.code})"
                )
                return
            except Exception as emit_err:
                logger.debug(f"[Responses WS] failed to emit response.failed: {emit_err}")
        raise
    finally:
        try:
            await response.aclose()
        except Exception as close_error:
            logger.debug(f"Error closing response: {close_error}")


async def collect_responses_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> dict:
    """
    Collect full response from Kiro stream in Responses API format.

    Used for non-streaming mode — collects all events and forms
    a single ResponseObject.

    Args:
        client: HTTP client
        response: HTTP response with stream
        model: Model name
        model_cache: Model cache
        auth_manager: Authentication manager
        request_messages: Original request messages (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)

    Returns:
        Dictionary with full response in Responses API format
    """
    response_id = _generate_response_id()
    created_at = int(time.time())

    full_content = ""
    full_thinking_content = ""
    metering_data = None
    context_usage_percentage = None
    tool_calls_from_stream = []

    async for event in parse_kiro_stream(response, FIRST_TOKEN_TIMEOUT):
        if event.type == "content" and event.content:
            full_content += event.content
        elif event.type == "thinking" and event.thinking_content:
            full_thinking_content += event.thinking_content
        elif event.type == "tool_use" and event.tool_use:
            tool_calls_from_stream.append(event.tool_use)
        elif event.type == "usage" and event.usage:
            metering_data = event.usage
        elif (
            event.type == "context_usage" and event.context_usage_percentage is not None
        ):
            context_usage_percentage = event.context_usage_percentage

    # Check bracket-style tool calls
    bracket_tool_calls = parse_bracket_tool_calls(full_content)
    all_tool_calls = tool_calls_from_stream + bracket_tool_calls
    all_tool_calls = deduplicate_tool_calls(all_tool_calls)

    # Calculate usage
    completion_tokens = count_tokens(full_content + full_thinking_content)
    prompt_tokens, total_tokens, prompt_source, total_source = (
        calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, model_cache, model
        )
    )

    if prompt_source == "unknown" and request_messages:
        prompt_tokens = count_message_tokens(
            request_messages, apply_claude_correction=False
        )
        if request_tools:
            prompt_tokens += count_tools_tokens(
                request_tools, apply_claude_correction=False
            )
        total_tokens = prompt_tokens + completion_tokens

    # Build output items
    output = []

    # Reasoning output item (emitted before the assistant message) unless disabled.
    emit_reasoning = FAKE_REASONING_HANDLING != "remove"
    if emit_reasoning and full_thinking_content:
        output.append(
            {
                "type": "reasoning",
                "id": _generate_item_id(),
                "summary": [],
                "content": [
                    {"type": "reasoning_text", "text": full_thinking_content}
                ],
                "encrypted_content": None,
            }
        )

    # Message output item
    if full_content:
        output.append(
            {
                "type": "message",
                "id": _generate_item_id(),
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": full_content,
                        "annotations": [],
                    }
                ],
            }
        )

    # Function call output items
    for tc in all_tool_calls:
        func = tc.get("function") or {}
        tool_id = tc.get("id") or _generate_item_id()
        output.append(
            {
                "type": "function_call",
                "id": _generate_item_id(),
                "call_id": tool_id,
                "name": func.get("name") or "",
                "arguments": func.get("arguments") or "{}",
                "status": "completed",
            }
        )

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": output,
        "status": "completed",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
