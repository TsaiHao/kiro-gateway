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
Converters for transforming OpenAI Responses API format to Kiro format.

This module is an adapter layer that converts Responses API-specific formats
to the unified format used by converters_core.py.

Contains functions for:
- Converting Responses API input items to unified messages
- Converting Responses API tools to unified format
- Building Kiro payload from Responses API requests
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_responses import ResponsesRequest

from kiro.converters_core import (
    extract_text_content,
    UnifiedMessage,
    UnifiedTool,
    ThinkingConfig,
    build_kiro_payload as core_build_kiro_payload,
)


# ==================================================================================================
# Built-in Tool Stubs
# ==================================================================================================

# OpenAI Responses API defines built-in tools (web_search, file_search, code_interpreter)
# that Kiro doesn't support natively. We convert them to function tool stubs so the model
# can invoke them and the client handles execution. This avoids warning logs and ensures
# the model is aware these capabilities exist.

_BUILTIN_TOOL_STUBS: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "name": "web_search",
        "description": "Search the web for current information. Returns relevant search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                },
            },
            "required": ["query"],
        },
    },
    "file_search": {
        "name": "file_search",
        "description": "Search through uploaded files for relevant content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find in files.",
                },
            },
            "required": ["query"],
        },
    },
    "code_interpreter": {
        "name": "code_interpreter",
        "description": "Execute code to perform calculations, data analysis, or generate outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute.",
                },
            },
            "required": ["code"],
        },
    },
}


# ==================================================================================================
# Responses API Input Processing
# ==================================================================================================


def _extract_message_content(content: Any) -> str:
    """
    Extracts text from Responses API message content.

    Handles both string content and structured content lists
    with input_text / output_text blocks.

    Args:
        content: Message content (string or list of content blocks)

    Returns:
        Extracted text string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("input_text", "output_text", "text"):
                    text_parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                text_parts.append(block.text)
        return "".join(text_parts)

    return str(content) if content else ""


def _normalize_tool_output(output: Any) -> str:
    """
    Normalizes a ``function_call_output.output`` value to a string.

    Codex's ``FunctionCallOutputPayload`` serializes as either a plain string
    or a list of content-item blocks (``[{"type": "output_text", "text": "..."}]``)
    for large or structured tool outputs. Downstream code stringifies this blindly
    when it's not a string, which produces Python list-repr garbage
    (``[{'type': 'output_text', 'text': '...'}]``) as the tool result the model sees.

    This helper extracts the textual payload regardless of shape:

    - String → returned as-is.
    - List of blocks → concatenates the ``text`` field of each known block
      (``output_text``/``text``/``input_text``). Unknown blocks are serialized
      as JSON so information isn't silently lost.
    - Dict with a ``text`` field → that text.
    - Empty/None → ``"(empty result)"`` (preserves existing fallback).

    Args:
        output: Raw output value from a Responses API ``function_call_output`` item.

    Returns:
        A plain string suitable for ``UnifiedMessage.tool_results[*].content``.

    Examples:
        >>> _normalize_tool_output("72F, sunny")
        '72F, sunny'
        >>> _normalize_tool_output([{"type": "output_text", "text": "hello"}])
        'hello'
        >>> _normalize_tool_output("")
        '(empty result)'
    """
    if output is None:
        return "(empty result)"

    if isinstance(output, str):
        return output or "(empty result)"

    if isinstance(output, list):
        parts: List[str] = []
        for block in output:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("output_text", "text", "input_text"):
                    parts.append(block.get("text", ""))
                elif "text" in block and isinstance(block["text"], str):
                    # Tolerate untyped blocks that still carry a text field.
                    parts.append(block["text"])
                else:
                    # Unknown block shape — JSON-dump so the data survives
                    # instead of getting lost to Python repr downstream.
                    try:
                        parts.append(json.dumps(block, ensure_ascii=False))
                    except (TypeError, ValueError):
                        parts.append(str(block))
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text", "") or "")
            elif isinstance(block, str):
                parts.append(block)
            else:
                try:
                    parts.append(json.dumps(block, ensure_ascii=False))
                except (TypeError, ValueError):
                    parts.append(str(block))
        joined = "".join(parts)
        return joined or "(empty result)"

    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, str) and text:
            return text
        try:
            return json.dumps(output, ensure_ascii=False) or "(empty result)"
        except (TypeError, ValueError):
            return str(output) or "(empty result)"

    return str(output) or "(empty result)"


def convert_responses_input_to_unified(
    request: ResponsesRequest,
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Converts Responses API input to unified message format.

    Handles:
    - String input (single user message)
    - List of input items (message, function_call, function_call_output)
    - Instructions (extracted as system prompt)

    Args:
        request: ResponsesRequest object

    Returns:
        Tuple of (system_prompt, unified_messages)
    """
    system_prompt = request.instructions or ""
    unified_messages: List[UnifiedMessage] = []

    # Simple string input → single user message
    if isinstance(request.input, str):
        unified_messages.append(UnifiedMessage(role="user", content=request.input))
        logger.debug(
            f"Converted Responses API string input: "
            f"system_prompt_length={len(system_prompt)}"
        )
        return system_prompt, unified_messages

    # List of input items
    pending_tool_results: List[Dict[str, Any]] = []

    for item in request.input:
        item_dict = (
            item
            if isinstance(item, dict)
            else item.model_dump()
            if hasattr(item, "model_dump")
            else {}
        )
        item_type = item_dict.get("type", "")

        if item_type == "message":
            # Flush pending tool results before a new message
            if pending_tool_results:
                unified_messages.append(
                    UnifiedMessage(
                        role="user",
                        content="",
                        tool_results=pending_tool_results.copy(),
                    )
                )
                pending_tool_results.clear()

            role = item_dict.get("role", "user")
            content = _extract_message_content(item_dict.get("content", ""))

            # System/developer messages go to system prompt
            if role in ("system", "developer"):
                if system_prompt:
                    system_prompt += "\n" + content
                else:
                    system_prompt = content
                continue

            unified_messages.append(UnifiedMessage(role=role, content=content))

        elif item_type == "function_call":
            # Assistant made a function call — convert to assistant message with tool_calls
            call_id = item_dict.get("call_id", item_dict.get("id", ""))
            name = item_dict.get("name", "")
            arguments = item_dict.get("arguments", "{}")

            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }

            # Check if last message is already an assistant with tool_calls — merge
            if (
                unified_messages
                and unified_messages[-1].role == "assistant"
                and unified_messages[-1].tool_calls
            ):
                unified_messages[-1].tool_calls.append(tool_call)
            else:
                unified_messages.append(
                    UnifiedMessage(
                        role="assistant",
                        content="",
                        tool_calls=[tool_call],
                    )
                )

        elif item_type == "function_call_output":
            # Tool result — collect and attach to next user message.
            # NOTE: codex may send ``output`` as either a plain string or a list
            # of content-item blocks (see codex-rs/protocol/src/models.rs
            # FunctionCallOutputPayload). Normalize both into a plain string so
            # that nothing downstream ends up stringifying a Python list into
            # "[{'type': 'output_text', ...}]" and feeding that to the model.
            call_id = item_dict.get("call_id", "")
            output = item_dict.get("output", "")
            normalized_output = _normalize_tool_output(output)

            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": normalized_output,
                }
            )

        elif item_type in ("reasoning", "reasoning_text", "reasoning_summary_text"):
            # Codex replays the prior turn's output items back as the next turn's
            # input, including any ``reasoning`` items we emitted during streaming.
            # The model's own prior reasoning is ephemeral state; replaying it as
            # a user-visible message confuses the model and has been observed to
            # make it "forget the question" after a couple of tool rounds.
            # Drop silently. The assistant message / tool_call items around it
            # still carry the actionable conversation state.
            logger.debug(
                f"Dropping Responses API input item type '{item_type}' "
                f"(id={item_dict.get('id', '')!r}) — reasoning items are not replayed"
            )
            continue

        else:
            # Unknown item type — try to extract as message
            logger.debug(f"Unknown Responses API input item type: {item_type}")
            content = item_dict.get("content", item_dict.get("text", ""))
            if content:
                unified_messages.append(
                    UnifiedMessage(role="user", content=str(content))
                )

    # Flush remaining tool results
    if pending_tool_results:
        unified_messages.append(
            UnifiedMessage(
                role="user",
                content="",
                tool_results=pending_tool_results.copy(),
            )
        )

    logger.debug(
        f"Converted {len(request.input)} Responses API input items: "
        f"{len(unified_messages)} unified messages, "
        f"system_prompt_length={len(system_prompt)}"
    )

    # Ensure at least one user message exists (Codex sends empty input on initial handshake)
    if not unified_messages:
        unified_messages.append(UnifiedMessage(role="user", content="(empty)"))
        logger.debug("Added synthetic user message for empty Responses API input")

    return system_prompt, unified_messages


def convert_responses_tools_to_unified(
    tools: Optional[List[Any]],
) -> Optional[List[UnifiedTool]]:
    """
    Converts Responses API tools to unified format.

    Function tools are converted directly. Built-in tools (web_search,
    file_search, code_interpreter) are converted to function tool stubs
    so the model can invoke them and the client handles execution.

    Args:
        tools: List of tool definitions from the request

    Returns:
        List of UnifiedTool objects, or None if no valid tools
    """
    if not tools:
        return None

    unified_tools: List[UnifiedTool] = []

    for tool in tools:
        tool_dict = (
            tool
            if isinstance(tool, dict)
            else tool.model_dump()
            if hasattr(tool, "model_dump")
            else {}
        )
        tool_type = tool_dict.get("type", "")

        if tool_type == "function":
            unified_tools.append(
                UnifiedTool(
                    name=tool_dict.get("name", ""),
                    description=tool_dict.get("description"),
                    input_schema=tool_dict.get("parameters"),
                )
            )
        elif tool_type in _BUILTIN_TOOL_STUBS:
            stub = _BUILTIN_TOOL_STUBS[tool_type]
            unified_tools.append(
                UnifiedTool(
                    name=stub["name"],
                    description=stub["description"],
                    input_schema=stub["parameters"],
                )
            )
            logger.debug(
                f"Converted built-in tool '{tool_type}' to function stub '{stub['name']}'"
            )
        else:
            logger.debug(f"Unknown tool type '{tool_type}', skipping")

    return unified_tools if unified_tools else None


# ==================================================================================================
# Main Entry Point
# ==================================================================================================


def build_kiro_payload(
    request_data: ResponsesRequest,
    conversation_id: str,
    profile_arn: str,
) -> dict:
    """
    Builds complete payload for Kiro API from Responses API request.

    This is the main entry point for Responses API → Kiro conversion.
    Uses the core build_kiro_payload function with Responses-specific adapters.

    Args:
        request_data: Request in Responses API format
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        Payload dictionary for POST request to Kiro API

    Raises:
        ValueError: If there are no messages to send
    """
    # Convert input to unified format
    system_prompt, unified_messages = convert_responses_input_to_unified(request_data)

    # Convert tools to unified format
    unified_tools = convert_responses_tools_to_unified(request_data.tools)

    # Get model ID for Kiro API
    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)

    logger.debug(
        f"Converting Responses API request: model={request_data.model} -> {model_id}, "
        f"messages={len(unified_messages)}, "
        f"tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}"
    )

    # Use core function to build payload
    result = core_build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=ThinkingConfig(enabled=True, budget_tokens=None),
    )

    return result.payload
