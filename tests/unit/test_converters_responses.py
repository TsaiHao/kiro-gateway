# -*- coding: utf-8 -*-

"""
Unit tests for converters_responses module.

Tests for Responses API-specific conversion logic:
- Converting Responses API input items to unified messages
- Converting Responses API tools to unified format
- Building Kiro payload from Responses API requests
"""

import pytest
from unittest.mock import patch

from kiro.converters_responses import (
    build_kiro_payload,
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
    _extract_message_content,
)
from kiro.models_responses import ResponsesRequest
from kiro.converters_core import UnifiedMessage, UnifiedTool


# ==================================================================================================
# Tests for _extract_message_content
# ==================================================================================================


class TestExtractMessageContent:
    """Tests for _extract_message_content helper function."""

    def test_string_content(self):
        """
        What it does: Extracts text from plain string content.
        Purpose: Ensure simplest content format works.
        """
        assert _extract_message_content("Hello") == "Hello"

    def test_empty_string(self):
        """
        What it does: Handles empty string content.
        Purpose: Ensure empty strings don't cause errors.
        """
        assert _extract_message_content("") == ""

    def test_none_content(self):
        """
        What it does: Handles None content.
        Purpose: Ensure None returns empty string.
        """
        assert _extract_message_content(None) == ""

    def test_input_text_blocks(self):
        """
        What it does: Extracts text from input_text content blocks.
        Purpose: Ensure structured content format works.
        """
        content = [
            {"type": "input_text", "text": "Hello"},
            {"type": "input_text", "text": " World"},
        ]
        assert _extract_message_content(content) == "Hello World"

    def test_output_text_blocks(self):
        """
        What it does: Extracts text from output_text content blocks.
        Purpose: Ensure assistant output content blocks work.
        """
        content = [
            {"type": "output_text", "text": "Response text"},
        ]
        assert _extract_message_content(content) == "Response text"

    def test_mixed_content_blocks(self):
        """
        What it does: Extracts text from mixed content block types.
        Purpose: Ensure different block types are handled together.
        """
        content = [
            {"type": "input_text", "text": "Part 1"},
            {"type": "output_text", "text": "Part 2"},
            {"type": "text", "text": "Part 3"},
        ]
        assert _extract_message_content(content) == "Part 1Part 2Part 3"

    def test_unknown_block_types_skipped(self):
        """
        What it does: Skips unknown content block types.
        Purpose: Ensure unsupported blocks don't break extraction.
        """
        content = [
            {"type": "input_text", "text": "Hello"},
            {"type": "input_image", "image_url": "http://example.com/img.png"},
        ]
        assert _extract_message_content(content) == "Hello"

    def test_non_string_non_list_content(self):
        """
        What it does: Handles unexpected content types.
        Purpose: Ensure fallback to str() works.
        """
        assert _extract_message_content(42) == "42"


# ==================================================================================================
# Tests for convert_responses_input_to_unified
# ==================================================================================================


class TestConvertResponsesInputToUnified:
    """Tests for convert_responses_input_to_unified function."""

    def test_string_input(self):
        """
        What it does: Converts simple string input to single user message.
        Purpose: Ensure simplest input format works.
        """
        req = ResponsesRequest(model="claude-sonnet-4", input="Hello")
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert system_prompt == ""
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_string_input_with_instructions(self):
        """
        What it does: Converts string input with instructions to system prompt + user message.
        Purpose: Ensure instructions become system prompt.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
            instructions="Be helpful.",
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert system_prompt == "Be helpful."
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_message_items(self):
        """
        What it does: Converts list of message items to unified messages.
        Purpose: Ensure structured message input works.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi!"},
                {"type": "message", "role": "user", "content": "How are you?"},
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert system_prompt == ""
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi!"
        assert messages[2].role == "user"
        assert messages[2].content == "How are you?"

    def test_system_message_extracted_to_system_prompt(self):
        """
        What it does: Extracts system messages from input items to system prompt.
        Purpose: Ensure system role messages become system prompt.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "system", "content": "You are helpful."},
                {"type": "message", "role": "user", "content": "Hello"},
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert system_prompt == "You are helpful."
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_developer_message_extracted_to_system_prompt(self):
        """
        What it does: Extracts developer messages from input items to system prompt.
        Purpose: Ensure developer role messages become system prompt.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "developer", "content": "Context info."},
                {"type": "message", "role": "user", "content": "Hello"},
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert system_prompt == "Context info."
        assert len(messages) == 1

    def test_instructions_combined_with_system_messages(self):
        """
        What it does: Combines instructions with system messages in input.
        Purpose: Ensure both sources of system prompt are merged.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "system", "content": "Extra context."},
                {"type": "message", "role": "user", "content": "Hello"},
            ],
            instructions="Be helpful.",
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert "Be helpful." in system_prompt
        assert "Extra context." in system_prompt

    def test_function_call_creates_assistant_message(self):
        """
        What it does: Converts function_call items to assistant messages with tool_calls.
        Purpose: Ensure function calls are properly represented.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert len(messages) == 2
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None
        assert len(messages[1].tool_calls) == 1
        assert messages[1].tool_calls[0]["id"] == "call_123"
        assert messages[1].tool_calls[0]["function"]["name"] == "get_weather"
        assert (
            messages[1].tool_calls[0]["function"]["arguments"] == '{"location": "NYC"}'
        )

    def test_consecutive_function_calls_merged(self):
        """
        What it does: Merges consecutive function_call items into one assistant message.
        Purpose: Ensure parallel tool calls are grouped correctly.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Get weather and time"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_time",
                    "arguments": '{"timezone": "EST"}',
                },
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert len(messages) == 2
        assert messages[1].role == "assistant"
        assert len(messages[1].tool_calls) == 2

    def test_function_call_output_creates_tool_results(self):
        """
        What it does: Converts function_call_output items to user messages with tool_results.
        Purpose: Ensure tool results are properly represented.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "72°F, sunny",
                },
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert len(messages) == 3
        # Last message should be user with tool_results
        assert messages[2].role == "user"
        assert messages[2].tool_results is not None
        assert len(messages[2].tool_results) == 1
        assert messages[2].tool_results[0]["tool_use_id"] == "call_123"
        assert messages[2].tool_results[0]["content"] == "72°F, sunny"

    def test_empty_function_output_gets_placeholder(self):
        """
        What it does: Ensures empty function output gets placeholder content.
        Purpose: Kiro API requires non-empty content.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Do something"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "do_thing",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "",
                },
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        tool_result = messages[2].tool_results[0]
        assert tool_result["content"] == "(empty result)"

    def test_full_conversation_round_trip(self):
        """
        What it does: Tests a full conversation with messages, function calls, and results.
        Purpose: Ensure complete agent loop is properly converted.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "What's the weather in NYC?",
                },
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "72°F, sunny",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "The weather in NYC is 72°F and sunny.",
                },
                {"type": "message", "role": "user", "content": "Thanks!"},
            ],
            instructions="You are a weather assistant.",
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert system_prompt == "You are a weather assistant."
        assert len(messages) == 5
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None
        assert messages[2].role == "user"
        assert messages[2].tool_results is not None
        assert messages[3].role == "assistant"
        assert messages[3].content == "The weather in NYC is 72°F and sunny."
        assert messages[4].role == "user"
        assert messages[4].content == "Thanks!"

    def test_unknown_item_type_with_content(self):
        """
        What it does: Handles unknown input item types gracefully.
        Purpose: Ensure forward compatibility with new item types.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "some_future_type", "content": "Some content"},
                {"type": "message", "role": "user", "content": "Hello"},
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        # Unknown type with content should be converted to user message
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Some content"

    def test_multiple_function_outputs_collected(self):
        """
        What it does: Collects multiple consecutive function_call_output items.
        Purpose: Ensure parallel tool results are grouped.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Get both"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_time",
                    "arguments": '{"tz": "EST"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "72°F",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "3:00 PM",
                },
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        # Should have: user, assistant (2 tool_calls), user (2 tool_results)
        assert len(messages) == 3
        assert messages[1].role == "assistant"
        assert len(messages[1].tool_calls) == 2
        assert messages[2].role == "user"
        assert len(messages[2].tool_results) == 2


# ==================================================================================================
# Tests for convert_responses_tools_to_unified
# ==================================================================================================


class TestConvertResponsesToolsToUnified:
    """Tests for convert_responses_tools_to_unified function."""

    def test_none_tools(self):
        """
        What it does: Returns None for None tools.
        Purpose: Ensure no tools is handled.
        """
        assert convert_responses_tools_to_unified(None) is None

    def test_empty_tools(self):
        """
        What it does: Returns None for empty tools list.
        Purpose: Ensure empty list is handled.
        """
        assert convert_responses_tools_to_unified([]) is None

    def test_function_tool(self):
        """
        What it does: Converts function tool to unified format.
        Purpose: Ensure function tools are properly converted.
        """
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]
        result = convert_responses_tools_to_unified(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].description == "Get weather for a location"
        assert result[0].input_schema is not None

    def test_multiple_function_tools(self):
        """
        What it does: Converts multiple function tools.
        Purpose: Ensure all tools are converted.
        """
        tools = [
            {"type": "function", "name": "tool_a", "description": "Tool A"},
            {"type": "function", "name": "tool_b", "description": "Tool B"},
        ]
        result = convert_responses_tools_to_unified(tools)

        assert len(result) == 2
        assert result[0].name == "tool_a"
        assert result[1].name == "tool_b"

    def test_builtin_tools_converted_to_stubs(self):
        """
        What it does: Converts built-in tool types to function stubs.
        Purpose: Ensure web_search, file_search, code_interpreter become function tools.
        """
        tools = [
            {"type": "web_search"},
            {"type": "function", "name": "my_tool", "description": "My tool"},
            {"type": "code_interpreter"},
        ]
        result = convert_responses_tools_to_unified(tools)

        assert len(result) == 3
        assert result[0].name == "web_search"
        assert result[0].description is not None
        assert result[0].input_schema is not None
        assert result[1].name == "my_tool"
        assert result[2].name == "code_interpreter"
        assert result[2].input_schema is not None

    def test_only_builtin_tools_returns_stubs(self):
        """
        What it does: Returns stubs when all tools are built-in types.
        Purpose: Ensure built-in-only lists produce valid tool stubs.
        """
        tools = [
            {"type": "web_search"},
            {"type": "file_search"},
        ]
        result = convert_responses_tools_to_unified(tools)

        assert result is not None
        assert len(result) == 2
        assert result[0].name == "web_search"
        assert result[1].name == "file_search"

    def test_web_search_stub_has_query_parameter(self):
        """
        What it does: Validates web_search stub has a query parameter.
        Purpose: Ensure the stub schema is usable by the model.
        """
        tools = [{"type": "web_search"}]
        result = convert_responses_tools_to_unified(tools)

        assert result[0].name == "web_search"
        assert "query" in result[0].input_schema["properties"]

    def test_file_search_stub_has_query_parameter(self):
        """
        What it does: Validates file_search stub has a query parameter.
        Purpose: Ensure the stub schema is usable by the model.
        """
        tools = [{"type": "file_search"}]
        result = convert_responses_tools_to_unified(tools)

        assert result[0].name == "file_search"
        assert "query" in result[0].input_schema["properties"]

    def test_code_interpreter_stub_has_code_parameter(self):
        """
        What it does: Validates code_interpreter stub has a code parameter.
        Purpose: Ensure the stub schema is usable by the model.
        """
        tools = [{"type": "code_interpreter"}]
        result = convert_responses_tools_to_unified(tools)

        assert result[0].name == "code_interpreter"
        assert "code" in result[0].input_schema["properties"]

    def test_unknown_tool_type_skipped(self):
        """
        What it does: Skips truly unknown tool types.
        Purpose: Ensure only recognized types are converted.
        """
        tools = [
            {"type": "some_future_tool"},
            {"type": "function", "name": "my_tool", "description": "My tool"},
        ]
        result = convert_responses_tools_to_unified(tools)

        assert len(result) == 1
        assert result[0].name == "my_tool"

    def test_function_tool_without_parameters(self):
        """
        What it does: Converts function tool without parameters.
        Purpose: Ensure optional parameters field works.
        """
        tools = [
            {"type": "function", "name": "do_thing", "description": "Does a thing"},
        ]
        result = convert_responses_tools_to_unified(tools)

        assert len(result) == 1
        assert result[0].name == "do_thing"
        assert result[0].input_schema is None


# ==================================================================================================
# Tests for build_kiro_payload
# ==================================================================================================


class TestBuildKiroPayload:
    """Tests for build_kiro_payload function (Responses API → Kiro)."""

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_simple_string_input(self):
        """
        What it does: Builds Kiro payload from simple string input.
        Purpose: Ensure minimal request produces valid payload.
        """
        req = ResponsesRequest(model="claude-sonnet-4", input="Hello")
        payload = build_kiro_payload(req, "conv-123", "")

        assert "conversationState" in payload
        state = payload["conversationState"]
        assert state["conversationId"] == "conv-123"
        assert "currentMessage" in state
        current = state["currentMessage"]["userInputMessage"]
        assert "Hello" in current["content"]

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_with_instructions(self):
        """
        What it does: Builds Kiro payload with instructions as system prompt.
        Purpose: Ensure instructions are included in the payload.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
            instructions="Be helpful.",
        )
        payload = build_kiro_payload(req, "conv-123", "")

        current = payload["conversationState"]["currentMessage"]["userInputMessage"]
        # Instructions should be prepended to content
        assert "Be helpful." in current["content"]
        assert "Hello" in current["content"]

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_with_conversation_history(self):
        """
        What it does: Builds Kiro payload with conversation history.
        Purpose: Ensure multi-turn conversations produce history.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi!"},
                {"type": "message", "role": "user", "content": "How are you?"},
            ],
        )
        payload = build_kiro_payload(req, "conv-123", "")

        state = payload["conversationState"]
        assert "history" in state
        assert len(state["history"]) == 2  # First 2 messages in history
        assert "How are you?" in state["currentMessage"]["userInputMessage"]["content"]

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_with_tools(self):
        """
        What it does: Builds Kiro payload with tool definitions.
        Purpose: Ensure tools are included in the payload.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="What's the weather?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )
        payload = build_kiro_payload(req, "conv-123", "")

        current = payload["conversationState"]["currentMessage"]["userInputMessage"]
        assert "userInputMessageContext" in current
        assert "tools" in current["userInputMessageContext"]
        assert len(current["userInputMessageContext"]["tools"]) == 1

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_with_tool_results_in_history(self):
        """
        What it does: Builds Kiro payload with tool call/result round-trip in history.
        Purpose: Ensure tool interactions are properly represented in history.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "72°F, sunny",
                },
                {"type": "message", "role": "user", "content": "Thanks!"},
            ],
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )
        payload = build_kiro_payload(req, "conv-123", "")

        state = payload["conversationState"]
        assert "history" in state
        # Current message should be "Thanks!"
        assert "Thanks!" in state["currentMessage"]["userInputMessage"]["content"]

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_empty_input_raises_error(self):
        """
        What it does: Raises ValueError for empty input list.
        Purpose: Ensure empty conversations are rejected.
        """
        req = ResponsesRequest(model="claude-sonnet-4", input=[])
        with pytest.raises(ValueError, match="No messages to send"):
            build_kiro_payload(req, "conv-123", "")

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_profile_arn_included_when_provided(self):
        """
        What it does: Includes profileArn in payload when provided.
        Purpose: Ensure Kiro Desktop auth profileArn is passed through.
        """
        req = ResponsesRequest(model="claude-sonnet-4", input="Hello")
        payload = build_kiro_payload(req, "conv-123", "arn:aws:test:profile")

        assert payload.get("profileArn") == "arn:aws:test:profile"

    @patch("kiro.converters_responses.HIDDEN_MODELS", {})
    @patch("kiro.converters_core.FAKE_REASONING_ENABLED", False)
    @patch("kiro.config.TRUNCATION_RECOVERY", False)
    def test_profile_arn_omitted_when_empty(self):
        """
        What it does: Omits profileArn from payload when empty.
        Purpose: Ensure AWS SSO OIDC users don't get profileArn.
        """
        req = ResponsesRequest(model="claude-sonnet-4", input="Hello")
        payload = build_kiro_payload(req, "conv-123", "")

        assert "profileArn" not in payload
