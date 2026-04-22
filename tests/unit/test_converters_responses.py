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
    _normalize_tool_output,
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
# Tests for _normalize_tool_output
# ==================================================================================================


class TestNormalizeToolOutput:
    """Tests for the _normalize_tool_output helper."""

    def test_plain_string_passthrough(self):
        """
        What it does: String inputs are returned unchanged.
        Purpose: The common, simple path must not be disturbed.
        """
        assert _normalize_tool_output("hello world") == "hello world"

    def test_empty_string_falls_back_to_placeholder(self):
        """
        What it does: Empty strings fall back to the "(empty result)" placeholder.
        Purpose: Preserves the pre-existing fallback contract; Kiro needs non-empty content.
        """
        assert _normalize_tool_output("") == "(empty result)"

    def test_none_falls_back_to_placeholder(self):
        """
        What it does: None falls back to the "(empty result)" placeholder.
        Purpose: Handle codex sending output=None for a failed/no-op tool call.
        """
        assert _normalize_tool_output(None) == "(empty result)"

    def test_list_of_output_text_blocks_concatenated(self):
        """
        What it does: Concatenates text from list-of-output_text blocks.
        Purpose: Codex sends this shape for large/structured tool outputs
                 (FunctionCallOutputPayload content_items form).
        """
        output = [
            {"type": "output_text", "text": "line one\n"},
            {"type": "output_text", "text": "line two"},
        ]
        assert _normalize_tool_output(output) == "line one\nline two"

    def test_list_of_mixed_block_types_extracted(self):
        """
        What it does: Extracts text from known block types even when intermixed.
        Purpose: Tolerate slight codex variations (input_text / text / output_text).
        """
        output = [
            {"type": "output_text", "text": "A"},
            {"type": "text", "text": "B"},
            {"type": "input_text", "text": "C"},
        ]
        assert _normalize_tool_output(output) == "ABC"

    def test_list_with_unknown_blocks_serialized_as_json(self):
        """
        What it does: Serializes unknown block shapes as JSON instead of dropping them.
        Purpose: A forward-compat guarantee so payload changes don't silently lose tool data.
        """
        output = [
            {"type": "output_text", "text": "known "},
            {"type": "future_block", "payload": {"k": "v"}},
        ]
        result = _normalize_tool_output(output)
        assert result.startswith("known ")
        # The unknown block is JSON-serialized, not str()-ified with single quotes.
        assert '"type": "future_block"' in result
        assert '"payload":' in result
        assert "'" not in result  # no Python-repr-style single quotes

    def test_list_with_raw_strings(self):
        """
        What it does: Accepts raw string elements in the list.
        Purpose: Some clients may send a list of plain strings.
        """
        assert _normalize_tool_output(["foo", "bar"]) == "foobar"

    def test_dict_with_text_field(self):
        """
        What it does: Extracts text from a bare dict with a ``text`` field.
        Purpose: Tolerate codex sending output as a single object rather than a list.
        """
        assert _normalize_tool_output({"type": "output_text", "text": "hi"}) == "hi"

    def test_dict_without_text_serialized_as_json(self):
        """
        What it does: Falls back to JSON for dicts without a text field.
        Purpose: Preserve payload instead of producing Python-repr garbage.
        """
        result = _normalize_tool_output({"status": "ok", "code": 200})
        assert '"status": "ok"' in result
        assert '"code": 200' in result

    def test_list_of_empty_blocks_falls_back(self):
        """
        What it does: A list where no block carries any text collapses to the placeholder.
        Purpose: Avoid emitting empty strings that break downstream Kiro validation.
        """
        output = [{"type": "output_text", "text": ""}]
        assert _normalize_tool_output(output) == "(empty result)"


# ==================================================================================================
# Tests for reasoning item handling (multi-turn bug)
# ==================================================================================================


class TestReasoningItemHandling:
    """
    Tests that verify reasoning items from codex are dropped and don't leak into
    user-visible messages. Relates to the multi-turn "forgets the question" bug.
    """

    def test_reasoning_item_alone_dropped(self):
        """
        What it does: A single ``reasoning`` item is dropped (no message emitted).
        Purpose: Reasoning is ephemeral model state; replaying it would corrupt the next turn.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Hello"},
                {
                    "type": "reasoning",
                    "id": "item_abc",
                    "summary": [],
                    "content": [
                        {"type": "reasoning_text", "text": "thinking..."}
                    ],
                    "encrypted_content": None,
                },
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        # Only the user message survives; reasoning item is dropped.
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        # The reasoning text must NOT appear anywhere in unified messages.
        for msg in messages:
            assert "thinking..." not in (msg.content or "")

    def test_reasoning_does_not_break_adjacent_messages(self):
        """
        What it does: Reasoning items between messages don't break neighbor semantics.
        Purpose: Dropping reasoning must be surgical, not disruptive.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Q1"},
                {
                    "type": "reasoning",
                    "id": "r1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "secret thoughts"}],
                },
                {"type": "message", "role": "assistant", "content": "A1"},
                {
                    "type": "reasoning",
                    "id": "r2",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "more secrets"}],
                },
                {"type": "message", "role": "user", "content": "Q2"},
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        assert [m.role for m in messages] == ["user", "assistant", "user"]
        assert [m.content for m in messages] == ["Q1", "A1", "Q2"]
        for msg in messages:
            assert "secret" not in (msg.content or "")

    def test_standalone_reasoning_text_item_dropped(self):
        """
        What it does: A bare ``reasoning_text`` input item is also dropped.
        Purpose: Defensive — some clients may emit it outside a ``reasoning`` wrapper.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "reasoning_text", "text": "raw reasoning"},
                {"type": "message", "role": "user", "content": "Hi"},
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        assert len(messages) == 1
        assert messages[0].content == "Hi"

    def test_reasoning_summary_text_item_dropped(self):
        """
        What it does: A bare ``reasoning_summary_text`` input item is dropped.
        Purpose: Defensive — older/newer codex variants.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "reasoning_summary_text", "text": "summary"},
                {"type": "message", "role": "user", "content": "Hi"},
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        assert len(messages) == 1
        assert messages[0].content == "Hi"


# ==================================================================================================
# Tests for function_call_output with structured content (multi-turn bug)
# ==================================================================================================


class TestFunctionCallOutputStructured:
    """
    Tests that verify function_call_output with list-form output is extracted correctly
    rather than stringified into Python list-repr garbage.
    """

    def test_function_call_output_string_unchanged(self):
        """
        What it does: String output passes through unchanged.
        Purpose: Ensure the normalization preserves the common path.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "run it"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "hello\nworld",
                },
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        tr = messages[2].tool_results[0]
        assert tr["content"] == "hello\nworld"

    def test_function_call_output_list_of_output_text(self):
        """
        What it does: List of output_text blocks is concatenated into a string.
        Purpose: Fix the bug where codex's structured output got stringified as Python list-repr.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "run it"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "output_text", "text": "file1.txt\n"},
                        {"type": "output_text", "text": "file2.txt"},
                    ],
                },
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        tr = messages[2].tool_results[0]
        assert tr["content"] == "file1.txt\nfile2.txt"
        # Must NOT be a Python list repr.
        assert "[{" not in tr["content"]
        assert "'type'" not in tr["content"]

    def test_function_call_output_mixed_block_types(self):
        """
        What it does: Mixed known/unknown block types keeps text and JSON-serializes unknowns.
        Purpose: Forward compatibility — never silently drop tool data.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "run it"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "output_text", "text": "known "},
                        {"type": "future_block", "data": {"x": 1}},
                    ],
                },
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        tr = messages[2].tool_results[0]
        assert tr["content"].startswith("known ")
        # Unknown block JSON-serialized, not Python-repr'd.
        assert '"type": "future_block"' in tr["content"]
        assert '"data":' in tr["content"]
        assert "'" not in tr["content"]

    def test_function_call_output_empty_string_placeholder_unchanged(self):
        """
        What it does: Empty-string output still maps to "(empty result)".
        Purpose: Don't regress the existing placeholder contract.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "run it"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "",
                },
            ],
        )
        _, messages = convert_responses_input_to_unified(req)
        assert messages[2].tool_results[0]["content"] == "(empty result)"


# ==================================================================================================
# End-to-end multi-turn codex-shape fixture
# ==================================================================================================


class TestCodexMultiTurnFixture:
    """
    Integration-flavored test using a fixture that mirrors the real codex replay shape
    observed in logs: user → reasoning → function_call → function_call_output → user.
    """

    def test_codex_replay_fixture_produces_clean_sequence(self):
        """
        What it does: Asserts the unified message sequence for a realistic codex replay.
        Purpose: Guards against regressions of both reasoning-leak and list-output bugs.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "List /tmp and /"}
                    ],
                },
                # Reasoning echo from the prior turn — must be silently dropped.
                {
                    "type": "reasoning",
                    "id": "item_reasoning_1",
                    "summary": [],
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "I should run ls /tmp first, then ls /.",
                        }
                    ],
                    "encrypted_content": None,
                },
                # Assistant's function call from the prior turn.
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_ls_tmp",
                    "name": "shell",
                    "arguments": '{"command":"ls /tmp"}',
                },
                # Tool result in the structured list form codex uses.
                {
                    "type": "function_call_output",
                    "call_id": "call_ls_tmp",
                    "output": [
                        {"type": "output_text", "text": "a.txt\nb.log\n"}
                    ],
                },
                # A follow-up user message in the same turn (rare but legal).
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "continue"}
                    ],
                },
            ],
        )
        _, messages = convert_responses_input_to_unified(req)

        # Expected sequence: user, assistant (tool_calls), user (tool_results), user.
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[0].content == "List /tmp and /"

        assert messages[1].role == "assistant"
        assert messages[1].tool_calls and len(messages[1].tool_calls) == 1
        assert messages[1].tool_calls[0]["id"] == "call_ls_tmp"
        assert messages[1].tool_calls[0]["function"]["name"] == "shell"

        assert messages[2].role == "user"
        assert messages[2].tool_results and len(messages[2].tool_results) == 1
        assert messages[2].tool_results[0]["tool_use_id"] == "call_ls_tmp"
        # Structured output extracted to a plain string.
        assert messages[2].tool_results[0]["content"] == "a.txt\nb.log\n"

        assert messages[3].role == "user"
        assert messages[3].content == "continue"

        # Reasoning content must NOT leak into any unified message.
        for msg in messages:
            content_str = str(msg.content or "")
            assert "should run ls /tmp first" not in content_str
            assert "reasoning_text" not in content_str
            # And no Python list repr leaking through either.
            assert "[{'type'" not in content_str


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


# ==================================================================================================
# Tests for merged-input shapes (previous_response_id resume)
# ==================================================================================================


class TestMergedInputForResume:
    """
    When the route handler merges a stored turn's input with the new
    delta items (codex's turn-N>=2 shape), the converter must produce
    a unified message sequence where prior ``function_call`` items and
    their matching ``function_call_output`` items land in adjacent
    assistant/user messages.
    """

    def test_merged_function_call_and_output_pair_up(self):
        """
        What it does: Merged input = [user msg, prior function_call,
            new function_call_output]. Convert and assert the result is
            [user, assistant-with-tool_calls, user-with-tool_results]
            and the call_id matches on both sides.
        Purpose: This is exactly the shape the resume path produces on
            turn 2. If pairing breaks, the model sees tool output with
            no matching call.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": "list /tmp then /",
                },
                {
                    "type": "function_call",
                    "id": "item_1",
                    "call_id": "call_abc",
                    "name": "bash",
                    "arguments": '{"cmd":"ls /tmp"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "file1.txt\nfile2.txt",
                },
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert "list /tmp" in messages[0].content

        # Assistant with a single tool_call.
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None
        assert len(messages[1].tool_calls) == 1
        assert messages[1].tool_calls[0]["id"] == "call_abc"
        assert messages[1].tool_calls[0]["function"]["name"] == "bash"

        # User with the matching tool_result.
        assert messages[2].role == "user"
        assert messages[2].tool_results is not None
        assert len(messages[2].tool_results) == 1
        assert messages[2].tool_results[0]["tool_use_id"] == "call_abc"
        assert messages[2].tool_results[0]["content"] == "file1.txt\nfile2.txt"

    def test_merged_multi_tool_round(self):
        """
        What it does: Merged input covers two tool rounds in the same
            assistant turn. Converter merges consecutive function_calls
            into one assistant message and the matching outputs into
            one user message — this is the expected behavior when the
            prior turn issued two parallel tool calls.
        Purpose: Verify the pairing logic holds for multi-call turns
            that codex assembles when the model issues parallel tools.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user",
                 "content": "run two commands"},
                {"type": "function_call", "id": "i1", "call_id": "c1",
                 "name": "bash", "arguments": '{"cmd":"a"}'},
                {"type": "function_call_output", "call_id": "c1",
                 "output": "OUT_A"},
                {"type": "function_call", "id": "i2", "call_id": "c2",
                 "name": "bash", "arguments": '{"cmd":"b"}'},
                {"type": "function_call_output", "call_id": "c2",
                 "output": "OUT_B"},
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        # Consecutive function_calls merge into the same assistant msg;
        # the matching outputs also merge into a single user msg.
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "user"]

        assert len(messages[1].tool_calls) == 2
        assert messages[1].tool_calls[0]["id"] == "c1"
        assert messages[1].tool_calls[1]["id"] == "c2"

        assert len(messages[2].tool_results) == 2
        tr_ids = {tr["tool_use_id"] for tr in messages[2].tool_results}
        assert tr_ids == {"c1", "c2"}
        tr_contents = {tr["content"] for tr in messages[2].tool_results}
        assert tr_contents == {"OUT_A", "OUT_B"}

    def test_merged_with_reasoning_items_are_dropped(self):
        """
        What it does: Merged input contains reasoning items between
            message and function_call. Convert and assert reasoning is
            silently dropped (converter already handles this).
        Purpose: Defense in depth — the replay sanitizer in
            response_store drops reasoning, but the converter must
            also be tolerant in case untrusted input reaches it.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "q"},
                {"type": "reasoning", "id": "r1",
                 "content": [{"type": "reasoning_text",
                              "text": "INTERNAL"}]},
                {"type": "function_call", "id": "i1", "call_id": "c1",
                 "name": "bash", "arguments": '{"cmd":"x"}'},
                {"type": "function_call_output", "call_id": "c1",
                 "output": "done"},
            ],
        )
        system_prompt, messages = convert_responses_input_to_unified(req)

        # No assistant message containing "INTERNAL" anywhere.
        joined = " ".join(
            str(m.content) for m in messages
        )
        assert "INTERNAL" not in joined
        # Pairing still works around the reasoning item.
        assert messages[1].tool_calls[0]["id"] == "c1"
        assert messages[2].tool_results[0]["tool_use_id"] == "c1"
