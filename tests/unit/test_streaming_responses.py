# -*- coding: utf-8 -*-

"""
Unit tests for streaming_responses module.

Tests for Responses API streaming logic:
- Converting Kiro events to Responses API SSE events
- Non-streaming response collection
- Event sequencing and format
- Tool call handling
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kiro.streaming_responses import (
    stream_kiro_to_responses,
    stream_kiro_to_responses_ws,
    collect_responses_response,
    _generate_response_id,
    _generate_item_id,
    _generate_call_id,
    _sse_event,
    _ws_event,
)
from kiro.responses_errors import (
    ResponsesError,
    build_response_failed_event,
    classify_exception,
    classify_upstream_error,
    CODE_CONTEXT_LENGTH,
    CODE_INSUFFICIENT_QUOTA,
    CODE_INVALID_PROMPT,
    CODE_RATE_LIMIT,
    CODE_SERVER_OVERLOADED,
    CODE_SERVER_ERROR,
)
from kiro.kiro_errors import KiroErrorInfo


# ==================================================================================================
# Tests for Helper Functions
# ==================================================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_generate_response_id_format(self):
        """
        What it does: Validates response ID format.
        Purpose: Ensure IDs have resp_ prefix.
        """
        resp_id = _generate_response_id()
        assert resp_id.startswith("resp_")
        assert len(resp_id) > 5

    def test_generate_item_id_format(self):
        """
        What it does: Validates item ID format.
        Purpose: Ensure IDs have item_ prefix.
        """
        item_id = _generate_item_id()
        assert item_id.startswith("item_")
        assert len(item_id) > 5

    def test_generate_call_id_format(self):
        """
        What it does: Validates call ID format.
        Purpose: Ensure IDs have call_ prefix.
        """
        call_id = _generate_call_id()
        assert call_id.startswith("call_")
        assert len(call_id) > 5

    def test_generate_unique_ids(self):
        """
        What it does: Validates that generated IDs are unique.
        Purpose: Ensure no collisions.
        """
        ids = {_generate_response_id() for _ in range(100)}
        assert len(ids) == 100

    def test_sse_event_format(self):
        """
        What it does: Validates SSE event formatting.
        Purpose: Ensure correct SSE format with event type and data,
            and that lifecycle events wrap payload under "response"
            with a top-level "type" field (the shape codex's
            ResponsesStreamEvent parser requires).
        """
        result = _sse_event("response.created", {"id": "resp_123"})
        assert result.startswith("event: response.created\n")
        assert "data: " in result
        assert result.endswith("\n\n")

        # Parse the data line
        lines = result.strip().split("\n")
        assert lines[0] == "event: response.created"
        data_line = lines[1]
        assert data_line.startswith("data: ")
        parsed = json.loads(data_line[6:])
        assert parsed["type"] == "response.created"
        assert parsed["response"]["id"] == "resp_123"

    def test_sse_event_non_lifecycle_spreads_payload(self):
        """
        What it does: Non-lifecycle events spread the payload as top-level keys.
        Purpose: Match codex ResponsesStreamEvent which reads fields like
            "delta", "item", and "content_index" from the top level.
        """
        result = _sse_event("response.output_text.delta", {"delta": "hi"})
        data_line = result.strip().split("\n")[1]
        parsed = json.loads(data_line[6:])
        assert parsed["type"] == "response.output_text.delta"
        assert parsed["delta"] == "hi"
        assert "response" not in parsed

    def test_sse_event_unicode(self):
        """
        What it does: Validates SSE event with unicode content.
        Purpose: Ensure non-ASCII characters are preserved.
        """
        result = _sse_event("response.output_text.delta", {"delta": "Привет мир"})
        assert "Привет мир" in result


# ==================================================================================================
# Tests for collect_responses_response (non-streaming)
# ==================================================================================================


class TestCollectResponsesResponse:
    """Tests for collect_responses_response function."""

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_simple_text_response(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Collects a simple text response.
        Purpose: Ensure basic non-streaming response works.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello, world!")
            yield KiroEvent(type="usage", usage={"credits": 1.0})

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        result = await collect_responses_response(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        )

        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["model"] == "claude-sonnet-4"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["role"] == "assistant"
        assert result["output"][0]["content"][0]["type"] == "output_text"
        assert result["output"][0]["content"][0]["text"] == "Hello, world!"
        assert "usage" in result

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=20)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 70, "test", "test"),
    )
    async def test_response_with_tool_calls(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Collects a response with tool calls.
        Purpose: Ensure tool calls are included in output.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me check.")
            yield KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                },
            )

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        result = await collect_responses_response(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        )

        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "message"
        assert result["output"][1]["type"] == "function_call"
        assert result["output"][1]["name"] == "get_weather"
        assert result["output"][1]["call_id"] == "call_abc"
        assert result["output"][1]["arguments"] == '{"location": "NYC"}'

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=0)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(0, 0, "unknown", "tiktoken"),
    )
    async def test_empty_response(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Handles empty response from Kiro.
        Purpose: Ensure empty streams don't crash.
        """

        async def mock_stream(*args, **kwargs):
            return
            yield  # Make it an async generator

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        result = await collect_responses_response(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        )

        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["output"] == []

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=15)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(0, 15, "unknown", "tiktoken"),
    )
    async def test_usage_fallback_to_tiktoken(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Falls back to tiktoken for token counting.
        Purpose: Ensure usage is calculated even without Kiro context_usage.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello!")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        result = await collect_responses_response(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
            request_messages=[{"role": "user", "content": "Hi"}],
        )

        assert result["usage"]["output_tokens"] == 15

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=5)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(10, 15, "test", "test"),
    )
    async def test_response_id_format(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Validates response ID format.
        Purpose: Ensure response IDs have resp_ prefix.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hi")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        result = await collect_responses_response(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        )

        assert result["id"].startswith("resp_")

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.FAKE_REASONING_HANDLING", "remove")
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(10, 20, "test", "test"),
    )
    async def test_thinking_content_removed_when_configured(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Removes thinking content when FAKE_REASONING_HANDLING is 'remove'.
        Purpose: Ensure thinking content is excluded from output.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="The answer is 42.")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        result = await collect_responses_response(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        )

        assert len(result["output"]) == 1
        # Only regular content, no thinking
        assert result["output"][0]["content"][0]["text"] == "The answer is 42."


# ==================================================================================================
# Tests for stream_kiro_to_responses (streaming)
# ==================================================================================================


class TestStreamKiroToResponses:
    """Tests for stream_kiro_to_responses streaming generator."""

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_streaming_event_sequence(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Validates the correct sequence of streaming events.
        Purpose: Ensure events are emitted in the right order.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello!")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        events = []
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            events.append(chunk)

        # Parse event types
        event_types = []
        for ev in events:
            for line in ev.strip().split("\n"):
                if line.startswith("event: "):
                    event_types.append(line[7:])

        # Verify event sequence
        assert "response.created" in event_types
        assert "response.in_progress" in event_types
        assert "response.output_item.added" in event_types
        assert "response.content_part.added" in event_types
        assert "response.output_text.delta" in event_types
        assert "response.content_part.done" in event_types
        assert "response.output_item.done" in event_types
        assert "response.completed" in event_types

        # Verify order: created before completed
        created_idx = event_types.index("response.created")
        completed_idx = event_types.index("response.completed")
        assert created_idx < completed_idx

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_streaming_text_deltas(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Validates text delta events contain correct content.
        Purpose: Ensure text is streamed correctly.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        deltas = []
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("event: response.output_text.delta"):
                    # Next line is data
                    continue
                if line.startswith("data: ") and "delta" in line:
                    data = json.loads(line[6:])
                    if "delta" in data:
                        deltas.append(data["delta"])

        assert "Hello" in deltas
        assert " World" in deltas

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_streaming_with_tool_calls(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Validates tool call events in streaming.
        Purpose: Ensure function calls are emitted as output items.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me check.")
            yield KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}',
                    },
                },
            )

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        events = []
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            events.append(chunk)

        # Find function_call in output_item events
        all_text = "".join(events)
        assert "function_call" in all_text
        assert "get_weather" in all_text

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_streaming_completed_event_has_usage(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Validates that response.completed event includes usage.
        Purpose: Ensure token usage is reported.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hi")
            yield KiroEvent(type="context_usage", context_usage_percentage=25.0)

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        completed_data = None
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data: ") and "completed" in chunk:
                    try:
                        data = json.loads(line[6:])
                        if (
                            data.get("type") == "response.completed"
                            and data.get("response", {}).get("status") == "completed"
                        ):
                            completed_data = data["response"]
                    except json.JSONDecodeError:
                        pass

        assert completed_data is not None
        assert "usage" in completed_data
        assert "input_tokens" in completed_data["usage"]
        assert "output_tokens" in completed_data["usage"]

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=0)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(0, 0, "unknown", "tiktoken"),
    )
    async def test_streaming_empty_response(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Handles empty stream gracefully.
        Purpose: Ensure empty responses still emit proper event sequence.
        """

        async def mock_stream(*args, **kwargs):
            return
            yield

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        events = []
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            events.append(chunk)

        # Should still have created and completed events
        all_text = "".join(events)
        assert "response.created" in all_text
        assert "response.completed" in all_text


# ==================================================================================================
# Regression test for codex ResponsesStreamEvent contract
# ==================================================================================================


class TestCodexSseContract:
    """
    Regression: codex's codex-api/src/sse/responses.rs parses each SSE data
    line as a ResponsesStreamEvent struct that requires a top-level "type"
    field and, for response-lifecycle events, the payload nested under
    "response". Events that don't match are silently dropped (continue), so
    codex eventually errors with "stream closed before response.completed"
    even though every HTTP response here returns 200.
    """

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_every_sse_data_line_matches_codex_schema(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Walks every chunk emitted by the streamer, parses the
            data line as JSON, and asserts: (a) each event has a "type"
            field, (b) the "type" matches the preceding "event:" line,
            (c) lifecycle events carry the payload under "response".
        Purpose: Guards the wire contract codex relies on. A regression here
            caused every /v1/responses turn to silently fail.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(
                type="tool_use",
                tool_use={
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                },
            )

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        lifecycle_types = {
            "response.created",
            "response.in_progress",
            "response.completed",
            "response.failed",
            "response.incomplete",
        }
        saw_completed = False
        parsed_events = []

        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            # Each chunk is one SSE event: "event: <type>\ndata: <json>\n\n"
            lines = chunk.strip().split("\n")
            assert len(lines) == 2, f"unexpected SSE shape: {chunk!r}"
            assert lines[0].startswith("event: ")
            event_header = lines[0][len("event: "):]
            assert lines[1].startswith("data: ")
            data = json.loads(lines[1][len("data: "):])

            # Codex requires a top-level "type" field.
            assert "type" in data, f"missing 'type' in data: {data!r}"
            # The event: header and data.type must agree.
            assert data["type"] == event_header, (
                f"event header {event_header!r} != data.type {data['type']!r}"
            )

            if data["type"] in lifecycle_types:
                # Codex deserializes ResponseCompleted etc. from data.response.
                assert "response" in data, (
                    f"lifecycle event {data['type']} missing 'response'"
                )
                assert isinstance(data["response"], dict)

            parsed_events.append(data)
            if data["type"] == "response.completed":
                saw_completed = True
                assert data["response"].get("status") == "completed"
                assert "usage" in data["response"]

        # The stream MUST reach response.completed; otherwise codex emits
        # "stream closed before response.completed" and retries.
        assert saw_completed, "response.completed missing — codex will retry"

        # Sanity: ordering — created before completed.
        event_order = [e["type"] for e in parsed_events]
        assert "response.created" in event_order
        created_idx = event_order.index("response.created")
        completed_idx = event_order.index("response.completed")
        assert created_idx < completed_idx


# ==================================================================================================
# Tests for reasoning / thinking stream forwarding
# ==================================================================================================


class TestReasoningStreamEvents:
    """
    Tests that thinking content from Kiro is forwarded as codex-compatible
    reasoning events instead of silently dropped.
    """

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_streaming_emits_reasoning_text_deltas(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Emits `response.reasoning_text.delta` for each thinking
            chunk, wraps them in a reasoning output_item.added/done pair, and
            places that reasoning item BEFORE the message item in both the
            stream and the final response.completed output list.
        Purpose: Codex parses reasoning events to render the model's thinking
            in its TUI; dropping them is a regression against the Responses
            API contract in codex-rs/codex-api/src/sse/responses.rs.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me ")
            yield KiroEvent(type="thinking", thinking_content="think...")
            yield KiroEvent(type="content", content="The answer is 42.")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        parsed = []
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            lines = chunk.strip().split("\n")
            data = json.loads(lines[1][len("data: "):])
            parsed.append(data)

        types = [e["type"] for e in parsed]

        # Reasoning deltas were emitted.
        reasoning_deltas = [
            e for e in parsed if e["type"] == "response.reasoning_text.delta"
        ]
        assert len(reasoning_deltas) == 2
        # Each delta carries the required fields at the top level.
        for ev in reasoning_deltas:
            assert "delta" in ev
            assert ev["content_index"] == 0
        # Concatenated deltas reproduce the full thinking text.
        assert "".join(ev["delta"] for ev in reasoning_deltas) == "Let me think..."

        # A reasoning output_item.added/done pair surrounds the deltas, and
        # both appear before the message output_item.added.
        reasoning_added_idx = None
        reasoning_done_idx = None
        message_added_idx = None
        for i, e in enumerate(parsed):
            if e["type"] == "response.output_item.added":
                item_type = e["item"]["type"]
                if item_type == "reasoning" and reasoning_added_idx is None:
                    reasoning_added_idx = i
                elif item_type == "message" and message_added_idx is None:
                    message_added_idx = i
            elif (
                e["type"] == "response.output_item.done"
                and e["item"]["type"] == "reasoning"
                and reasoning_done_idx is None
            ):
                reasoning_done_idx = i

        assert reasoning_added_idx is not None
        assert reasoning_done_idx is not None
        assert message_added_idx is not None
        assert reasoning_added_idx < reasoning_done_idx < message_added_idx

        # Reasoning deltas land between added and done.
        first_delta_idx = types.index("response.reasoning_text.delta")
        assert reasoning_added_idx < first_delta_idx < reasoning_done_idx

        # Final response.completed output carries reasoning before message.
        completed = next(e for e in parsed if e["type"] == "response.completed")
        out = completed["response"]["output"]
        assert out[0]["type"] == "reasoning"
        assert out[0]["content"][0]["type"] == "reasoning_text"
        assert out[0]["content"][0]["text"] == "Let me think..."
        assert out[1]["type"] == "message"

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.FAKE_REASONING_HANDLING", "remove")
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_streaming_reasoning_suppressed_when_removed(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: When FAKE_REASONING_HANDLING='remove', no reasoning
            events are emitted and the completed output contains no reasoning
            item — but token accounting still includes the thinking tokens.
        Purpose: Honor the existing escape hatch for users who explicitly
            want thinking stripped from output.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="hidden thoughts")
            yield KiroEvent(type="content", content="answer")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        parsed = []
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            parsed.append(json.loads(chunk.strip().split("\n")[1][len("data: "):]))

        # No reasoning events at all.
        assert not any(
            e["type"] == "response.reasoning_text.delta" for e in parsed
        )
        for e in parsed:
            if e["type"] in ("response.output_item.added", "response.output_item.done"):
                assert e["item"]["type"] != "reasoning"

        completed = next(e for e in parsed if e["type"] == "response.completed")
        assert all(
            it["type"] != "reasoning" for it in completed["response"]["output"]
        )

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_collect_responses_includes_reasoning_item(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Non-streaming collector builds a reasoning output item
            when thinking content is present, ordered before the message.
        Purpose: Parity with the streaming path so non-streaming callers also
            see the reasoning trace.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="step-by-step")
            yield KiroEvent(type="content", content="final answer")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        result = await collect_responses_response(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        )

        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "reasoning"
        assert result["output"][0]["summary"] == []
        assert result["output"][0]["content"][0]["type"] == "reasoning_text"
        assert result["output"][0]["content"][0]["text"] == "step-by-step"
        assert result["output"][1]["type"] == "message"
        assert result["output"][1]["content"][0]["text"] == "final answer"

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_streaming_reasoning_events_match_codex_schema(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Ensures every new reasoning SSE data line satisfies the
            codex wire contract — top-level `type` agrees with the `event:`
            header, deltas expose `delta` and `content_index` at the top
            level, and lifecycle events still nest under `response`.
        Purpose: Guard against regressions where reasoning events break the
            same parser the existing `TestCodexSseContract` test protects.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="hmm")
            yield KiroEvent(type="content", content="done")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        saw_reasoning_delta = False
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            lines = chunk.strip().split("\n")
            assert len(lines) == 2
            assert lines[0].startswith("event: ")
            header = lines[0][len("event: "):]
            data = json.loads(lines[1][len("data: "):])
            assert data["type"] == header
            if data["type"] == "response.reasoning_text.delta":
                saw_reasoning_delta = True
                assert isinstance(data["delta"], str)
                assert data["content_index"] == 0
                assert "response" not in data

        assert saw_reasoning_delta


# ==================================================================================================
# Tests for response.failed error classification
# ==================================================================================================


class TestResponsesErrorClassification:
    """
    Coverage for kiro.responses_errors.classify_upstream_error and
    classify_exception. These map Kiro API failures onto the codex SSE
    error.code vocabulary so codex's retry logic (fatal vs. backoff vs.
    retry) fires correctly.
    """

    @pytest.mark.parametrize(
        "status,reason,message,expected_code",
        [
            # Direct reason matches
            (400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD", "Input is too long.", CODE_CONTEXT_LENGTH),
            (429, "MONTHLY_REQUEST_COUNT", "Monthly limit hit.", CODE_INSUFFICIENT_QUOTA),
            # Message-based classification (no reason)
            (400, "UNKNOWN", "Input is too long and context exceeded.", CODE_CONTEXT_LENGTH),
            (429, "UNKNOWN", "Please slow down, rate limit hit.", CODE_RATE_LIMIT),
            (503, "UNKNOWN", "Server is overloaded, try later.", CODE_SERVER_OVERLOADED),
            (400, "UNKNOWN", "Improperly formed request.", CODE_INVALID_PROMPT),
            # Status-based fallback
            (429, "UNKNOWN", "", CODE_RATE_LIMIT),
            (500, "UNKNOWN", "", CODE_SERVER_OVERLOADED),
            (502, "UNKNOWN", "", CODE_SERVER_OVERLOADED),
            (400, "UNKNOWN", "", CODE_INVALID_PROMPT),
            (401, "UNKNOWN", "", CODE_INVALID_PROMPT),
            (403, "UNKNOWN", "", CODE_INVALID_PROMPT),
            # Total fallback — unknown status
            (418, "UNKNOWN", "", CODE_SERVER_ERROR),
        ],
    )
    def test_classify_upstream_error_matrix(self, status, reason, message, expected_code):
        """
        What it does: Matrix of (status, reason, message) → error.code.
        Purpose: Every codex-recognized code must be reachable; the matrix
            documents the classification contract.
        """
        info = KiroErrorInfo(
            reason=reason,
            user_message=message,
            original_message=message,
        )
        result = classify_upstream_error(
            status_code=status, error_info=info, raw_message=message
        )
        assert result.code == expected_code, (
            f"status={status}, reason={reason}, message={message!r}: "
            f"expected {expected_code}, got {result.code}"
        )

    def test_rate_limit_preserves_try_again_in_phrase(self):
        """
        What it does: If Kiro already provides "try again in 15s", preserve it.
        Purpose: Codex's regex in responses.rs:487 requires the exact wording.
        """
        info = KiroErrorInfo(
            reason="UNKNOWN",
            user_message="Rate limit hit. Please try again in 15s.",
            original_message="Rate limit hit. Please try again in 15s.",
        )
        result = classify_upstream_error(status_code=429, error_info=info)
        assert result.code == CODE_RATE_LIMIT
        assert "try again in 15s" in result.message.lower()

    def test_rate_limit_synthesizes_retry_wording_when_missing(self):
        """
        What it does: On a bare 429 with no retry hint, we synthesize
            "Please try again in 30s." so codex's regex finds a delay.
        Purpose: Without the phrase codex falls back to 0 delay and tight-loops.
        """
        result = classify_upstream_error(
            status_code=429,
            error_info=None,
            raw_message="Too many requests.",
        )
        assert result.code == CODE_RATE_LIMIT
        assert "try again in" in result.message.lower()
        # Matches codex's regex literally
        import re
        assert re.search(r"try again in\s*\d+(?:\.\d+)?\s*(s|ms|seconds?)",
                         result.message, re.IGNORECASE)

    def test_raw_message_used_when_no_error_info(self):
        """
        What it does: When Kiro returns non-JSON plaintext, raw_message drives
            the classifier.
        Purpose: Cover the error_info=None branch.
        """
        result = classify_upstream_error(
            status_code=500,
            error_info=None,
            raw_message="Server overload detected",
        )
        assert result.code == CODE_SERVER_OVERLOADED

    def test_message_fallback_when_empty(self):
        """
        What it does: Empty upstream message → synthesizes a generic one.
        Purpose: Never emit a failed event with an empty error.message —
            codex will still show it to the user.
        """
        result = classify_upstream_error(
            status_code=500, error_info=None, raw_message=""
        )
        assert result.message  # non-empty
        assert result.code == CODE_SERVER_OVERLOADED

    def test_classify_exception_timeout(self):
        """
        What it does: Timeout-ish exceptions map to server_is_overloaded.
        Purpose: Codex's ServerOverloaded is retryable — the right bucket for
            a transient timeout.
        """
        import httpx
        err = classify_exception(httpx.ReadTimeout("read timed out"))
        assert err.code == CODE_SERVER_OVERLOADED
        assert "read timed out" in err.message

    def test_classify_exception_fallback(self):
        """
        What it does: Unknown exception → server_error.
        Purpose: Cover the bottom of the classify_exception cascade.
        """
        err = classify_exception(RuntimeError("unexpected"))
        assert err.code == CODE_SERVER_ERROR
        assert "unexpected" in err.message


# ==================================================================================================
# Tests for mid-stream response.failed emission
# ==================================================================================================


class TestMidStreamFailure:
    """
    Guards the mid-stream failure path: when the Kiro stream raises after
    response.created has been emitted, the streamer must yield a terminal
    response.failed event before ending. Otherwise codex reads it as
    "stream closed before response.completed" and retries.
    """

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_midstream_emits_response_failed_after_content(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: Content chunk arrives, header is sent, then the Kiro
            stream raises. The streamer must yield response.failed as its
            final event, ordered after response.created.
        Purpose: Codex needs a terminal failed event to stop.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="partial")
            raise RuntimeError("kiro stream closed unexpectedly")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        parsed = []
        async for chunk in stream_kiro_to_responses(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            data = json.loads(chunk.strip().split("\n")[1][len("data: "):])
            parsed.append(data)

        types = [e["type"] for e in parsed]
        # created came first
        assert "response.created" in types
        # and response.failed is the last event
        assert types[-1] == "response.failed"
        created_idx = types.index("response.created")
        failed_idx = types.index("response.failed")
        assert created_idx < failed_idx
        # response.completed must NOT be emitted after a failure
        assert "response.completed" not in types

        # Wire-shape check: lifecycle event nests payload under "response"
        failed = parsed[-1]
        assert failed["type"] == "response.failed"
        assert "response" in failed
        assert failed["response"]["status"] == "failed"
        assert failed["response"]["error"]["code"]  # non-empty
        assert failed["response"]["error"]["message"]

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    async def test_midstream_without_header_reraises(self, mock_parse):
        """
        What it does: When the stream fails BEFORE any chunk — i.e., before
            response.created was emitted — the streamer re-raises rather than
            emitting response.failed.
        Purpose: If no header was sent, the HTTP layer can still produce a
            proper error response. We don't want to silently swallow pre-
            header exceptions.
        """
        async def mock_stream(*args, **kwargs):
            raise RuntimeError("boom before first token")
            yield  # pragma: no cover

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        with pytest.raises(RuntimeError, match="boom before first token"):
            async for _ in stream_kiro_to_responses(
                mock_client,
                mock_response,
                "claude-sonnet-4",
                mock_model_cache,
                mock_auth,
            ):
                pass

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    async def test_first_token_timeout_still_reraises(self, mock_parse):
        """
        What it does: Preserve FirstTokenTimeoutError re-raise so the retry
            wrapper can trigger a retry.
        Purpose: Regression guard — we only swallow post-header RuntimeErrors.
        """
        from kiro.streaming_core import FirstTokenTimeoutError

        async def mock_stream(*args, **kwargs):
            raise FirstTokenTimeoutError("first token timeout")
            yield  # pragma: no cover

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        with pytest.raises(FirstTokenTimeoutError):
            async for _ in stream_kiro_to_responses(
                mock_client,
                mock_response,
                "claude-sonnet-4",
                mock_model_cache,
                mock_auth,
            ):
                pass

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_ws_midstream_emits_response_failed(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: WS variant — same mid-stream failure path, but yields
            JSON WebSocket frames instead of SSE.
        Purpose: Parity with SSE path so WS codex clients don't retry-storm.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="partial")
            raise RuntimeError("kiro ws stream closed")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        frames = []
        async for frame in stream_kiro_to_responses_ws(
            mock_client,
            mock_response,
            "claude-sonnet-4",
            mock_model_cache,
            mock_auth,
        ):
            frames.append(json.loads(frame))

        types = [f["type"] for f in frames]
        assert "response.created" in types
        assert types[-1] == "response.failed"
        failed = frames[-1]
        assert failed["response"]["status"] == "failed"
        assert failed["response"]["error"]["code"]

    @pytest.mark.asyncio
    @patch("kiro.streaming_responses.parse_kiro_stream")
    @patch("kiro.streaming_responses.parse_bracket_tool_calls", return_value=[])
    @patch("kiro.streaming_responses.count_tokens", return_value=10)
    @patch(
        "kiro.streaming_responses.calculate_tokens_from_context_usage",
        return_value=(50, 60, "test", "test"),
    )
    async def test_midstream_failed_matches_codex_contract(
        self, mock_calc, mock_count, mock_bracket, mock_parse
    ):
        """
        What it does: The terminal response.failed event MUST satisfy the
            codex ResponsesStreamEvent wire contract — top-level "type" and
            the payload nested under "response" with shape
            {error: {code, message}, status: "failed", output: [], usage: null}.
        Purpose: Parity with TestCodexSseContract but for the failure path.
        """
        from kiro.streaming_core import KiroEvent

        async def mock_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="x")
            raise RuntimeError("input is too long, context exceeded")

        mock_parse.return_value = mock_stream()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.aclose = AsyncMock()
        mock_model_cache = MagicMock()
        mock_auth = MagicMock()

        last_chunk = None
        async for chunk in stream_kiro_to_responses(
            mock_client, mock_response, "claude-sonnet-4", mock_model_cache, mock_auth,
        ):
            last_chunk = chunk

        assert last_chunk is not None
        lines = last_chunk.strip().split("\n")
        assert lines[0] == "event: response.failed"
        data = json.loads(lines[1][len("data: "):])
        assert data["type"] == "response.failed"
        # Lifecycle event: payload nested under "response"
        assert "response" in data and isinstance(data["response"], dict)
        resp = data["response"]
        assert resp["status"] == "failed"
        assert resp["output"] == []
        assert resp["usage"] is None
        assert resp["incomplete_details"] is None
        # The classifier picked up "input is too long"
        assert resp["error"]["code"] == CODE_CONTEXT_LENGTH
        assert resp["error"]["message"]


# ==================================================================================================
# Tests for build_response_failed_event
# ==================================================================================================


class TestBuildResponseFailedEvent:
    """Tests for the payload builder."""

    def test_build_response_failed_event_shape(self):
        """
        What it does: Builder returns a dict with the exact fields codex
            expects, and no extras that would trip serde.
        Purpose: Wire-shape regression guard.
        """
        err = ResponsesError(code=CODE_CONTEXT_LENGTH, message="too long")
        payload = build_response_failed_event(
            response_id="resp_abc",
            created_at=123456,
            model="claude-sonnet-4",
            error=err,
        )
        assert payload == {
            "id": "resp_abc",
            "object": "response",
            "created_at": 123456,
            "model": "claude-sonnet-4",
            "status": "failed",
            "error": {"code": CODE_CONTEXT_LENGTH, "message": "too long"},
            "output": [],
            "usage": None,
            "incomplete_details": None,
        }

