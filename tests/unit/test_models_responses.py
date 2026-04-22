# -*- coding: utf-8 -*-

"""
Unit tests for models_responses module.

Tests for Pydantic models used by the OpenAI Responses API:
- Request model validation
- Input item parsing
- Tool definitions
- Response model construction
"""

import pytest
import time

from kiro.models_responses import (
    ResponsesRequest,
    EasyInputMessage,
    FunctionCallInput,
    FunctionCallOutputInput,
    FunctionToolDefinition,
    ResponseObject,
    ResponseOutputMessage,
    ResponseFunctionCall,
    ResponseUsage,
    OutputTextBlock,
    InputTextContent,
    OutputTextContent,
)


# ==================================================================================================
# Tests for ResponsesRequest
# ==================================================================================================


class TestResponsesRequestValidation:
    """Tests for ResponsesRequest model validation."""

    def test_minimal_string_input(self):
        """
        What it does: Validates minimal request with string input.
        Purpose: Ensure simplest request format works.
        """
        req = ResponsesRequest(model="claude-sonnet-4", input="Hello")
        assert req.model == "claude-sonnet-4"
        assert req.input == "Hello"
        assert req.stream is False
        assert req.instructions is None
        assert req.tools is None

    def test_string_input_with_instructions(self):
        """
        What it does: Validates request with instructions.
        Purpose: Ensure instructions field is accepted.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
            instructions="You are a helpful assistant.",
        )
        assert req.instructions == "You are a helpful assistant."

    def test_list_input_with_messages(self):
        """
        What it does: Validates request with list of message items.
        Purpose: Ensure structured input format works.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi there!"},
                {"type": "message", "role": "user", "content": "How are you?"},
            ],
        )
        assert isinstance(req.input, list)
        assert len(req.input) == 3

    def test_list_input_with_function_calls(self):
        """
        What it does: Validates request with function call and output items.
        Purpose: Ensure tool calling round-trip format works.
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
        assert len(req.input) == 3

    def test_stream_flag(self):
        """
        What it does: Validates stream flag.
        Purpose: Ensure streaming can be enabled.
        """
        req = ResponsesRequest(model="claude-sonnet-4", input="Hello", stream=True)
        assert req.stream is True

    def test_with_function_tools(self):
        """
        What it does: Validates request with function tool definitions.
        Purpose: Ensure tools are accepted.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
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
        assert len(req.tools) == 1

    def test_stateless_fields_accepted(self):
        """
        What it does: Validates that stateless gateway fields are accepted but ignored.
        Purpose: Ensure compatibility with clients that send these fields.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
            previous_response_id="resp_abc123",
            store=True,
            metadata={"user_id": "123"},
        )
        assert req.previous_response_id == "resp_abc123"
        assert req.store is True
        assert req.metadata == {"user_id": "123"}

    def test_extra_fields_allowed(self):
        """
        What it does: Validates that unknown fields don't cause validation errors.
        Purpose: Ensure forward compatibility with new API fields.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
            some_future_field="value",
        )
        assert req.model == "claude-sonnet-4"

    def test_max_output_tokens(self):
        """
        What it does: Validates max_output_tokens field.
        Purpose: Ensure token limit can be set.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
            max_output_tokens=4096,
        )
        assert req.max_output_tokens == 4096

    def test_temperature_and_top_p(self):
        """
        What it does: Validates temperature and top_p fields.
        Purpose: Ensure sampling parameters are accepted.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4",
            input="Hello",
            temperature=0.7,
            top_p=0.9,
        )
        assert req.temperature == 0.7
        assert req.top_p == 0.9


# ==================================================================================================
# Tests for Input Item Models
# ==================================================================================================


class TestInputItemModels:
    """Tests for individual input item Pydantic models."""

    def test_easy_input_message_string_content(self):
        """
        What it does: Validates EasyInputMessage with string content.
        Purpose: Ensure simple message format works.
        """
        msg = EasyInputMessage(role="user", content="Hello")
        assert msg.type == "message"
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_easy_input_message_structured_content(self):
        """
        What it does: Validates EasyInputMessage with structured content.
        Purpose: Ensure content block list format works.
        """
        msg = EasyInputMessage(
            role="user",
            content=[
                {"type": "input_text", "text": "Hello"},
                {"type": "input_text", "text": " World"},
            ],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_function_call_input(self):
        """
        What it does: Validates FunctionCallInput model.
        Purpose: Ensure function call items parse correctly.
        """
        fc = FunctionCallInput(
            call_id="call_abc",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )
        assert fc.type == "function_call"
        assert fc.call_id == "call_abc"
        assert fc.name == "get_weather"
        assert fc.arguments == '{"location": "NYC"}'

    def test_function_call_output_input(self):
        """
        What it does: Validates FunctionCallOutputInput model.
        Purpose: Ensure function output items parse correctly.
        """
        fco = FunctionCallOutputInput(
            call_id="call_abc",
            output="72°F, sunny",
        )
        assert fco.type == "function_call_output"
        assert fco.call_id == "call_abc"
        assert fco.output == "72°F, sunny"


# ==================================================================================================
# Tests for Tool Models
# ==================================================================================================


class TestToolModels:
    """Tests for tool definition models."""

    def test_function_tool_definition(self):
        """
        What it does: Validates FunctionToolDefinition model.
        Purpose: Ensure function tools parse correctly.
        """
        tool = FunctionToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        assert tool.type == "function"
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a location"
        assert "properties" in tool.parameters

    def test_function_tool_minimal(self):
        """
        What it does: Validates minimal function tool (name only).
        Purpose: Ensure tools work without optional fields.
        """
        tool = FunctionToolDefinition(name="do_something")
        assert tool.name == "do_something"
        assert tool.description is None
        assert tool.parameters is None


# ==================================================================================================
# Tests for Response Models
# ==================================================================================================


class TestResponseModels:
    """Tests for response output models."""

    def test_response_output_message(self):
        """
        What it does: Validates ResponseOutputMessage construction.
        Purpose: Ensure message output items build correctly.
        """
        msg = ResponseOutputMessage(
            id="item_abc",
            content=[OutputTextBlock(text="Hello!")],
        )
        assert msg.type == "message"
        assert msg.role == "assistant"
        assert msg.status == "completed"
        assert len(msg.content) == 1
        assert msg.content[0].text == "Hello!"

    def test_response_function_call(self):
        """
        What it does: Validates ResponseFunctionCall construction.
        Purpose: Ensure function call output items build correctly.
        """
        fc = ResponseFunctionCall(
            id="item_abc",
            call_id="call_123",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )
        assert fc.type == "function_call"
        assert fc.call_id == "call_123"
        assert fc.name == "get_weather"

    def test_response_object(self):
        """
        What it does: Validates full ResponseObject construction.
        Purpose: Ensure complete response builds correctly.
        """
        resp = ResponseObject(
            id="resp_abc",
            model="claude-sonnet-4",
            output=[
                ResponseOutputMessage(
                    id="item_1",
                    content=[OutputTextBlock(text="Hello!")],
                )
            ],
            usage=ResponseUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        assert resp.id == "resp_abc"
        assert resp.object == "response"
        assert resp.status == "completed"
        assert len(resp.output) == 1
        assert resp.usage.total_tokens == 15

    def test_response_usage(self):
        """
        What it does: Validates ResponseUsage defaults.
        Purpose: Ensure usage defaults to zeros.
        """
        usage = ResponseUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
