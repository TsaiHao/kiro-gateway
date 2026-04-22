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
Pydantic models for OpenAI Responses API (/v1/responses).

Defines data schemas for requests and responses compatible with
OpenAI's Responses API specification.

Reference: https://platform.openai.com/docs/api-reference/responses
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ==================================================================================================
# Input Item Models (request)
# ==================================================================================================


class InputTextContent(BaseModel):
    """Text content block within an input message."""

    type: Literal["input_text"] = "input_text"
    text: str


class InputImageContent(BaseModel):
    """Image content block within an input message (URL or base64)."""

    type: Literal["input_image"] = "input_image"
    image_url: Optional[str] = None
    detail: Optional[str] = None

    model_config = {"extra": "allow"}


class OutputTextContent(BaseModel):
    """
    Output text content block (used in input items to represent prior assistant output).
    """

    type: Literal["output_text"] = "output_text"
    text: str


# Union of content types that can appear inside a message input item
MessageContent = Union[
    InputTextContent, OutputTextContent, InputImageContent, Dict[str, Any]
]


class EasyInputMessage(BaseModel):
    """
    Easy input message format for the Responses API.

    Supports the simplified {"role": "...", "content": "..."} format
    as well as the structured content list format.

    Attributes:
        role: Message role (user, assistant, system, developer)
        content: Message content (string or list of content blocks)
    """

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    content: Union[str, List[MessageContent]]

    model_config = {"extra": "allow"}


class FunctionCallInput(BaseModel):
    """
    Function call input item (represents a prior function call by the model).

    Attributes:
        type: Always "function_call"
        id: Unique ID for this function call
        call_id: The call ID used to match with function_call_output
        name: Function name
        arguments: JSON string of function arguments
    """

    type: Literal["function_call"] = "function_call"
    id: Optional[str] = None
    call_id: str
    name: str
    arguments: str

    model_config = {"extra": "allow"}


class FunctionCallOutputInput(BaseModel):
    """
    Function call output input item (represents a tool result).

    Attributes:
        type: Always "function_call_output"
        call_id: The call ID this output corresponds to
        output: The string output of the function
    """

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str

    model_config = {"extra": "allow"}


# Union of all input item types
InputItem = Union[
    EasyInputMessage, FunctionCallInput, FunctionCallOutputInput, Dict[str, Any]
]


# ==================================================================================================
# Tool Models
# ==================================================================================================


class FunctionToolParameters(BaseModel):
    """Parameters schema for a function tool."""

    type: Optional[str] = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

    model_config = {"extra": "allow"}


class FunctionToolDefinition(BaseModel):
    """
    Function tool definition for the Responses API.

    Attributes:
        type: Always "function"
        name: Function name
        description: Function description
        parameters: JSON Schema for function parameters
        strict: Whether to enforce strict schema validation
    """

    type: Literal["function"] = "function"
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    model_config = {"extra": "allow"}


# Only function tools are supported (web_search, file_search, code_interpreter are not)
ResponseTool = Union[FunctionToolDefinition, Dict[str, Any]]


# ==================================================================================================
# Request Model
# ==================================================================================================


class ResponsesRequest(BaseModel):
    """
    Request to OpenAI Responses API (POST /v1/responses).

    Attributes:
        model: Model ID (e.g., "claude-sonnet-4")
        input: Input content - string, list of input items, or list of messages
        instructions: System instructions (prepended as system message)
        tools: List of tool definitions
        tool_choice: Tool selection strategy
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_output_tokens: Maximum tokens in response
        stream: Whether to stream the response
        previous_response_id: ID of previous response (not supported - stateless gateway)
        store: Whether to store the response (ignored - stateless gateway)
        metadata: Request metadata (ignored)
        truncation: Truncation strategy (ignored)
    """

    model: str
    input: Union[str, List[InputItem]]

    # Optional parameters
    instructions: Optional[str] = None
    tools: Optional[List[ResponseTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    stream: bool = False

    # Stateless gateway - these are accepted but not used
    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    truncation: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Response Output Item Models
# ==================================================================================================


class OutputTextBlock(BaseModel):
    """Text content block in a response output message."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: List[Any] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """
    Message output item in the response.

    Attributes:
        type: Always "message"
        id: Unique ID for this output item
        role: Always "assistant"
        status: Completion status
        content: List of content blocks
    """

    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: str = "completed"
    content: List[OutputTextBlock]


class ResponseFunctionCall(BaseModel):
    """
    Function call output item in the response.

    Attributes:
        type: Always "function_call"
        id: Unique ID for this output item
        call_id: The call ID for matching with function_call_output
        name: Function name
        arguments: JSON string of function arguments
        status: Completion status
    """

    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: str = "completed"


# Union of all output item types
OutputItem = Union[ResponseOutputMessage, ResponseFunctionCall]


# ==================================================================================================
# Usage Model
# ==================================================================================================


class ResponseUsage(BaseModel):
    """
    Token usage information for the Responses API.

    Attributes:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total number of tokens
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# ==================================================================================================
# Response Model
# ==================================================================================================


class ResponseObject(BaseModel):
    """
    Full response from the Responses API (non-streaming).

    Attributes:
        id: Unique response ID (prefixed with "resp_")
        object: Always "response"
        created_at: Creation timestamp
        model: Model used
        output: List of output items
        status: Response status
        usage: Token usage information
        error: Error information (if any)
    """

    id: str
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    output: List[OutputItem]
    status: str = "completed"
    usage: ResponseUsage
    error: Optional[Dict[str, Any]] = None

    # Fields accepted but not used by stateless gateway
    metadata: Dict[str, Any] = Field(default_factory=dict)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: List[Any] = Field(default_factory=list)
    incomplete_details: Optional[Dict[str, Any]] = None
