# -*- coding: utf-8 -*-

"""
Unit tests for POST /v1/messages/count_tokens.

The endpoint exists because clients like Claude Code probe it before
sending a message to decide whether the prompt will fit in context.
Without it, the probe returns 404 and the SDK's next bytes trip uvicorn
into the "Invalid HTTP request received" noise we've been seeing.

Tests assert:
- auth parity with /v1/messages (x-api-key, Authorization Bearer, 401)
- basic token counting against tiktoken-backed estimator
- tools + system prompt are counted (not just messages)
- Pydantic validation rejects empty messages / bad shapes
- extras like anthropic-beta/anthropic-version headers are accepted
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiro.config import PROXY_API_KEY
from kiro.routes_anthropic import router


@pytest.fixture
def app():
    """Minimal FastAPI app hosting the Anthropic router only."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def auth_headers():
    return {"x-api-key": PROXY_API_KEY, "Content-Type": "application/json"}


# ==================================================================================================
# Happy path
# ==================================================================================================


class TestCountTokensHappyPath:
    """Basic shapes that should return a positive input_tokens."""

    def test_simple_user_message(self, app, auth_headers):
        """
        What it does: Counts a single user message.
        Purpose: Prove the endpoint runs end-to-end and returns the
            documented shape ({"input_tokens": int}).
        """
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers=auth_headers,
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "Hello world"}],
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert list(body.keys()) == ["input_tokens"]
        assert isinstance(body["input_tokens"], int)
        assert body["input_tokens"] > 0

    def test_longer_content_costs_more_tokens(self, app, auth_headers):
        """
        What it does: Compares two requests where one has much more text.
        Purpose: Sanity-check that the tokenizer is actually counting
            content rather than returning a fixed value.
        """
        def post(text: str) -> int:
            with TestClient(app) as client:
                r = client.post(
                    "/v1/messages/count_tokens",
                    headers=auth_headers,
                    json={
                        "model": "claude-sonnet-4",
                        "messages": [{"role": "user", "content": text}],
                    },
                )
            assert r.status_code == 200
            return r.json()["input_tokens"]

        small = post("hi")
        large = post("hello " * 400)
        assert large > small * 5

    def test_system_prompt_contributes(self, app, auth_headers):
        """
        What it does: Same messages, with vs without a long system prompt.
        Purpose: Ensure system prompt tokens are included — the streaming
            path uses estimate_request_tokens which should do this, but a
            regression here would silently undercount.
        """
        base = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "ok"}],
        }
        with TestClient(app) as client:
            r1 = client.post(
                "/v1/messages/count_tokens", headers=auth_headers, json=base
            )
            r2 = client.post(
                "/v1/messages/count_tokens",
                headers=auth_headers,
                json={**base, "system": "You are a helpful assistant. " * 50},
            )
        assert r1.status_code == r2.status_code == 200
        assert r2.json()["input_tokens"] > r1.json()["input_tokens"] + 50

    def test_tools_contribute(self, app, auth_headers):
        """
        What it does: Adds a tool definition and confirms token count grows.
        Purpose: Clients that pass tools (Claude Code, codex) rely on tool
            schemas being counted so they don't overrun the context window.
        """
        base = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "what is the weather"}],
        }
        with TestClient(app) as client:
            r1 = client.post(
                "/v1/messages/count_tokens", headers=auth_headers, json=base
            )
            r2 = client.post(
                "/v1/messages/count_tokens",
                headers=auth_headers,
                json={
                    **base,
                    "tools": [
                        {
                            "name": "get_weather",
                            "description": "Fetch the current weather for a location.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name",
                                    }
                                },
                                "required": ["location"],
                            },
                        }
                    ],
                },
            )
        assert r1.status_code == r2.status_code == 200
        assert r2.json()["input_tokens"] > r1.json()["input_tokens"]

    def test_cache_aware_system_blocks_accepted(self, app, auth_headers):
        """
        What it does: Sends the Anthropic cache-aware system prompt shape
            ([{type:"text", text:..., cache_control:...}, ...]).
        Purpose: Confirm we don't 422 on the block-list system variant
            Claude Code uses when prompt caching is enabled.
        """
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers=auth_headers,
                json={
                    "model": "claude-sonnet-4",
                    "system": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert r.status_code == 200, r.text
        assert r.json()["input_tokens"] > 0

    def test_anthropic_beta_header_accepted(self, app, auth_headers):
        """
        What it does: Sends anthropic-beta: token-counting-2024-11-01.
        Purpose: Claude Code includes this header; rejecting it would
            break them even though it doesn't change the count.
        """
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers={
                    **auth_headers,
                    "anthropic-beta": "token-counting-2024-11-01",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert r.status_code == 200


# ==================================================================================================
# Auth
# ==================================================================================================


class TestCountTokensAuth:
    """Parity with /v1/messages auth: x-api-key or Authorization Bearer."""

    def test_x_api_key_header_accepted(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers={"x-api-key": PROXY_API_KEY},
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert r.status_code == 200

    def test_bearer_token_accepted(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert r.status_code == 200

    def test_missing_auth_returns_401(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers={},
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert r.status_code == 401

    def test_wrong_key_returns_401(self, app):
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers={"x-api-key": "nope"},
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert r.status_code == 401


# ==================================================================================================
# Validation
# ==================================================================================================


class TestCountTokensValidation:
    """Pydantic-rejected shapes — mirror the 422s /v1/messages would raise."""

    def test_empty_messages_list_rejected(self, app, auth_headers):
        """min_length=1 on messages matches AnthropicMessagesRequest."""
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers=auth_headers,
                json={"model": "claude-sonnet-4", "messages": []},
            )
        assert r.status_code == 422

    def test_missing_model_rejected(self, app, auth_headers):
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers=auth_headers,
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 422

    def test_extra_fields_allowed(self, app, auth_headers):
        """
        What it does: Sends fields that aren't on our schema.
        Purpose: AnthropicCountTokensRequest has model_config extra="allow"
            so new anthropic fields don't break us.
        """
        with TestClient(app) as client:
            r = client.post(
                "/v1/messages/count_tokens",
                headers=auth_headers,
                json={
                    "model": "claude-sonnet-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "unknown_future_field": True,
                    "nested": {"blob": [1, 2, 3]},
                },
            )
        assert r.status_code == 200
