# -*- coding: utf-8 -*-

"""
Unit tests for /v1/responses/compact.

Codex calls this endpoint to summarize a long conversation into a handoff
message. The wire shape:

Request (codex CompactionInput — superset of ResponsesRequest):
    {
      "model": "...",
      "input": [ResponseItem, ...],
      "instructions": "...",
      "tools": [...],
      "parallel_tool_calls": bool,
      "reasoning": {...},
      "text": {...}
    }

Response (codex CompactHistoryResponse):
    {"output": [ResponseItem, ...]}

These tests cover:
- Happy path (JSON 200 with expected shape and non-empty summary).
- Empty input rejected with 400.
- Missing API key → 401.
- Malformed body → 400.
- Both shapes codex may send (minimal vs. full).
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiro.routes_openai import router, verify_api_key


def _mock_response(status_code: int, body: bytes):
    """Create a mock that looks like httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.aread = AsyncMock(return_value=body)
    resp.aclose = AsyncMock()
    return resp


def _patch_http_client(monkeypatch, mock_resp):
    """Patch kiro.routes_openai.KiroHttpClient so request_with_retry returns mock_resp."""
    instance = MagicMock()
    instance.request_with_retry = AsyncMock(return_value=mock_resp)
    instance.close = AsyncMock()
    instance.client = MagicMock()

    def _factory(*args, **kwargs):
        return instance

    monkeypatch.setattr("kiro.routes_openai.KiroHttpClient", _factory)
    return instance


def _patch_collect(monkeypatch, summary_text: str):
    """Patch collect_responses_response to return a canned Responses-shape dict."""
    output = (
        [
            {
                "type": "message",
                "id": "item_abc",
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
        ]
        if summary_text
        else []
    )
    canned = {
        "id": "resp_xyz",
        "object": "response",
        "created_at": 1700000000,
        "model": "claude-sonnet-4",
        "output": output,
        "status": "completed",
        "usage": {
            "input_tokens": 42,
            "output_tokens": 17,
            "total_tokens": 59,
        },
    }

    async def _fake_collect(*args, **kwargs):
        return canned

    monkeypatch.setattr(
        "kiro.routes_openai.collect_responses_response", _fake_collect
    )


@pytest.fixture
def app_with_mocked_state():
    """
    Build a FastAPI app wired to routes_openai.router with pre-populated
    app.state. Auth is bypassed by default; enable it in specific tests.
    """
    from kiro.auth import AuthType

    app = FastAPI()
    app.include_router(router)

    # Bypass API key auth for happy-path tests; specific tests can flip
    # this back by clearing the override.
    app.dependency_overrides[verify_api_key] = lambda: True

    app.state.auth_manager = MagicMock()
    app.state.auth_manager.auth_type = AuthType.KIRO_DESKTOP
    app.state.auth_manager.profile_arn = "arn:test"
    app.state.auth_manager.api_host = "https://fake-kiro"
    app.state.model_cache = MagicMock()
    app.state.http_client = MagicMock()
    app.state.token_stats = MagicMock()
    app.state.token_stats.record = MagicMock()

    return app


class TestCompactHappyPath:
    """Happy-path shapes codex sends."""

    def test_minimal_request_returns_compact_output(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: POST with a minimal CompactionInput-shaped payload
            returns {"output": [...]} with a non-empty assistant message.
        Purpose: Match codex CompactHistoryResponse so parsed.output is
            consumable.
        """
        summary = "Handoff summary: user asked to refactor auth.py; core done."
        _patch_http_client(monkeypatch, _mock_response(200, b""))
        _patch_collect(monkeypatch, summary)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "refactor auth.py please",
                        }
                    ],
                    "instructions": "",
                    "tools": [],
                    "parallel_tool_calls": False,
                },
            )

        assert r.status_code == 200, r.text
        data = r.json()
        assert "output" in data
        assert isinstance(data["output"], list)
        assert len(data["output"]) == 1
        item = data["output"][0]
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["status"] == "completed"
        assert item["content"][0]["type"] == "output_text"
        assert item["content"][0]["text"] == summary
        assert item["content"][0]["annotations"] == []

    def test_full_request_with_reasoning_and_text_accepted(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: POST with reasoning/text control fields (codex
            sends these on full compact requests) succeeds.
        Purpose: model_config=extra="allow" must tolerate codex's extras.
        """
        summary = "Full summary goes here."
        _patch_http_client(monkeypatch, _mock_response(200, b""))
        _patch_collect(monkeypatch, summary)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "hello",
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": "hi",
                        },
                    ],
                    "instructions": "Be helpful.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "weather tool",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                    "parallel_tool_calls": True,
                    "reasoning": {"effort": "high", "summary": "auto"},
                    "text": {"verbosity": "medium"},
                },
            )

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["output"][0]["content"][0]["text"] == summary

    def test_empty_summary_falls_back_to_placeholder(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: When the model returns no content, the response
            still carries a non-empty placeholder string.
        Purpose: codex should never receive empty output text that would
            break the compaction display.
        """
        _patch_http_client(monkeypatch, _mock_response(200, b""))
        _patch_collect(monkeypatch, "")  # model produced nothing

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "message", "role": "user", "content": "x"}
                    ],
                },
            )

        assert r.status_code == 200
        text = r.json()["output"][0]["content"][0]["text"]
        assert text  # non-empty
        assert "no summary" in text.lower() or "compaction" in text.lower()


class TestCompactValidationErrors:
    """Validation / malformed request handling."""

    def test_empty_input_list_rejected_with_400(self, app_with_mocked_state):
        """
        What it does: input=[] returns 400.
        Purpose: Reject payloads that carry nothing to summarize instead
            of silently sending an empty prompt to Kiro.
        """
        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={"model": "claude-sonnet-4", "input": []},
            )
        assert r.status_code == 400
        assert "non-empty" in r.json()["detail"].lower()

    def test_missing_input_rejected_with_400(self, app_with_mocked_state):
        """
        What it does: Payload without "input" returns 400.
        Purpose: input is required; don't let Pydantic's own error shape
            leak through unformatted.
        """
        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={"model": "claude-sonnet-4"},
            )
        assert r.status_code == 400

    def test_non_object_body_rejected_with_400(self, app_with_mocked_state):
        """
        What it does: Top-level JSON array (not an object) returns 400.
        Purpose: Guard against a client that sends a bare list.
        """
        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json=["not", "an", "object"],
            )
        assert r.status_code == 400

    def test_malformed_json_rejected_with_400(self, app_with_mocked_state):
        """
        What it does: Body that isn't valid JSON returns 400.
        Purpose: Don't 500 on garbage input.
        """
        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                content=b"{not valid json",
                headers={"content-type": "application/json"},
            )
        assert r.status_code == 400


class TestCompactAuth:
    """Auth enforcement."""

    def test_missing_api_key_returns_401(self):
        """
        What it does: No Authorization header → 401.
        Purpose: Same auth contract as every other /v1/* endpoint.
        """
        from kiro.routes_openai import router

        app = FastAPI()
        app.include_router(router)
        # Do NOT override verify_api_key — we want real auth here.

        with TestClient(app) as client:
            r = client.post(
                "/v1/responses/compact",
                json={"model": "claude-sonnet-4", "input": "hi"},
            )
        assert r.status_code == 401

    def test_wrong_api_key_returns_401(self):
        """
        What it does: Incorrect bearer token → 401.
        Purpose: Matches /v1/responses auth behavior.
        """
        from kiro.routes_openai import router

        app = FastAPI()
        app.include_router(router)

        with TestClient(app) as client:
            r = client.post(
                "/v1/responses/compact",
                headers={"Authorization": "Bearer nope"},
                json={"model": "claude-sonnet-4", "input": "hi"},
            )
        assert r.status_code == 401


class TestCompactUpstreamErrors:
    """Upstream Kiro failures surface as JSON errors (compact is unary)."""

    def test_upstream_429_returns_json_error(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: Kiro 429 → gateway returns 429 JSON with
            kiro_api_error type.
        Purpose: codex's CompactClient reads HTTP status; keep the
            contract unary.
        """
        body = json.dumps({"message": "Too many requests."}).encode()
        _patch_http_client(monkeypatch, _mock_response(429, body))

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "message", "role": "user", "content": "x"}
                    ],
                },
            )

        assert r.status_code == 429
        err = r.json()["error"]
        assert err["type"] == "kiro_api_error"
        assert err["code"] == 429

    def test_upstream_5xx_returns_json_error(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: Kiro 503 → gateway returns 503 JSON.
        Purpose: Same unary error contract; no SSE fallback.
        """
        _patch_http_client(monkeypatch, _mock_response(503, b"overloaded"))

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "message", "role": "user", "content": "x"}
                    ],
                },
            )
        assert r.status_code == 503


class TestCompactInputShapes:
    """Parametrized: minimal vs. rich input shapes codex may send."""

    @pytest.mark.parametrize(
        "input_payload",
        [
            # String input (easy form)
            "hello world",
            # Single message
            [{"type": "message", "role": "user", "content": "hi"}],
            # Multi-turn with function_call pair
            [
                {"type": "message", "role": "user", "content": "list files"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "list_files",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "a.py b.py",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Two files.",
                },
            ],
        ],
    )
    def test_compacts_across_input_shapes(
        self, app_with_mocked_state, monkeypatch, input_payload
    ):
        """
        What it does: Each input shape codex produces summarizes
            successfully.
        Purpose: Prevent regressions when codex evolves what it sends.
        """
        summary = "summary goes here"
        _patch_http_client(monkeypatch, _mock_response(200, b""))
        _patch_collect(monkeypatch, summary)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses/compact",
                json={"model": "claude-sonnet-4", "input": input_payload},
            )

        assert r.status_code == 200, r.text
        assert r.json()["output"][0]["content"][0]["text"] == summary
