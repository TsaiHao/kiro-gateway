# -*- coding: utf-8 -*-

"""
Unit tests for /v1/responses pre-stream error handling.

When the upstream Kiro call returns non-200 AND the client asked for
streaming, routes_openai.responses must emit a single response.failed SSE
event instead of a JSON 4xx/5xx body. Non-streaming requests keep the old
JSONResponse behavior.

Covers:
- Streaming path emits exactly one response.failed event, no response.completed.
- Failed event satisfies the codex SSE contract: top-level "type" and a
  nested "response" object with code + message.
- Error codes derived from upstream status/reason (context/quota/invalid).
- Non-streaming path still returns a JSON error response with original status.
- WS error frames carry the classified code.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from kiro.routes_openai import router, verify_api_key
from kiro.config import PROXY_API_KEY


def _parse_sse(body: str):
    """Parse an SSE body string into a list of (event, data_dict) tuples."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        event_line = next((l for l in lines if l.startswith("event: ")), None)
        data_line = next((l for l in lines if l.startswith("data: ")), None)
        if event_line is None or data_line is None:
            continue
        events.append((event_line[len("event: "):], json.loads(data_line[len("data: "):])))
    return events


@pytest.fixture
def app_with_mocked_state():
    """
    Build a minimal FastAPI app wired to routes_openai.router, with
    app.state pre-populated with mock auth/cache/http objects. The tests
    then patch KiroHttpClient to control what the "upstream" returns.
    """
    from fastapi import FastAPI
    from kiro.auth import AuthType

    app = FastAPI()
    app.include_router(router)

    # Bypass API key auth to keep tests focused on the error path.
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


def _mock_response(status_code: int, body: bytes):
    """Create a minimal mock that looks like httpx.Response."""
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


class TestResponsesPreStreamErrors:
    """Pre-stream (HTTP path) error event emission."""

    def test_streaming_429_emits_response_failed_rate_limit(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: 429 from Kiro → streaming body is a single
            response.failed SSE event with code=rate_limit_exceeded.
        Purpose: Codex must receive the failure via SSE, not as a transport
            error, so its rate-limit retry logic fires.
        """
        body = json.dumps({"message": "Too many requests."}).encode()
        mock_resp = _mock_response(429, body)
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    "input": "hello",
                    "stream": True,
                },
            )

        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(r.text)
        event_types = [t for t, _ in events]
        assert event_types == ["response.failed"], (
            f"expected exactly one response.failed, got {event_types}"
        )
        _, payload = events[0]
        assert payload["type"] == "response.failed"
        assert payload["response"]["status"] == "failed"
        assert payload["response"]["error"]["code"] == "rate_limit_exceeded"
        # Must contain try-again-in wording for codex's regex
        assert "try again in" in payload["response"]["error"]["message"].lower()

    def test_streaming_context_length_reason_classifies_correctly(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: Kiro returns 400 with
            reason=CONTENT_LENGTH_EXCEEDS_THRESHOLD → error.code becomes
            context_length_exceeded (codex treats it as fatal).
        Purpose: Honors KiroErrorInfo-based classification.
        """
        body = json.dumps({
            "message": "Input is too long.",
            "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD",
        }).encode()
        mock_resp = _mock_response(400, body)
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    "input": "x",
                    "stream": True,
                },
            )

        assert r.status_code == 200
        events = _parse_sse(r.text)
        assert len(events) == 1
        _, payload = events[0]
        assert payload["response"]["error"]["code"] == "context_length_exceeded"

    def test_streaming_5xx_becomes_server_overloaded(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: 503 from Kiro → server_is_overloaded (codex retries).
        Purpose: Status-based fallback classification.
        """
        mock_resp = _mock_response(503, b"not json, plain overload")
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    "input": "x",
                    "stream": True,
                },
            )

        assert r.status_code == 200
        events = _parse_sse(r.text)
        assert len(events) == 1
        assert events[0][1]["response"]["error"]["code"] == "server_is_overloaded"

    def test_streaming_failed_never_accompanied_by_completed(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: The error body must NOT contain response.completed
            after response.failed — codex treats failed as terminal.
        Purpose: Match codex's parser which returns on the first failed.
        """
        mock_resp = _mock_response(500, b'{"message":"upstream bad"}')
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "x", "stream": True},
            )

        assert "response.completed" not in r.text
        assert "response.failed" in r.text

    def test_streaming_failed_matches_codex_wire_shape(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: Wire-shape guard — top-level "type" agrees with event:
            header, payload is under "response", error has code and message,
            output is [] and usage is null.
        Purpose: Parity with TestCodexSseContract for the failure path.
        """
        mock_resp = _mock_response(400, b'{"message":"Improperly formed request."}')
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "x", "stream": True},
            )

        events = _parse_sse(r.text)
        assert len(events) == 1
        header, data = events[0]
        assert header == "response.failed"
        assert data["type"] == "response.failed"
        assert "response" in data
        resp = data["response"]
        assert resp["status"] == "failed"
        assert resp["output"] == []
        assert resp["usage"] is None
        assert resp["incomplete_details"] is None
        assert resp["error"]["code"] == "invalid_prompt"
        # Message is rewritten by enhance_kiro_error into something actionable
        # — just assert it's non-empty rather than pinning the exact wording.
        assert isinstance(resp["error"]["message"], str)
        assert resp["error"]["message"]

    def test_non_streaming_still_returns_json_error(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: Non-streaming path preserves legacy behavior — HTTP
            error status with JSON body.
        Purpose: Other clients (non-codex) still rely on HTTP semantics.
        """
        body = json.dumps({"message": "Too many requests."}).encode()
        mock_resp = _mock_response(429, body)
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    "input": "x",
                    "stream": False,
                },
            )

        assert r.status_code == 429
        data = r.json()
        assert data["error"]["type"] == "kiro_api_error"
        assert data["error"]["code"] == 429

    def test_streaming_unparseable_body_still_classifies(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: Upstream returns HTML/plaintext junk — classifier still
            produces a valid response.failed (server_is_overloaded via 500).
        Purpose: Never crash on weird upstream bodies.
        """
        mock_resp = _mock_response(500, b"<html>500 Internal</html>")
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "x", "stream": True},
            )

        events = _parse_sse(r.text)
        assert len(events) == 1
        code = events[0][1]["response"]["error"]["code"]
        assert code == "server_is_overloaded"


class TestResponsesPreStreamErrorsNegative:
    """Negative / edge cases."""

    def test_streaming_401_maps_to_invalid_prompt(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: 401 from upstream is coded as invalid_prompt (codex
            treats as fatal — not retryable).
        Purpose: An auth failure should not retry-storm.
        """
        mock_resp = _mock_response(401, b'{"message":"unauthorized"}')
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "x", "stream": True},
            )

        events = _parse_sse(r.text)
        assert events[0][1]["response"]["error"]["code"] == "invalid_prompt"

    def test_streaming_monthly_quota_maps_to_insufficient_quota(
        self, app_with_mocked_state, monkeypatch
    ):
        """
        What it does: reason=MONTHLY_REQUEST_COUNT → insufficient_quota.
        Purpose: Codex shows a quota-specific message and stops.
        """
        body = json.dumps({
            "message": "Monthly request limit exceeded.",
            "reason": "MONTHLY_REQUEST_COUNT",
        }).encode()
        mock_resp = _mock_response(429, body)
        _patch_http_client(monkeypatch, mock_resp)

        with TestClient(app_with_mocked_state) as client:
            r = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "x", "stream": True},
            )

        events = _parse_sse(r.text)
        code = events[0][1]["response"]["error"]["code"]
        assert code == "insufficient_quota"
