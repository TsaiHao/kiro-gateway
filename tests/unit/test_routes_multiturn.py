# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for /v1/responses multi-turn resume via ``previous_response_id``.

Codex uses a delta protocol: on turn N>=2 it sends only the new input
items (typically ``function_call_output``) plus a
``previous_response_id`` pointing at the prior turn. The gateway keeps
an in-memory :class:`ResponseStore` and stitches the conversation back
together before calling Kiro.

These tests exercise the HTTP and WebSocket paths end-to-end at the
route level, with the upstream Kiro HTTP call mocked. They cover:

- Turn 1 stores its canonical state under the emitted response_id.
- Turn 2 with ``previous_response_id`` replays the stored state plus
  the new ``function_call_output`` items.
- Cache miss is non-fatal (logs warning, processes request as-is).
- Reasoning items stored on turn 1 are dropped when replayed on turn 2.
- WebSocket equivalent stores and resumes the same way.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiro.routes_openai import router, verify_api_key
from kiro.response_store import ResponseStore


# ------------------------------------------------------------------
# Helpers: minimal fake httpx.Response / KiroHttpClient patching
# ------------------------------------------------------------------


def _fake_upstream_response():
    """Stand-in for the httpx.Response returned by Kiro. The route only
    checks status_code and hands the object to collect/stream helpers,
    which we monkeypatch out in these tests."""
    resp = MagicMock()
    resp.status_code = 200
    resp.aread = AsyncMock(return_value=b"")
    resp.aclose = AsyncMock()
    return resp


def _patch_http_client(monkeypatch):
    """Patch KiroHttpClient so request_with_retry returns our fake
    200 response. Captures the kiro payload on each call for assertions."""
    captured = {"payloads": []}
    instance = MagicMock()

    async def _request_with_retry(method, url, payload, stream=False, **kwargs):
        captured["payloads"].append(payload)
        return _fake_upstream_response()

    instance.request_with_retry = AsyncMock(side_effect=_request_with_retry)
    instance.close = AsyncMock()
    instance.client = MagicMock()

    def _factory(*args, **kwargs):
        return instance

    monkeypatch.setattr("kiro.routes_openai.KiroHttpClient", _factory)
    return captured


def _patch_non_streaming_collect(monkeypatch, output_items, response_id):
    """Patch collect_responses_response to return a canned Responses dict."""
    canned = {
        "id": response_id,
        "object": "response",
        "created_at": 1700000000,
        "model": "claude-sonnet-4",
        "output": output_items,
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    async def _fake_collect(*args, **kwargs):
        return canned

    monkeypatch.setattr(
        "kiro.routes_openai.collect_responses_response", _fake_collect
    )


# ------------------------------------------------------------------
# App fixture with a real in-memory ResponseStore
# ------------------------------------------------------------------


@pytest.fixture
def app_with_store():
    """
    Build a minimal FastAPI app wired to the openai router with a REAL
    ResponseStore in app.state (not a mock — we want to observe real
    put/get behavior across turns).
    """
    from kiro.auth import AuthType

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_api_key] = lambda: True

    app.state.auth_manager = MagicMock()
    app.state.auth_manager.auth_type = AuthType.KIRO_DESKTOP
    app.state.auth_manager.profile_arn = "arn:test"
    app.state.auth_manager.api_host = "https://fake-kiro"
    app.state.model_cache = MagicMock()
    app.state.http_client = MagicMock()
    app.state.token_stats = MagicMock()
    app.state.token_stats.record = MagicMock()
    app.state.response_store = ResponseStore(max_entries=32, ttl_seconds=60)

    return app


# ------------------------------------------------------------------
# HTTP multi-turn tests
# ------------------------------------------------------------------


class TestHttpMultiTurnResume:
    """Non-streaming HTTP /v1/responses resume flow."""

    def test_turn1_stores_canonical_state(self, app_with_store, monkeypatch):
        """
        What it does: After a non-streaming turn 1 completes, the store
            holds the turn keyed by the response_id.
        Purpose: Establish the precondition for turn 2 resume.
        """
        captured = _patch_http_client(monkeypatch)
        _patch_non_streaming_collect(
            monkeypatch,
            output_items=[
                {
                    "type": "function_call",
                    "id": "item_1",
                    "call_id": "call_abc",
                    "name": "bash",
                    "arguments": '{"cmd":"ls /tmp"}',
                    "status": "completed",
                }
            ],
            response_id="resp_T1",
        )

        with TestClient(app_with_store) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "message", "role": "user",
                         "content": "list /tmp then /"}
                    ],
                    "stream": False,
                },
            )

        assert r.status_code == 200, r.text
        assert r.json()["id"] == "resp_T1"

        # Store should now have resp_T1.
        import asyncio
        stored = asyncio.run(app_with_store.state.response_store.get("resp_T1"))
        assert stored is not None
        assert stored.model == "claude-sonnet-4"
        assert len(stored.input_items) == 1
        assert stored.input_items[0]["role"] == "user"
        assert len(stored.output_items) == 1
        assert stored.output_items[0]["type"] == "function_call"

    def test_turn2_replays_prior_context_via_previous_response_id(
        self, app_with_store, monkeypatch
    ):
        """
        What it does: Pre-seed the store with turn 1, then POST turn 2
            with ``previous_response_id`` and only a function_call_output.
            Assert the Kiro payload sent upstream contains messages
            reconstructed from the stored state (original user question
            + prior function_call + new function_call_output).
        Purpose: This is the whole point of the store — the model must
            see turn 1's context, not just the delta.
        """
        # Seed the store with turn 1's canonical snapshot.
        from kiro.response_store import StoredTurn
        import asyncio

        seeded = StoredTurn(
            input_items=[
                {"type": "message", "role": "user",
                 "content": "list /tmp then /"},
            ],
            output_items=[
                {
                    "type": "function_call",
                    "id": "item_1",
                    "call_id": "call_abc",
                    "name": "bash",
                    "arguments": '{"cmd":"ls /tmp"}',
                    "status": "completed",
                }
            ],
            model="claude-sonnet-4",
            instructions="Be helpful.",
        )
        # Note: codex replays the prior function_call as part of the NEXT
        # turn's input_items (that's how OpenAI's real backend rebuilds it
        # server-side). We emulate that by making the seeded input contain
        # both the original message AND the function_call — i.e. the full
        # reconstructed conversation that led to resp_T1. The turn-2 delta
        # then only carries the function_call_output.
        seeded.input_items.append({
            "type": "function_call",
            "id": "item_1",
            "call_id": "call_abc",
            "name": "bash",
            "arguments": '{"cmd":"ls /tmp"}',
        })
        asyncio.run(app_with_store.state.response_store.put("resp_T1", seeded))

        captured = _patch_http_client(monkeypatch)
        _patch_non_streaming_collect(
            monkeypatch,
            output_items=[
                {
                    "type": "message",
                    "id": "item_2",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text",
                         "text": "both listings returned", "annotations": []}
                    ],
                }
            ],
            response_id="resp_T2",
        )

        with TestClient(app_with_store) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    # Turn 2 delta: ONLY the tool output.
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_abc",
                            "output": "file1.txt\nfile2.txt",
                        }
                    ],
                    "previous_response_id": "resp_T1",
                    "stream": False,
                },
            )

        assert r.status_code == 200, r.text
        assert len(captured["payloads"]) == 1
        payload = captured["payloads"][0]

        # Dig out the message history Kiro would see. The Kiro payload
        # structure is conversationState.currentMessage.userInputMessage
        # plus conversationState.history for prior turns.
        conv_state = payload["conversationState"]
        history = conv_state.get("history", [])
        current = conv_state.get("currentMessage", {})

        # Flatten all message content into text for a blunt substring check.
        def _collect_text(msg_block):
            bits = []
            if isinstance(msg_block, dict):
                for v in msg_block.values():
                    bits.extend(_collect_text(v))
            elif isinstance(msg_block, list):
                for v in msg_block:
                    bits.extend(_collect_text(v))
            elif isinstance(msg_block, str):
                bits.append(msg_block)
            return bits

        all_text = " ".join(_collect_text(history) + _collect_text(current))

        # Original user question must be present.
        assert "list /tmp then /" in all_text, (
            f"Turn-1 user question missing from Kiro payload. "
            f"History: {json.dumps(history, indent=2)[:500]}"
        )
        # Tool output from turn 2 must be present.
        assert "file1.txt" in all_text, (
            f"Turn-2 tool output missing from Kiro payload."
        )

    def test_cache_miss_processes_request_as_is(
        self, app_with_store, monkeypatch
    ):
        """
        What it does: previous_response_id points at an unknown id →
            request succeeds; warning is logged; Kiro payload only
            carries the delta (no KeyError, no 500).
        Purpose: Store miss is non-fatal. Codex may restart or talk to
            a different backend; we must not error out.
        """
        captured = _patch_http_client(monkeypatch)
        _patch_non_streaming_collect(
            monkeypatch,
            output_items=[
                {
                    "type": "message",
                    "id": "item_x",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "ok", "annotations": []}
                    ],
                }
            ],
            response_id="resp_T_miss",
        )

        with TestClient(app_with_store) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "message", "role": "user", "content": "hi"}
                    ],
                    "previous_response_id": "resp_doesnotexist",
                    "stream": False,
                },
            )

        assert r.status_code == 200, r.text
        # Payload was still sent upstream exactly once.
        assert len(captured["payloads"]) == 1

    def test_reasoning_items_dropped_on_replay(
        self, app_with_store, monkeypatch
    ):
        """
        What it does: Seed the store with a turn whose input_items
            include a reasoning item. Resume on the next turn and
            assert the reconstructed Kiro payload does NOT carry the
            reasoning item.
        Purpose: Reasoning is server-side state; replaying it confuses
            the model (see comment in convert_responses_input_to_unified).
        """
        from kiro.response_store import StoredTurn
        import asyncio

        seeded = StoredTurn(
            input_items=[
                {"type": "message", "role": "user", "content": "compute 1+1"},
                # This should be stripped before replay.
                {"type": "reasoning", "id": "r1",
                 "content": [{"type": "reasoning_text",
                              "text": "INTERNAL PRIOR THOUGHT"}]},
            ],
            output_items=[
                {"type": "message", "id": "item_a",
                 "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "2",
                              "annotations": []}]}
            ],
            model="claude-sonnet-4",
        )
        asyncio.run(app_with_store.state.response_store.put("resp_R", seeded))

        captured = _patch_http_client(monkeypatch)
        _patch_non_streaming_collect(
            monkeypatch,
            output_items=[
                {"type": "message", "id": "item_b",
                 "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "ok",
                              "annotations": []}]}
            ],
            response_id="resp_R2",
        )

        with TestClient(app_with_store) as client:
            r = client.post(
                "/v1/responses",
                json={
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "message", "role": "user", "content": "again"}
                    ],
                    "previous_response_id": "resp_R",
                    "stream": False,
                },
            )

        assert r.status_code == 200, r.text
        assert len(captured["payloads"]) == 1
        payload_str = json.dumps(captured["payloads"][0])
        assert "INTERNAL PRIOR THOUGHT" not in payload_str, (
            "Reasoning content leaked into Kiro payload on replay"
        )
        # Original user message should still be there.
        assert "compute 1+1" in payload_str


# ------------------------------------------------------------------
# WebSocket multi-turn test
# ------------------------------------------------------------------


class TestWsMultiTurnResume:
    """
    The WS path uses the same _resume_from_prev_response/_persist_turn
    helpers as HTTP. We exercise at least one full resume flow via
    TestClient's WebSocket mode.
    """

    def test_ws_resume_and_persist(self, app_with_store, monkeypatch):
        """
        What it does: Connect WS, send turn 1 (stored), send turn 2 with
            previous_response_id, assert turn 2's Kiro payload contains
            the turn-1 user message.
        Purpose: Verify the WS handler runs resume + persist.
        """
        # Patch KiroHttpClient to capture the outgoing payloads.
        captured = _patch_http_client(monkeypatch)

        # Patch stream_kiro_to_responses_ws to:
        # 1. Emit a canned response.completed with a known response_id.
        # 2. Invoke the on_complete callback so _persist_turn runs.
        async def _fake_ws_stream(client, response, model, model_cache,
                                  auth_manager, *,
                                  first_token_timeout=None,
                                  request_messages=None, request_tools=None,
                                  usage_collector=None, on_complete=None):
            # Canned output for this "turn".
            resp_id = _fake_ws_stream.response_id
            output_items = _fake_ws_stream.output_items
            # Persist before emitting completed.
            if on_complete is not None:
                await on_complete(resp_id, output_items)
            payload = {
                "id": resp_id,
                "object": "response",
                "created_at": 1700000000,
                "model": model,
                "output": output_items,
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "total_tokens": 2},
            }
            yield json.dumps({"type": "response.completed",
                              "response": payload})

        _fake_ws_stream.response_id = "resp_WS1"
        _fake_ws_stream.output_items = []

        monkeypatch.setattr(
            "kiro.routes_openai.stream_kiro_to_responses_ws",
            _fake_ws_stream,
        )

        # Override verify_api_key is for HTTP; WS reads headers directly.
        # Set PROXY_API_KEY env at runtime via the router's module var —
        # simpler: monkeypatch the expected header check at module level.
        from kiro.config import PROXY_API_KEY as real_key
        headers = {"Authorization": f"Bearer {real_key}"}

        with TestClient(app_with_store) as client:
            with client.websocket_connect(
                "/v1/responses", headers=headers
            ) as ws:
                # --- Turn 1: user question ---
                _fake_ws_stream.response_id = "resp_WS1"
                _fake_ws_stream.output_items = [
                    {
                        "type": "function_call",
                        "id": "item_1",
                        "call_id": "call_z",
                        "name": "bash",
                        "arguments": '{"cmd":"ls /tmp"}',
                        "status": "completed",
                    }
                ]
                ws.send_text(json.dumps({
                    "type": "response.create",
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "message", "role": "user",
                         "content": "list /tmp please"}
                    ],
                }))
                msg = ws.receive_text()
                envelope = json.loads(msg)
                assert envelope["type"] == "response.completed"
                assert envelope["response"]["id"] == "resp_WS1"

                # --- Turn 2: resume ---
                _fake_ws_stream.response_id = "resp_WS2"
                _fake_ws_stream.output_items = [
                    {"type": "message", "id": "item_2", "role": "assistant",
                     "status": "completed",
                     "content": [{"type": "output_text",
                                  "text": "done", "annotations": []}]}
                ]
                ws.send_text(json.dumps({
                    "type": "response.create",
                    "model": "claude-sonnet-4",
                    "input": [
                        {"type": "function_call_output",
                         "call_id": "call_z",
                         "output": "aaa\nbbb"}
                    ],
                    "previous_response_id": "resp_WS1",
                }))
                msg2 = ws.receive_text()
                envelope2 = json.loads(msg2)
                assert envelope2["type"] == "response.completed"
                assert envelope2["response"]["id"] == "resp_WS2"

        # Turn 2 must have produced an upstream payload carrying the
        # turn-1 user question.
        assert len(captured["payloads"]) == 2
        turn2_payload = captured["payloads"][1]
        payload_str = json.dumps(turn2_payload)
        assert "list /tmp please" in payload_str, (
            "Turn-1 user question missing from turn-2 Kiro payload "
            "(WS resume did not stitch)."
        )
        assert "aaa" in payload_str, "Turn-2 tool output missing"
