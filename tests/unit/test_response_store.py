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
Unit tests for the :class:`ResponseStore`.

The store is the backbone of codex multi-turn resume — every
``previous_response_id`` lookup flows through here. These tests exercise
the contract the route handler relies on:

- put/get roundtrip preserves canonical state
- LRU eviction kicks in exactly at capacity overflow
- TTL-expired entries are purged on access and report miss
- Concurrent put/get under asyncio.gather does not lose updates
- Missing id returns None
- Reasoning-type items are dropped by the replay sanitizer
"""

import asyncio
import time

import pytest

from kiro.response_store import (
    ResponseStore,
    StoredTurn,
    _sanitize_stored_input_for_replay,
)


def _turn(input_items=None, output_items=None, model="claude-sonnet-4",
          created_at=None, instructions=None):
    """Build a StoredTurn with sensible defaults."""
    return StoredTurn(
        input_items=input_items if input_items is not None else [
            {"type": "message", "role": "user", "content": "hi"}
        ],
        output_items=output_items if output_items is not None else [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "hello"}
            ]}
        ],
        model=model,
        created_at=created_at if created_at is not None else time.time(),
        instructions=instructions,
    )


class TestResponseStoreBasics:
    """Happy-path put/get behaviour."""

    @pytest.mark.asyncio
    async def test_put_then_get_roundtrip(self):
        """
        What it does: Store a turn and retrieve it by id.
        Purpose: The canonical path the route handler depends on.
        """
        store = ResponseStore(max_entries=4, ttl_seconds=60)
        turn = _turn()
        await store.put("resp_A", turn)
        got = await store.get("resp_A")
        assert got is turn
        assert got.model == "claude-sonnet-4"
        assert got.input_items == turn.input_items
        assert got.output_items == turn.output_items

    @pytest.mark.asyncio
    async def test_get_missing_id_returns_none(self):
        """
        What it does: Looking up an unknown id returns None.
        Purpose: Cache miss must be non-fatal at the caller.
        """
        store = ResponseStore()
        assert await store.get("resp_nope") is None

    @pytest.mark.asyncio
    async def test_put_overwrites_existing_key(self):
        """
        What it does: Putting the same id twice replaces the entry.
        Purpose: A response id should only ever map to one snapshot.
        """
        store = ResponseStore()
        first = _turn(model="claude-haiku-4.5")
        second = _turn(model="claude-opus-4.5")
        await store.put("resp_X", first)
        await store.put("resp_X", second)
        got = await store.get("resp_X")
        assert got is second
        assert await store.size() == 1

    @pytest.mark.asyncio
    async def test_size_tracks_insertions(self):
        """
        What it does: size() reflects the number of stored entries.
        Purpose: Verifies LRU container stays consistent.
        """
        store = ResponseStore(max_entries=10)
        assert await store.size() == 0
        await store.put("a", _turn())
        await store.put("b", _turn())
        assert await store.size() == 2


class TestResponseStoreLRU:
    """LRU eviction behaviour at capacity overflow."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_over_capacity(self):
        """
        What it does: With max_entries=2, adding a third entry evicts
            the least-recently-used one.
        Purpose: Core LRU contract.
        """
        store = ResponseStore(max_entries=2, ttl_seconds=60)
        await store.put("a", _turn())
        await store.put("b", _turn())
        await store.put("c", _turn())
        assert await store.get("a") is None  # evicted
        assert await store.get("b") is not None
        assert await store.get("c") is not None

    @pytest.mark.asyncio
    async def test_get_updates_recency(self):
        """
        What it does: Accessing an entry moves it to the MRU end, so a
            later insertion evicts the OTHER old entry.
        Purpose: move_to_end on read is what distinguishes LRU from FIFO.
        """
        store = ResponseStore(max_entries=2)
        await store.put("a", _turn())
        await store.put("b", _turn())
        # Touch a so b becomes LRU.
        assert await store.get("a") is not None
        await store.put("c", _turn())
        assert await store.get("a") is not None
        assert await store.get("b") is None  # b was LRU, evicted
        assert await store.get("c") is not None

    @pytest.mark.asyncio
    async def test_put_same_key_moves_to_mru(self):
        """
        What it does: Re-putting an existing key should NOT evict the
            other entry; it refreshes recency instead.
        Purpose: Overwriting must not corrupt LRU bookkeeping.
        """
        store = ResponseStore(max_entries=2)
        await store.put("a", _turn())
        await store.put("b", _turn())
        # Refresh a.
        await store.put("a", _turn(model="claude-opus-4.5"))
        await store.put("c", _turn())
        assert await store.get("b") is None  # b was LRU
        got_a = await store.get("a")
        assert got_a is not None and got_a.model == "claude-opus-4.5"
        assert await store.get("c") is not None

    def test_rejects_non_positive_capacity(self):
        """
        What it does: Constructor raises on max_entries < 1.
        Purpose: A zero-capacity store is nonsense and would mask bugs.
        """
        with pytest.raises(ValueError):
            ResponseStore(max_entries=0)


class TestResponseStoreTTL:
    """TTL expiry behaviour."""

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none_and_is_evicted(self):
        """
        What it does: An entry older than ttl_seconds returns None on
            get AND is removed from the store.
        Purpose: TTL must actually free memory, not just hide entries.
        """
        store = ResponseStore(max_entries=4, ttl_seconds=0.05)
        await store.put("a", _turn(created_at=time.time() - 10))
        assert await store.get("a") is None
        assert await store.size() == 0

    @pytest.mark.asyncio
    async def test_fresh_entry_survives(self):
        """
        What it does: An entry created just now is returned as-is.
        Purpose: Sanity: TTL gate isn't over-aggressive.
        """
        store = ResponseStore(ttl_seconds=60)
        await store.put("a", _turn())
        assert await store.get("a") is not None

    @pytest.mark.asyncio
    async def test_ttl_zero_disables_expiry(self):
        """
        What it does: ttl_seconds=0 keeps entries forever.
        Purpose: Allow operators to opt out of TTL.
        """
        store = ResponseStore(ttl_seconds=0)
        await store.put("a", _turn(created_at=time.time() - 99999))
        assert await store.get("a") is not None

    @pytest.mark.asyncio
    async def test_clear_expired_sweeps_old_entries(self):
        """
        What it does: clear_expired removes entries past their TTL.
        Purpose: Support a background sweep helper.
        """
        store = ResponseStore(ttl_seconds=1.0)
        await store.put("old", _turn(created_at=time.time() - 100))
        await store.put("fresh", _turn())
        removed = await store.clear_expired()
        assert removed == 1
        assert await store.get("old") is None
        assert await store.get("fresh") is not None


class TestResponseStoreConcurrency:
    """Concurrent access under asyncio.gather — no lost updates."""

    @pytest.mark.asyncio
    async def test_concurrent_puts_and_gets_no_lost_updates(self):
        """
        What it does: Fire many puts and gets concurrently against the
            same store; every successfully-put key must be retrievable.
        Purpose: Lock correctness under async contention.
        """
        store = ResponseStore(max_entries=200)

        async def do_put(i: int):
            await store.put(f"k{i}", _turn(model=f"m{i}"))

        async def do_get(i: int):
            # Racing get may land before put; that's fine.
            return await store.get(f"k{i}")

        # Interleave puts and gets.
        ops = []
        for i in range(100):
            ops.append(do_put(i))
            ops.append(do_get(i))
        await asyncio.gather(*ops)

        # Every put must have landed (capacity >> total inserts).
        assert await store.size() == 100
        for i in range(100):
            t = await store.get(f"k{i}")
            assert t is not None, f"lost entry k{i}"
            assert t.model == f"m{i}"

    @pytest.mark.asyncio
    async def test_concurrent_put_same_key_last_writer_wins(self):
        """
        What it does: Many concurrent puts to the same key — store ends
            up with exactly one entry and a value from one of the puts.
        Purpose: No partial-state artifacts under contention.
        """
        store = ResponseStore()

        async def do_put(i: int):
            await store.put("shared", _turn(model=f"m{i}"))

        await asyncio.gather(*(do_put(i) for i in range(50)))
        assert await store.size() == 1
        t = await store.get("shared")
        assert t is not None
        assert t.model.startswith("m")


class TestSanitizeStoredInputForReplay:
    """Reasoning items must be dropped before replay."""

    def test_drops_reasoning(self):
        """
        What it does: Reasoning, reasoning_text, reasoning_summary_text
            are all filtered out.
        Purpose: Match the comment in convert_responses_input_to_unified:
            the model doesn't need its own prior reasoning replayed.
        """
        items = [
            {"type": "message", "role": "user", "content": "q"},
            {"type": "reasoning", "id": "r1", "content": []},
            {"type": "reasoning_text", "id": "r2"},
            {"type": "reasoning_summary_text", "id": "r3"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        ]
        out = _sanitize_stored_input_for_replay(items)
        types = [i.get("type") for i in out]
        assert "reasoning" not in types
        assert "reasoning_text" not in types
        assert "reasoning_summary_text" not in types
        assert "message" in types
        assert "function_call" in types
        assert len(out) == 2

    def test_preserves_order_of_kept_items(self):
        """
        What it does: Non-reasoning items keep their relative order.
        Purpose: Order matters for function_call/function_call_output pairing.
        """
        items = [
            {"type": "message", "role": "user", "content": "1"},
            {"type": "reasoning", "id": "r"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]
        out = _sanitize_stored_input_for_replay(items)
        assert [i["type"] for i in out] == [
            "message", "function_call", "function_call_output"
        ]

    def test_tolerates_non_dict_items(self):
        """
        What it does: Unknown shapes (e.g. raw strings) are kept as-is.
        Purpose: The sanitizer must never raise on odd input.
        """
        items = ["stray-string", {"type": "reasoning"}, {"type": "message"}]
        out = _sanitize_stored_input_for_replay(items)
        # The stray string is preserved; reasoning is dropped.
        assert "stray-string" in out
        assert {"type": "message"} in out
        assert {"type": "reasoning"} not in out
