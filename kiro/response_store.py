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
In-memory LRU+TTL store for OpenAI Responses API conversation state.

Codex's Responses API client uses a delta protocol: on turn N>=2 it only
sends NEW input items plus a ``previous_response_id``. The real OpenAI
backend stitches the full conversation from that id. This gateway is
otherwise stateless, so without a store the second turn would reach Kiro
with only the tool results and no original question.

The store maps ``response_id -> StoredTurn`` holding the canonical
``input_items`` (everything codex would have sent non-delta) plus the
``output_items`` this response produced. On resume, the handler merges
``stored.input_items + request.input`` and runs the normal Kiro payload
build against the merged sequence.

Thread-safety: a single ``asyncio.Lock`` guards all mutations. The store
is designed for in-process use only. Entries expire via simple TTL on
``get`` (lazy eviction) and LRU on capacity overflow.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import asyncio
from loguru import logger


# Input item ``type`` values we filter out when replaying a stored turn.
# Reasoning is a server-side artifact — codex doesn't replay its own prior
# reasoning, and replaying it as user-visible input has been observed to
# make the model forget the original question.
_REASONING_TYPES = frozenset(
    {"reasoning", "reasoning_text", "reasoning_summary_text"}
)


@dataclass
class StoredTurn:
    """
    Canonical snapshot of a single Responses API turn.

    Attributes:
        input_items: Full list of input items that led to THIS response.
            This is what codex would have sent if it were not using the
            delta protocol (prior turn's input + new items).
        output_items: Output items produced by this response (reasoning,
            message, function_call). Stored as raw dicts so they can be
            replayed verbatim.
        model: Model name used for the turn. Kept so resume can notice
            model-swap attempts.
        created_at: Wall-clock timestamp at store time (seconds since
            epoch), used for TTL.
        instructions: Optional system prompt for this chain. Inherited by
            later turns if they don't provide their own.
    """

    input_items: List[Dict[str, Any]]
    output_items: List[Dict[str, Any]]
    model: str
    created_at: float = field(default_factory=time.time)
    instructions: Optional[str] = None


def _sanitize_stored_input_for_replay(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Drop reasoning-type items from a stored input list before replay.

    Rationale: reasoning is server-side state. The model does not need
    its own prior reasoning replayed as user-visible input, and replaying
    it has been observed to confuse the model across tool rounds.

    Args:
        items: Raw input items as stored in :class:`StoredTurn`.

    Returns:
        New list omitting reasoning-type items, preserving order of the
        remaining items.
    """
    filtered: List[Dict[str, Any]] = []
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        if item.get("type") in _REASONING_TYPES:
            dropped += 1
            continue
        filtered.append(item)
    if dropped:
        logger.debug(
            f"ResponseStore: dropped {dropped} reasoning item(s) from replay input"
        )
    return filtered


class ResponseStore:
    """
    In-memory LRU + TTL store keyed by ``response_id``.

    Uses :class:`collections.OrderedDict` for LRU ordering. ``get`` moves
    the accessed entry to the end (most-recently-used). ``put`` evicts
    the least-recently-used entry when size exceeds ``max_entries``.

    TTL expiry is lazy: entries older than ``ttl_seconds`` are removed
    on ``get``. An optional ``clear_expired`` helper exists for tests or
    a future background sweep.

    All public methods take :class:`asyncio.Lock`. They are safe to call
    concurrently from request handlers in a single-process async app.
    """

    def __init__(
        self,
        max_entries: int = 512,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """
        Initialize the store.

        Args:
            max_entries: Maximum number of turns to retain. Oldest turn
                (by access order) is evicted on overflow.
            ttl_seconds: Seconds before an entry is considered expired
                on ``get``. Pass 0 or negative to disable expiry.
        """
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, StoredTurn]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, response_id: str) -> Optional[StoredTurn]:
        """
        Fetch a stored turn by response id.

        On hit, moves the entry to the MRU end. On TTL expiry, removes
        the entry and returns None.

        Args:
            response_id: Id previously returned by a successful response.

        Returns:
            The :class:`StoredTurn`, or None on miss/expired.
        """
        async with self._lock:
            turn = self._entries.get(response_id)
            if turn is None:
                return None

            if self._ttl_seconds > 0:
                age = time.time() - turn.created_at
                if age > self._ttl_seconds:
                    # Expired — evict and report miss.
                    self._entries.pop(response_id, None)
                    logger.debug(
                        f"ResponseStore: expired entry {response_id} "
                        f"(age={age:.1f}s > ttl={self._ttl_seconds}s)"
                    )
                    return None

            # Move to MRU end.
            self._entries.move_to_end(response_id)
            return turn

    async def put(self, response_id: str, turn: StoredTurn) -> None:
        """
        Insert or replace a stored turn.

        Evicts the LRU entry if the size would exceed ``max_entries``.

        Args:
            response_id: Id of the response this turn produced.
            turn: Canonical conversation snapshot.
        """
        async with self._lock:
            if response_id in self._entries:
                # Overwrite semantics: move to MRU end.
                self._entries.move_to_end(response_id)
            self._entries[response_id] = turn

            # Evict LRU entries over capacity.
            while len(self._entries) > self._max_entries:
                oldest_key, _ = self._entries.popitem(last=False)
                logger.debug(
                    f"ResponseStore: evicted LRU entry {oldest_key} "
                    f"(capacity={self._max_entries})"
                )

    async def clear_expired(self) -> int:
        """
        Sweep and remove all expired entries.

        Returns:
            Number of entries removed. Always 0 when TTL is disabled.
        """
        if self._ttl_seconds <= 0:
            return 0
        now = time.time()
        removed = 0
        async with self._lock:
            expired_keys = [
                key
                for key, turn in self._entries.items()
                if now - turn.created_at > self._ttl_seconds
            ]
            for key in expired_keys:
                self._entries.pop(key, None)
                removed += 1
        if removed:
            logger.debug(f"ResponseStore: swept {removed} expired entries")
        return removed

    async def size(self) -> int:
        """Return the current number of stored entries."""
        async with self._lock:
            return len(self._entries)
