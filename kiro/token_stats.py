# -*- coding: utf-8 -*-

"""
Token usage statistics with SQLite storage.

Tracks token consumption by model with hourly granularity.
Supports querying by day, hour, and model with automatic 2-week retention.
"""

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Default retention: 14 days
RETENTION_DAYS = 14

# Default DB path
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "kiro-gateway" / "token_stats.db"


class UsageCollector:
    """Lightweight container for capturing usage from streaming generators."""

    __slots__ = ("model", "input_tokens", "output_tokens")

    def __init__(self):
        self.model: str = ""
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def set(self, model: str, input_tokens: int, output_tokens: int):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class TokenStats:
    """SQLite-backed token usage statistics."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=5.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info(f"Token stats DB initialized: {self._db_path}")

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                model TEXT NOT NULL,
                hour TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                request_count INTEGER DEFAULT 0,
                PRIMARY KEY (model, hour)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_usage_hour ON token_usage(hour)"
        )
        self._conn.commit()

    def record(self, model: str, input_tokens: int, output_tokens: int):
        """Record token usage for the current hour bucket."""
        if not model or (input_tokens <= 0 and output_tokens <= 0):
            return
        hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO token_usage (model, hour, input_tokens, output_tokens, request_count)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(model, hour) DO UPDATE SET
                    input_tokens = input_tokens + excluded.input_tokens,
                    output_tokens = output_tokens + excluded.output_tokens,
                    request_count = request_count + 1
                """,
                (model, hour, input_tokens, output_tokens),
            )
            self._conn.commit()

    def get_summary(self, model: Optional[str] = None) -> List[dict]:
        """Get total usage per model (within retention window)."""
        cutoff = self._cutoff_hour()
        if model:
            rows = self._conn.execute(
                """
                SELECT model,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       SUM(request_count) as request_count
                FROM token_usage
                WHERE model = ? AND hour >= ?
                GROUP BY model
                """,
                (model, cutoff),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT model,
                       SUM(input_tokens) as input_tokens,
                       SUM(output_tokens) as output_tokens,
                       SUM(request_count) as request_count
                FROM token_usage
                WHERE hour >= ?
                GROUP BY model
                ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
                """,
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_hourly(self, date_str: Optional[str] = None, model: Optional[str] = None) -> List[dict]:
        """Get hourly breakdown for a specific day."""
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        hour_prefix = date_str + "T"
        params: list = [hour_prefix + "%"]
        query = """
            SELECT model, hour, input_tokens, output_tokens, request_count
            FROM token_usage
            WHERE hour LIKE ?
        """
        if model:
            query += " AND model = ?"
            params.append(model)
        query += " ORDER BY hour, model"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_daily(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[dict]:
        """Get daily summaries, optionally filtered by date range and model."""
        cutoff = self._cutoff_hour()
        params: list = [cutoff]
        query = """
            SELECT model,
                   SUBSTR(hour, 1, 10) as date,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(request_count) as request_count
            FROM token_usage
            WHERE hour >= ?
        """
        if start_date:
            query += " AND hour >= ?"
            params.append(start_date + "T00")
        if end_date:
            query += " AND hour <= ?"
            params.append(end_date + "T23")
        if model:
            query += " AND model = ?"
            params.append(model)
        query += " GROUP BY model, SUBSTR(hour, 1, 10) ORDER BY date, model"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def cleanup(self):
        """Delete rows older than retention period."""
        cutoff = self._cutoff_hour()
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM token_usage WHERE hour < ?", (cutoff,)
            )
            self._conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Token stats cleanup: deleted {cursor.rowcount} old rows")

    def close(self):
        self._conn.close()

    def _cutoff_hour(self) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        return cutoff.strftime("%Y-%m-%dT%H")
