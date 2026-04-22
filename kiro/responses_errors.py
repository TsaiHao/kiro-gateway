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
Error classification and `response.failed` payload construction for the
OpenAI Responses API endpoint.

This module lives alongside `kiro_errors.py` (Kiro's native error enhancer)
but is narrow to the Responses surface: it translates either an
`EnhancedKiroError` (parsed Kiro upstream error) or a raw exception into the
subset of `error.code` values that codex's SSE parser recognizes as actionable.

Why a dedicated module (per CLAUDE.md "Systems Over Patches"):
- The mapping between Kiro errors / HTTP statuses and codex's `error.code`
  vocabulary is load-bearing; it drives codex's retry vs. fatal decision. A
  one-off if/else sprinkled across routes_openai.py would accumulate copies.
- We reuse this from three call sites (HTTP pre-stream, WS pre-stream,
  mid-stream generator failure) and all of them must agree.

Codex contract (see codex-rs/codex-api/src/sse/responses.rs:274-316):
    code == "context_length_exceeded"  -> ApiError::ContextWindowExceeded  (fatal)
    code == "insufficient_quota"       -> ApiError::QuotaExceeded          (fatal)
    code == "usage_not_included"       -> ApiError::UsageNotIncluded       (fatal)
    code == "invalid_prompt"           -> ApiError::InvalidRequest         (fatal)
    code == "server_is_overloaded"
        or code == "slow_down"         -> ApiError::ServerOverloaded       (retryable)
    code == "rate_limit_exceeded"      -> ApiError::Retryable with delay parsed
                                          from the message via
                                          "try again in <N>(s|ms|seconds?)"
    anything else                      -> ApiError::Retryable (no delay)

The rate-limit regex is strict (`try again in\\s*(\\d+(?:\\.\\d+)?)\\s*(s|ms|seconds?)`);
this module preserves that wording verbatim when we synthesize a message.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from kiro.kiro_errors import KiroErrorInfo


# Codex-recognized error codes. Anything else is treated as a generic
# retryable error by codex.
CODE_CONTEXT_LENGTH = "context_length_exceeded"
CODE_INSUFFICIENT_QUOTA = "insufficient_quota"
CODE_USAGE_NOT_INCLUDED = "usage_not_included"
CODE_INVALID_PROMPT = "invalid_prompt"
CODE_SERVER_OVERLOADED = "server_is_overloaded"
CODE_RATE_LIMIT = "rate_limit_exceeded"
CODE_SERVER_ERROR = "server_error"  # fallback; codex treats as retryable


@dataclass
class ResponsesError:
    """
    Normalized view of an upstream failure, ready to be serialized into
    the `response.failed` payload.

    Attributes:
        code: One of the codex-recognized codes above, or a fallback string.
        message: Human-readable message. For rate_limit_exceeded, must contain
            the wording "try again in <N>s" (or "<N>ms") so codex's regex can
            parse the retry delay.
    """

    code: str
    message: str


def _classify_by_reason(reason: str) -> Optional[str]:
    """Map Kiro `reason` codes (from kiro_errors.KiroErrorReason) to codex codes."""
    reason_upper = (reason or "").upper()
    if reason_upper == "CONTENT_LENGTH_EXCEEDS_THRESHOLD":
        return CODE_CONTEXT_LENGTH
    if reason_upper == "MONTHLY_REQUEST_COUNT":
        return CODE_INSUFFICIENT_QUOTA
    return None


def _classify_by_message(message: str) -> Optional[str]:
    """
    Best-effort content-based classification when `reason` isn't set.

    Kiro's "Improperly formed request" umbrella error (see CLAUDE.md) means we
    cannot rely on reason alone; some quota / context messages arrive with
    only a message string.
    """
    if not message:
        return None
    lowered = message.lower()
    if "input is too long" in lowered or "context" in lowered and "exceed" in lowered:
        return CODE_CONTEXT_LENGTH
    if "monthly request" in lowered or "quota" in lowered:
        return CODE_INSUFFICIENT_QUOTA
    if "rate limit" in lowered or "too many requests" in lowered or "throttl" in lowered:
        return CODE_RATE_LIMIT
    if "overload" in lowered or "slow down" in lowered:
        return CODE_SERVER_OVERLOADED
    if "improperly formed" in lowered or "invalid" in lowered and "request" in lowered:
        return CODE_INVALID_PROMPT
    return None


def _classify_by_status(status_code: int) -> str:
    """
    Fallback based on HTTP status when we have nothing else.

    400 -> invalid_prompt (client-side)
    401/403 -> invalid_prompt (auth / permission errors are fatal to codex)
    429 -> rate_limit_exceeded
    5xx -> server_is_overloaded
    else -> server_error
    """
    if status_code == 429:
        return CODE_RATE_LIMIT
    if 500 <= status_code < 600:
        return CODE_SERVER_OVERLOADED
    if status_code in (400, 401, 403, 404, 422):
        return CODE_INVALID_PROMPT
    return CODE_SERVER_ERROR


def classify_upstream_error(
    status_code: Optional[int],
    error_info: Optional[KiroErrorInfo],
    raw_message: Optional[str] = None,
) -> ResponsesError:
    """
    Produce a `ResponsesError` from an upstream Kiro failure.

    Resolution order:
    1. If we have an `EnhancedKiroError` with a known `reason`, use it.
    2. Else, content-match the message.
    3. Else, fall back to HTTP status.

    Args:
        status_code: HTTP status from Kiro (None if unknown / transport failure).
        error_info: Parsed `KiroErrorInfo` from `enhance_kiro_error`, or None.
        raw_message: Free-form error text (used only if `error_info` is None).

    Returns:
        ResponsesError with codex-ready `code` and a message safe to surface.
    """
    message = ""
    code: Optional[str] = None

    if error_info is not None:
        message = error_info.user_message or error_info.original_message or ""
        code = _classify_by_reason(error_info.reason)
        if code is None:
            code = _classify_by_message(error_info.original_message or message)
    else:
        message = raw_message or ""
        code = _classify_by_message(message)

    if code is None and status_code is not None:
        code = _classify_by_status(status_code)

    if code is None:
        code = CODE_SERVER_ERROR

    # For rate_limit_exceeded, make sure the message contains the exact
    # wording codex's regex expects. Kiro sometimes returns plain 429s
    # without any Retry-After hint; in that case synthesize a minimal one so
    # codex backs off instead of tight-looping.
    if code == CODE_RATE_LIMIT:
        if "try again in" not in (message or "").lower():
            suffix = " Please try again in 30s."
            message = (message or "Rate limit exceeded.") + suffix

    if not message:
        message = "Upstream error from Kiro API."

    return ResponsesError(code=code, message=message)


def classify_exception(exc: BaseException) -> ResponsesError:
    """
    Classify a mid-stream exception into a ResponsesError.

    Mid-stream we rarely have a status code; we only have an exception
    (timeout, connection reset, JSON parse error, etc.). We map by type name
    and message to the nearest codex bucket.

    Args:
        exc: The exception raised mid-stream.

    Returns:
        ResponsesError with codex-ready `code` and message.
    """
    name = type(exc).__name__
    raw = str(exc) or name

    # Network/timeout errors are retryable at codex's layer.
    if "Timeout" in name or "Connect" in name or "ReadError" in name:
        return ResponsesError(
            code=CODE_SERVER_OVERLOADED,
            message=f"Upstream connection error: {raw}",
        )

    # Try content-based classification first.
    code = _classify_by_message(raw)
    if code is None:
        code = CODE_SERVER_ERROR
    if code == CODE_RATE_LIMIT and "try again in" not in raw.lower():
        raw = raw + " Please try again in 30s."
    return ResponsesError(code=code, message=raw)


def build_response_failed_event(
    *,
    response_id: str,
    created_at: int,
    model: str,
    error: ResponsesError,
) -> Dict[str, Any]:
    """
    Build the `response.failed` payload body.

    This is the inner `response` object; the caller still wraps it with the
    `{"type": "response.failed", "response": ...}` envelope via `_sse_event`
    or `_ws_event`.

    Args:
        response_id: Stable response ID to use. When possible, pass the same
            id that was emitted in `response.created` so codex can correlate.
        created_at: Unix timestamp, seconds.
        model: Model name echoed back to the client.
        error: Classified error.

    Returns:
        Dict shaped like a Responses API failed response object.
    """
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": "failed",
        "error": {
            "code": error.code,
            "message": error.message,
        },
        "output": [],
        "usage": None,
        "incomplete_details": None,
    }
