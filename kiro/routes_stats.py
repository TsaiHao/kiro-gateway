# -*- coding: utf-8 -*-

"""
API routes for token usage statistics.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from kiro.routes_openai import verify_api_key

router = APIRouter()


@router.get("/v1/stats/tokens", dependencies=[Depends(verify_api_key)])
async def token_summary(
    request: Request,
    model: Optional[str] = Query(None, description="Filter by model"),
):
    """Total token usage per model (within 2-week retention window)."""
    stats = request.app.state.token_stats
    data = stats.get_summary(model=model)
    total_input = sum(r["input_tokens"] for r in data)
    total_output = sum(r["output_tokens"] for r in data)
    total_requests = sum(r["request_count"] for r in data)
    return JSONResponse(content={
        "total": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "request_count": total_requests,
        },
        "by_model": data,
    })


@router.get("/v1/stats/tokens/daily", dependencies=[Depends(verify_api_key)])
async def token_daily(
    request: Request,
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    model: Optional[str] = Query(None, description="Filter by model"),
):
    """Daily token usage breakdown."""
    stats = request.app.state.token_stats
    data = stats.get_daily(start_date=start, end_date=end, model=model)
    return JSONResponse(content={"daily": data})


@router.get("/v1/stats/tokens/hourly", dependencies=[Depends(verify_api_key)])
async def token_hourly(
    request: Request,
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to today"),
    model: Optional[str] = Query(None, description="Filter by model"),
):
    """Hourly token usage breakdown for a specific day."""
    stats = request.app.state.token_stats
    data = stats.get_hourly(date_str=date, model=model)
    return JSONResponse(content={"date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"), "hourly": data})
