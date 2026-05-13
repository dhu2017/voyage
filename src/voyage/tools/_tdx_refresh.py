"""Refresh helper: merge vipdoc local + mootdx online, persist to parquet cache.

Split out so tools/data.py doesn't pull mootdx at import time.
"""

from __future__ import annotations

import pandas as pd

from ..data import tdx_client
from ..data.cache import read_cache, write_cache


def refresh_one(market: str, code: str) -> dict:
    """Merge vipdoc + mootdx online for a single (market, code), write to parquet.

    Returns status dict with rows_before, rows_after, added, latest_date.
    """
    from ..data import tdx_online  # lazy — avoid importing mootdx on hot path

    # 1) Load local vipdoc history (may be empty if the user hasn't synced via client)
    f = tdx_client.find_day_file(code, prefer_market=market)
    local_df = tdx_client.read_day_file(f) if f is not None else pd.DataFrame()

    # 2) Also merge any existing parquet cache (in case a prior refresh wrote beyond vipdoc)
    cache_key = f"{market}_{code}"
    cached = read_cache("tdx", cache_key)
    base = _union(local_df, cached)

    rows_before = len(base)
    latest_before = base["date"].iloc[-1] if rows_before else None

    # 3) Pull from mootdx online
    online_df = tdx_online.fetch_online(market, code)

    # 4) Union + dedupe (prefer local values on conflict — vipdoc is the canonical format)
    merged = _union(base, online_df)

    # 5) Persist
    write_cache("tdx", cache_key, merged)

    rows_after = len(merged)
    latest_after = merged["date"].iloc[-1] if rows_after else None

    return {
        "market": market,
        "code": code,
        "status": "ok",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "added": rows_after - rows_before,
        "latest_before": latest_before.strftime("%Y-%m-%d") if latest_before is not None else None,
        "latest_after": latest_after.strftime("%Y-%m-%d") if latest_after is not None else None,
    }


def _union(a: pd.DataFrame | None, b: pd.DataFrame | None) -> pd.DataFrame:
    """Union two date-indexed dataframes, prefer rows in `a` (local vipdoc) on conflict."""
    if a is None or a.empty:
        return b.copy() if b is not None else pd.DataFrame()
    if b is None or b.empty:
        return a.copy()
    a = a.copy(); b = b.copy()
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])
    extra = b[~b["date"].isin(a["date"])]
    cols = [c for c in a.columns if c in extra.columns]
    out = pd.concat([a[cols], extra[cols]], ignore_index=True)
    return out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
