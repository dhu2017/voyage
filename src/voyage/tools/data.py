"""MCP Tool: get_index_data — fetch index historical data.

Resolution priority (lowest cost first):
    1. 通达信本地 vipdoc (.day 文件)            ← 优先, 离线, 无限速
    2. 项目内 parquet 缓存 (data/*.parquet)
    3. 远程 (akshare / yfinance) — 仅在本地不覆盖目标区间时调用
"""

import pandas as pd
from datetime import datetime
from typing import Optional

from ..data import resolve_index, suggest_candidates
from ..data.cache import read_cache, merge_and_save, get_cached_date_range, write_cache
from ..data import akshare_client, yfinance_client, tdx_client
from ..engine.indicators import annualized_volatility

def get_index_data(
    index: str,
    start_date: str,
    end_date: Optional[str] = None,
    fields: Optional[list[str]] = None,
    summary_only: bool = False,
) -> dict:
    """Fetch index historical data with TDX-first / cache / remote fallback."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if fields is None:
        fields = ["close"]

    info = resolve_index(index)
    if info is None:
        return {
            "error": f"无法识别指数 '{index}'",
            "candidates": suggest_candidates(index),
            "hint": "请使用上述候选名称重试",
        }

    df, used_source = _load_data(info, start_date, end_date)
    if df is None or df.empty:
        return {"error": f"指数 '{info.name}' 在 {start_date} ~ {end_date} 无数据"}

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return {"error": f"指数 '{info.name}' 在 {start_date} ~ {end_date} 无数据"}

    output_cols = ["date"] + [f for f in fields if f in df.columns]
    result_df = df[output_cols].copy()
    result_df["date"] = result_df["date"].dt.strftime("%Y-%m-%d")

    if "close" in df.columns:
        close = df["close"].astype(float)
        total_return = float(close.iloc[-1] / close.iloc[0] - 1)
        ann_vol = annualized_volatility(close)
        summary = {
            "index_name": info.name,
            "source": used_source,
            "period": f"{start_date} ~ {end_date}",
            "data_points": len(result_df),
            "total_return": round(total_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "latest_close": round(float(close.iloc[-1]), 2),
        }
    else:
        summary = {
            "index_name": info.name,
            "source": used_source,
            "period": f"{start_date} ~ {end_date}",
            "data_points": len(result_df),
        }

    if summary_only:
        return {"summary": summary}

    return {
        "data": result_df.to_dict(orient="records"),
        "summary": summary,
    }


def _load_data(info, start_date: str, end_date: str) -> tuple[Optional[pd.DataFrame], str]:
    """Resolve data following the TDX → parquet → remote priority.

    Policy: 本地有就用本地, 不再因为末尾差几天就打远程 (akshare/yfinance 经常限速).
    只有当本地数据不存在 OR 本地数据起点晚于 start_date 时才会触发远程。
    Returns (df, used_source_label).
    """
    # 1) Local TDX: vipdoc + refreshed parquet cache (mootdx-populated) union
    tdx_df = _try_tdx(info)
    if tdx_df is not None and _covers_start(tdx_df, start_date):
        return tdx_df, f"tdx:{_tdx_cache_key(info)}"

    # 2) Project parquet cache (under primary or fallback key)
    source, code = info.source, info.code
    cached_df = read_cache(source, code)
    fb_cached_df = None
    if info.fallback_source and info.fallback_code:
        fb_cached_df = read_cache(info.fallback_source, info.fallback_code)
    if cached_df is None and fb_cached_df is not None:
        cached_df = fb_cached_df
        source, code = info.fallback_source, info.fallback_code

    if cached_df is not None and _covers_start(cached_df, start_date):
        return cached_df, f"cache:{source}:{code}"

    # 3) Remote fetch (akshare / yfinance) — only when nothing local covers start_date
    new_df = _fetch_with_fallback(info, start_date, end_date)
    actual_source = new_df.attrs.get("source", source)
    actual_code = new_df.attrs.get("code", code)
    df = merge_and_save(actual_source, actual_code, cached_df, new_df)

    if tdx_df is not None:
        df = _merge_tdx_with_remote(tdx_df, df)

    return df, f"remote:{actual_source}"


def _try_tdx(info) -> Optional[pd.DataFrame]:
    """Return unioned (vipdoc ∪ refreshed-parquet) history for this index, or None.

    vipdoc is canonical history; the refreshed parquet may contain newer rows added
    by mootdx via refresh_local_data. On date overlap, vipdoc wins.
    """
    for market, code in info.tdx_candidates:
        f = tdx_client.find_day_file(code, prefer_market=market)
        vipdoc_df = tdx_client.read_day_file(f) if f is not None else None
        refreshed = read_cache("tdx", f"{market}_{code}")
        merged = _union_tdx(vipdoc_df, refreshed)
        if merged is not None and not merged.empty:
            return merged
    return None


def _union_tdx(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Union two TDX frames on date, prefer rows in `a` (vipdoc) on conflict."""
    if (a is None or a.empty) and (b is None or b.empty):
        return None
    if a is None or a.empty:
        return b.copy()
    if b is None or b.empty:
        return a.copy()
    a = a.copy(); b = b.copy()
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])
    extra = b[~b["date"].isin(a["date"])]
    cols = [c for c in a.columns if c in extra.columns]
    out = pd.concat([a[cols], extra[cols]], ignore_index=True)
    return out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def _tdx_cache_key(info) -> str:
    """Stable parquet key for TDX-sourced data of this index."""
    if info.tdx_candidates:
        market, code = info.tdx_candidates[0]
        return f"{market}_{code}"
    return info.code


def _covers_start(df: pd.DataFrame, start_date: str) -> bool:
    """本地数据起点是否覆盖 start_date。末端差几天不触发远程(避开限速)."""
    if df is None or df.empty or "date" not in df.columns:
        return False
    dates = pd.to_datetime(df["date"])
    return dates.min() <= pd.Timestamp(start_date)


def _merge_tdx_with_remote(tdx_df: pd.DataFrame, remote_df: pd.DataFrame) -> pd.DataFrame:
    """Union TDX history (longer) with remote (more current). De-dupe on date, prefer TDX."""
    a = tdx_df.copy()
    b = remote_df.copy()
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])
    # rows only in remote (newer than TDX or earlier missing dates)
    b_extra = b[~b["date"].isin(a["date"])]
    common_cols = [c for c in a.columns if c in b_extra.columns]
    merged = pd.concat([a[common_cols], b_extra[common_cols]], ignore_index=True)
    return merged.sort_values("date").reset_index(drop=True)


def _fetch_with_fallback(info, start_date: str, end_date: str) -> pd.DataFrame:
    """Try primary source first, fallback to alternative on failure."""
    try:
        df = _fetch_from_source(info.source, info.code, start_date, end_date)
        df.attrs["source"] = info.source
        df.attrs["code"] = info.code
        return df
    except Exception as primary_err:
        if info.fallback_source and info.fallback_code:
            try:
                df = _fetch_from_source(info.fallback_source, info.fallback_code, start_date, end_date)
                df.attrs["source"] = info.fallback_source
                df.attrs["code"] = info.fallback_code
                return df
            except Exception:
                pass
        raise primary_err


def _fetch_from_source(source: str, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Dispatch to the correct data client."""
    if source == "akshare":
        if code == "bond_zh_10y":
            return akshare_client.fetch_bond_yield(start_date, end_date)
        return akshare_client.fetch_index_daily(code, start_date, end_date)
    elif source == "yfinance":
        return yfinance_client.fetch_index_daily(code, start_date, end_date)
    elif source == "akshare_us":
        return akshare_client.fetch_us_index_daily(code, start_date, end_date)
    elif source == "akshare_gold":
        return akshare_client.fetch_gold_daily(start_date, end_date)
    elif source == "akshare_hk":
        return akshare_client.fetch_hk_index_daily(code, start_date, end_date)
    elif source == "akshare_hk_sina":
        return akshare_client.fetch_hk_index_daily_sina(code, start_date, end_date)
    else:
        raise ValueError(f"Unknown data source: {source}")


def list_local_data() -> dict:
    """Return a snapshot of locally-available data: TDX vipdoc + parquet cache."""
    tdx_info = tdx_client.scan_vipdoc()

    from ..data.cache import _CACHE_DIR
    parquet_files = []
    if _CACHE_DIR.exists():
        for p in sorted(_CACHE_DIR.glob("*.parquet")):
            parquet_files.append({"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)})

    # Map registered indices → which local source(s) they hit + local freshness
    coverage = []
    for info_dict in [
        {"name": i.name, "tdx_candidates": list(i.tdx_candidates), "source": i.source, "code": i.code}
        for i in __import__("voyage.data", fromlist=["_REGISTRY"])._REGISTRY
    ]:
        hits = []
        latest = None
        for market, code in info_dict["tdx_candidates"]:
            f = tdx_client.find_day_file(code, prefer_market=market)
            vipdoc_df = tdx_client.read_day_file(f) if f is not None else None
            refreshed = read_cache("tdx", f"{market}_{code}")
            merged = _union_tdx(vipdoc_df, refreshed)
            if merged is None or merged.empty:
                continue
            last_date = merged["date"].iloc[-1].strftime("%Y-%m-%d")
            vipdoc_last = vipdoc_df["date"].iloc[-1].strftime("%Y-%m-%d") if vipdoc_df is not None and not vipdoc_df.empty else None
            if latest is None or last_date > latest:
                latest = last_date
            hits.append({
                "market": market,
                "code": code,
                "vipdoc_path": str(f.path) if f is not None else None,
                "vipdoc_latest": vipdoc_last,
                "merged_latest": last_date,
                "rows": len(merged),
            })
        coverage.append({
            "name": info_dict["name"],
            "remote_source": info_dict["source"],
            "remote_code": info_dict["code"],
            "tdx_hits": hits,
            "latest_local": latest,
        })

    return {
        "tdx": tdx_info,
        "parquet_cache": parquet_files,
        "coverage": coverage,
    }


def refresh_local_data(index: Optional[str] = None) -> dict:
    """通过 mootdx 直连通达信行情服务器补全本地数据。

    - 无参: 刷新所有挂了 tdx_candidates 的指数 (A股指数)。
    - index=某名称/别名: 仅刷新这一个。
    - 外盘 / 黄金 / 国债 (没有 tdx_candidates) 会被跳过; 这些走 akshare/yfinance。

    数据会 append 到项目 parquet 缓存 (data/tdx_{market}_{code}_daily.parquet),
    合并时以 vipdoc 本地历史为底, mootdx 返回的新数据往后拼接并去重。
    **不写入通达信 vipdoc 目录** — 那是通达信客户端的地盘。

    Returns:
        per-index status dict including rows added & latest_date.
    """
    from ..data import _REGISTRY
    from . import _tdx_refresh  # lazy import to avoid pulling mootdx at module load

    if index:
        info = resolve_index(index)
        if info is None:
            return {"error": f"无法识别指数 '{index}'", "candidates": suggest_candidates(index)}
        targets = [info]
    else:
        targets = [i for i in _REGISTRY if i.tdx_candidates]

    results = []
    for info in targets:
        if not info.tdx_candidates:
            results.append({"name": info.name, "status": "skipped", "reason": "no tdx_candidates (remote-only asset)"})
            continue
        market, code = info.tdx_candidates[0]
        try:
            r = _tdx_refresh.refresh_one(market, code)
            r["name"] = info.name
            results.append(r)
        except Exception as e:
            results.append({
                "name": info.name,
                "market": market,
                "code": code,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })

    return {"refreshed": results}
