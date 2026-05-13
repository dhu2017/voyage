"""Event study helper functions — reusable building blocks for dynamic analysis scripts."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def load_prices(asset_name: str, start_date: str = "2005-01-01", end_date: str = None) -> pd.Series:
    """
    便捷函数：名称解析 → 读缓存 → 返回 close Series (DatetimeIndex)。
    如果缓存不存在或不够新，调用方应先通过 MCP get_index_data 触发缓存更新。
    """
    from voyage.data import resolve_index
    from voyage.data.cache import read_cache

    info = resolve_index(asset_name)
    if info is None:
        raise ValueError(f"无法识别资产: {asset_name}")
    df = read_cache(info.source, info.code)
    if df is None:
        if info.fallback_source and info.fallback_code:
            df = read_cache(info.fallback_source, info.fallback_code)
        if df is None:
            raise ValueError(f"缓存中无 {asset_name} 数据，请先调用 get_index_data 获取")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if end_date is None:
        end_date = df.index.max().strftime("%Y-%m-%d")
    series = df.loc[start_date:end_date, "close"].astype(float)
    return series


def post_event_returns(
    prices: pd.Series,
    event_dates: pd.DatetimeIndex,
    holding_period: int,
) -> np.ndarray:
    """
    计算每个事件后 holding_period 个交易日的收益率。
    事件日期不在 prices 中的会被跳过，持有期超出数据范围的也跳过。
    """
    all_dates = prices.index
    returns = []
    for d in event_dates:
        if d not in all_dates:
            continue
        loc = all_dates.get_loc(d)
        end_loc = loc + holding_period
        if end_loc >= len(all_dates):
            continue
        ret = prices.iloc[end_loc] / prices.iloc[loc] - 1
        returns.append(ret)
    return np.array(returns)


def return_stats(returns: np.ndarray) -> dict:
    """收益分布统计量。"""
    if len(returns) == 0:
        return {"count": 0}
    return {
        "count": len(returns),
        "mean": round(float(np.mean(returns)), 4),
        "median": round(float(np.median(returns)), 4),
        "std": round(float(np.std(returns, ddof=1)), 4) if len(returns) > 1 else 0.0,
        "min": round(float(np.min(returns)), 4),
        "max": round(float(np.max(returns)), 4),
        "q25": round(float(np.percentile(returns, 25)), 4),
        "q75": round(float(np.percentile(returns, 75)), 4),
        "positive_rate": round(float(np.mean(returns > 0)), 4),
    }


def benchmark_comparison(
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> dict:
    """
    超额收益统计 + 显著性检验。
    asset_returns 和 benchmark_returns 必须一一对应（同一组事件）。
    """
    if len(asset_returns) == 0 or len(benchmark_returns) == 0:
        return {"count": 0}
    excess = asset_returns - benchmark_returns
    win_rate = float(np.mean(excess > 0))
    if len(excess) > 1 and np.std(excess, ddof=1) > 0:
        t_stat, p_value = stats.ttest_1samp(excess, 0)
    else:
        t_stat, p_value = 0.0, 1.0
    return {
        "excess_mean": round(float(np.mean(excess)), 4),
        "excess_median": round(float(np.median(excess)), 4),
        "win_rate": round(win_rate, 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant_5pct": bool(p_value < 0.05),
    }


def decluster(
    event_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    min_gap: int,
) -> pd.DatetimeIndex:
    """
    事件去聚集：贪心保留，触发后跳过 min_gap-1 个交易日。
    """
    if min_gap <= 1 or len(event_dates) == 0:
        return event_dates
    positions = [all_dates.get_loc(d) for d in event_dates if d in all_dates]
    positions.sort()
    kept = []
    last_pos = -min_gap
    for pos in positions:
        if pos - last_pos >= min_gap:
            kept.append(all_dates[pos])
            last_pos = pos
    return pd.DatetimeIndex(kept)
