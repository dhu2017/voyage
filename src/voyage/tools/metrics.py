"""MCP Tool: calc_metrics & correlation_matrix."""

from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..tools.data import get_index_data
from ..engine.indicators import compute_all_metrics


def _period_to_dates(period: str) -> tuple[str, str]:
    """Convert period string to (start_date, end_date)."""
    end = datetime.now()
    end_str = end.strftime("%Y-%m-%d")

    if period == "ytd":
        start_str = f"{end.year}-01-01"
    elif period == "1y":
        start_str = (end - timedelta(days=365)).strftime("%Y-%m-%d")
    elif period == "3y":
        start_str = (end - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    elif period == "5y":
        start_str = (end - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    elif period == "max":
        start_str = "2000-01-01"
    else:
        start_str = (end - timedelta(days=365)).strftime("%Y-%m-%d")

    return start_str, end_str


def calc_metrics(
    targets: list[str],
    period: str = "1y",
    benchmark: Optional[str] = None,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Calculate risk/return metrics for one or more assets.

    Args:
        targets: List of index names
        period: "1y" | "3y" | "5y" | "ytd" | "max"
        benchmark: Optional benchmark index name
        risk_free_rate: Annual risk-free rate

    Returns:
        dict with metrics per target
    """
    start_date, end_date = _period_to_dates(period)

    # Fetch benchmark if specified
    benchmark_prices = None
    if benchmark:
        bench_result = get_index_data(benchmark, start_date, end_date, fields=["close"])
        if "error" not in bench_result:
            bench_df = pd.DataFrame(bench_result["data"])
            bench_df["date"] = pd.to_datetime(bench_df["date"])
            benchmark_prices = bench_df.set_index("date")["close"].astype(float)

    results = {}
    for target in targets:
        result = get_index_data(target, start_date, end_date, fields=["close"])
        if "error" in result:
            results[target] = {"error": result["error"]}
            continue

        df = pd.DataFrame(result["data"])
        df["date"] = pd.to_datetime(df["date"])
        prices = df.set_index("date")["close"].astype(float)

        if len(prices) < 10:
            results[target] = {"error": "数据不足，无法计算指标"}
            continue

        metrics = compute_all_metrics(prices, risk_free_rate, benchmark_prices)
        results[target] = metrics

    return {"period": period, "metrics": results}


def correlation_matrix(
    assets: list[str],
    period: str = "1y",
    method: str = "pearson",
) -> dict:
    """
    Compute correlation matrix for a set of assets.

    Args:
        assets: List of index names
        period: "1y" | "3y" | "5y"
        method: "pearson" | "spearman"

    Returns:
        dict with correlation matrix and text summary
    """
    start_date, end_date = _period_to_dates(period)

    # Fetch all price series
    returns_dict = {}
    for asset in assets:
        result = get_index_data(asset, start_date, end_date, fields=["close"])
        if "error" in result:
            return {"error": f"获取 {asset} 数据失败: {result['error']}"}
        df = pd.DataFrame(result["data"])
        df["date"] = pd.to_datetime(df["date"])
        series = df.set_index("date")["close"].astype(float).pct_change().dropna()
        returns_dict[asset] = series

    # Build returns DataFrame and compute correlation
    returns_df = pd.DataFrame(returns_dict).dropna()

    if len(returns_df) < 10:
        return {"error": "数据重叠不足，无法计算相关性"}

    corr = returns_df.corr(method=method)

    # Build matrix as list of lists
    matrix = corr.round(4).values.tolist()

    # Text summary: find highest and lowest correlations
    summary_lines = []
    n = len(assets)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((assets[i], assets[j], corr.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    if pairs:
        high = pairs[0]
        low = pairs[-1]
        summary_lines.append(f"最高相关: {high[0]} & {high[1]} = {high[2]:.4f}")
        summary_lines.append(f"最低相关: {low[0]} & {low[1]} = {low[2]:.4f}")

    return {
        "assets": assets,
        "method": method,
        "period": period,
        "matrix": matrix,
        "summary": summary_lines,
    }
