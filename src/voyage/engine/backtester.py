"""Backtesting engine for portfolio strategies."""

import pandas as pd
import numpy as np
from typing import Optional
from ..engine.indicators import compute_all_metrics


def run_backtest(
    prices_dict: dict[str, pd.Series],
    weights: dict[str, float],
    start_date: str,
    end_date: str,
    rebalance: str = "quarterly",
    initial_capital: float = 1_000_000,
    transaction_cost: float = 0.001,
    benchmark_prices: Optional[pd.Series] = None,
) -> dict:
    """
    Run a historical backtest on a weighted portfolio.

    Args:
        prices_dict: {asset_name: price_series} with DatetimeIndex or date column
        weights: {asset_name: weight}
        start_date / end_date: date range
        rebalance: "monthly" | "quarterly" | "yearly" | "none"
        initial_capital: starting capital
        transaction_cost: proportional cost per trade
        benchmark_prices: optional benchmark for comparison

    Returns:
        dict with nav_curve, metrics, rebalance_log, benchmark_comparison
    """
    # Build aligned price DataFrame
    prices_df = pd.DataFrame(prices_dict).dropna()
    prices_df.index = pd.to_datetime(prices_df.index) if not isinstance(prices_df.index, pd.DatetimeIndex) else prices_df.index
    prices_df = prices_df.loc[start_date:end_date]

    if prices_df.empty or len(prices_df) < 2:
        raise ValueError("Insufficient data for the given date range")

    assets = list(weights.keys())
    w = np.array([weights[a] for a in assets])

    # Daily returns
    returns_df = prices_df[assets].pct_change().fillna(0.0)

    # Determine rebalance dates
    rebal_dates = _rebalance_dates(prices_df.index, rebalance)

    # Simulate
    n_days = len(prices_df)
    nav = np.zeros(n_days)
    nav[0] = initial_capital
    current_weights = w.copy()
    rebalance_log = []

    for i in range(1, n_days):
        date = prices_df.index[i]
        daily_ret = returns_df.iloc[i][assets].values

        # Portfolio return for the day
        port_ret = np.dot(current_weights, daily_ret)
        nav[i] = nav[i - 1] * (1 + port_ret)

        # Update drifted weights
        current_weights = current_weights * (1 + daily_ret)
        weight_sum = current_weights.sum()
        if weight_sum > 0:
            current_weights = current_weights / weight_sum

        # Rebalance check
        if date in rebal_dates:
            turnover = np.sum(np.abs(current_weights - w))
            cost = nav[i] * turnover * transaction_cost
            nav[i] -= cost
            current_weights = w.copy()
            rebalance_log.append({
                "date": date.strftime("%Y-%m-%d"),
                "turnover": round(float(turnover), 4),
                "cost": round(float(cost), 2),
                "nav_after": round(float(nav[i]), 2),
            })

    # Build NAV series
    nav_series = pd.Series(nav, index=prices_df.index)

    # Compute metrics
    metrics = compute_all_metrics(nav_series, benchmark_prices=benchmark_prices)

    # NAV curve as list of {date, nav}
    # Sample to max 100 points to avoid bloating MCP response
    nav_full = list(zip(prices_df.index, nav))
    if len(nav_full) <= 100:
        nav_curve = [
            {"date": d.strftime("%Y-%m-%d"), "nav": round(float(v), 2)}
            for d, v in nav_full
        ]
    else:
        # Sample evenly + always include first/last
        step = len(nav_full) // 99
        sampled = [nav_full[0]] + [nav_full[i] for i in range(step, len(nav_full), step)] + [nav_full[-1]]
        nav_curve = [
            {"date": d.strftime("%Y-%m-%d"), "nav": round(float(v), 2)}
            for d, v in sampled
        ]

    result = {
        "nav_curve": nav_curve,
        "metrics": metrics,
        "rebalance_log": rebalance_log,
        "initial_capital": initial_capital,
        "final_nav": round(float(nav[-1]), 2),
        "total_return": round(float(nav[-1] / nav[0] - 1), 4),
    }

    # Benchmark comparison
    if benchmark_prices is not None:
        bench = benchmark_prices.loc[start_date:end_date].dropna()
        if len(bench) > 1:
            bench_metrics = compute_all_metrics(bench)
            result["benchmark_metrics"] = bench_metrics
            result["benchmark_total_return"] = round(float(bench.iloc[-1] / bench.iloc[0] - 1), 4)

    return result


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """Determine rebalance dates from the index."""
    if freq == "none":
        return set()

    dates = set()
    if freq == "monthly":
        for i in range(1, len(index)):
            if index[i].month != index[i - 1].month:
                dates.add(index[i])
    elif freq == "quarterly":
        for i in range(1, len(index)):
            if (index[i].month - 1) // 3 != (index[i - 1].month - 1) // 3:
                dates.add(index[i])
    elif freq == "yearly":
        for i in range(1, len(index)):
            if index[i].year != index[i - 1].year:
                dates.add(index[i])

    return dates
