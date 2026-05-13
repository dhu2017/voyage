"""MCP Tool: run_backtest — historical portfolio backtesting."""

from typing import Optional
import pandas as pd
from ..tools.data import get_index_data
from ..engine.backtester import run_backtest as _run_backtest


def run_backtest(
    weights: dict[str, float],
    start_date: str,
    end_date: str,
    rebalance: str = "quarterly",
    initial_capital: float = 1_000_000,
    transaction_cost: float = 0.001,
    benchmark: Optional[str] = None,
) -> dict:
    """
    Backtest a weighted portfolio over a historical period.

    Args:
        weights: {asset_name: weight}, e.g. {"沪深300": 0.3, "黄金": 0.2}
        start_date / end_date: date range "YYYY-MM-DD"
        rebalance: "monthly" | "quarterly" | "yearly" | "none"
        initial_capital: starting capital (default 1,000,000)
        transaction_cost: proportional cost per trade (default 0.001)
        benchmark: optional benchmark index name for comparison

    Returns:
        dict with nav_curve, metrics, rebalance_log, benchmark_comparison
    """
    assets = list(weights.keys())

    # Fetch price data for all assets (sequential — py_mini_racer crashes under ThreadPool)
    prices_dict = {}
    for asset in assets:
        try:
            result = get_index_data(asset, start_date, end_date, fields=["close"])
            if "error" in result:
                return {"error": f"获取 {asset} 数据失败: {result['error']}"}
            df = pd.DataFrame(result["data"])
            df["date"] = pd.to_datetime(df["date"])
            prices_dict[asset] = df.set_index("date")["close"].astype(float)
        except Exception as e:
            return {"error": f"获取 {asset} 数据失败: {str(e)}"}

    # Fetch benchmark
    benchmark_prices = None
    if benchmark:
        try:
            bench_result = get_index_data(benchmark, start_date, end_date, fields=["close"])
            if "error" not in bench_result:
                bench_df = pd.DataFrame(bench_result["data"])
                bench_df["date"] = pd.to_datetime(bench_df["date"])
                benchmark_prices = bench_df.set_index("date")["close"].astype(float)
        except Exception:
            pass  # benchmark failure is non-fatal

    try:
        result = _run_backtest(
            prices_dict=prices_dict,
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            rebalance=rebalance,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            benchmark_prices=benchmark_prices,
        )
    except Exception as e:
        return {"error": f"回测失败: {str(e)}"}

    result["config"] = {
        "weights": weights,
        "start_date": start_date,
        "end_date": end_date,
        "rebalance": rebalance,
        "initial_capital": initial_capital,
        "transaction_cost": transaction_cost,
        "benchmark": benchmark,
    }

    return result
