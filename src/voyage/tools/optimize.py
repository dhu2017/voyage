"""MCP Tool: optimize_portfolio — portfolio optimization."""

from typing import Optional
from ..tools.data import get_index_data
from ..engine.portfolio import (
    equal_weight, risk_parity, mean_variance,
    hierarchical_risk_parity, portfolio_expected_stats,
)
import pandas as pd


def optimize_portfolio(
    assets: list[str],
    method: str = "risk_parity",
    risk_free_rate: float = 0.02,
    lookback_days: int = 252,
    constraints: Optional[dict] = None,
) -> dict:
    """
    Optimize portfolio weights for a given set of assets.

    Args:
        assets: List of index names (e.g., ["沪深300", "纳斯达克100", "黄金"])
        method: "risk_parity" | "mvo" | "hrp" | "equal_weight"
        risk_free_rate: Annual risk-free rate (default 0.02)
        lookback_days: Lookback window in trading days (default 252)
        constraints: Optional {"max_weight": 0.4, "min_weight": 0.05}

    Returns:
        dict with weights, expected stats, method info
    """
    if method == "equal_weight":
        weights = equal_weight(assets)
        # Still need prices for expected stats
        prices_dict = _fetch_prices(assets, lookback_days)
        if "error" in prices_dict:
            return prices_dict
        stats = portfolio_expected_stats(prices_dict, weights, risk_free_rate)
        return {
            "method": "equal_weight",
            "weights": weights,
            "portfolio_stats": stats,
        }

    # Need price data for optimization
    prices_dict = _fetch_prices(assets, lookback_days)
    if "error" in prices_dict:
        return prices_dict

    try:
        if method == "risk_parity":
            weights = risk_parity(prices_dict, constraints)
        elif method == "mvo":
            weights = mean_variance(prices_dict, risk_free_rate, constraints)
        elif method == "hrp":
            weights = hierarchical_risk_parity(prices_dict, constraints)
        else:
            return {"error": f"不支持的优化方法: {method}，可选: risk_parity, mvo, hrp, equal_weight"}
    except Exception as e:
        return {"error": f"优化失败: {str(e)}"}

    stats = portfolio_expected_stats(prices_dict, weights, risk_free_rate)

    return {
        "method": method,
        "weights": weights,
        "portfolio_stats": stats,
        "lookback_days": lookback_days,
        "risk_free_rate": risk_free_rate,
    }


def _fetch_prices(assets: list[str], lookback_days: int) -> dict:
    """Fetch close prices for all assets, return dict of pd.Series keyed by name."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=int(lookback_days * 1.5))).strftime("%Y-%m-%d")

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

    return prices_dict
