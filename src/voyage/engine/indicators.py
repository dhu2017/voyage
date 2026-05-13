"""Risk/return indicators calculation."""

import pandas as pd
import numpy as np
from typing import Optional


def annualized_return(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return from a price series."""
    if len(prices) < 2:
        return 0.0
    total_return = prices.iloc[-1] / prices.iloc[0]
    n_periods = len(prices) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return total_return ** (1 / years) - 1


def annualized_volatility(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility from a price series."""
    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    ann_ret = annualized_return(prices, periods_per_year)
    ann_vol = annualized_volatility(prices, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(prices: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    returns = prices.pct_change().dropna()
    ann_ret = annualized_return(prices, periods_per_year)
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std() * np.sqrt(periods_per_year)
    if downside_std == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / downside_std


def max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown (returned as a negative number)."""
    if len(prices) < 2:
        return 0.0
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()


def calmar_ratio(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio (annualized return / |max drawdown|)."""
    ann_ret = annualized_return(prices, periods_per_year)
    mdd = max_drawdown(prices)
    if mdd == 0:
        return 0.0
    return ann_ret / abs(mdd)


def alpha_beta(prices: pd.Series, benchmark_prices: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> dict:
    """Calculate alpha and beta relative to a benchmark."""
    ret = prices.pct_change().dropna()
    bench_ret = benchmark_prices.pct_change().dropna()

    # Align dates
    aligned = pd.concat([ret, bench_ret], axis=1, join="inner")
    aligned.columns = ["asset", "benchmark"]

    if len(aligned) < 10:
        return {"alpha": 0.0, "beta": 0.0, "information_ratio": 0.0}

    cov = aligned.cov()
    beta = cov.iloc[0, 1] / cov.iloc[1, 1] if cov.iloc[1, 1] != 0 else 0.0

    ann_ret = annualized_return(prices, periods_per_year)
    ann_bench = annualized_return(benchmark_prices, periods_per_year)
    alpha = ann_ret - (risk_free_rate + beta * (ann_bench - risk_free_rate))

    # Information ratio
    tracking_error = (aligned["asset"] - aligned["benchmark"]).std() * np.sqrt(periods_per_year)
    ir = (ann_ret - ann_bench) / tracking_error if tracking_error != 0 else 0.0

    return {"alpha": alpha, "beta": beta, "information_ratio": ir}


def compute_all_metrics(prices: pd.Series, risk_free_rate: float = 0.02,
                        benchmark_prices: Optional[pd.Series] = None) -> dict:
    """Compute all metrics for a single price series."""
    result = {
        "annualized_return": round(annualized_return(prices), 4),
        "annualized_volatility": round(annualized_volatility(prices), 4),
        "sharpe_ratio": round(sharpe_ratio(prices, risk_free_rate), 4),
        "sortino_ratio": round(sortino_ratio(prices, risk_free_rate), 4),
        "max_drawdown": round(max_drawdown(prices), 4),
        "calmar_ratio": round(calmar_ratio(prices), 4),
    }
    if benchmark_prices is not None:
        ab = alpha_beta(prices, benchmark_prices, risk_free_rate)
        result["alpha"] = round(ab["alpha"], 4)
        result["beta"] = round(ab["beta"], 4)
        result["information_ratio"] = round(ab["information_ratio"], 4)
    return result
