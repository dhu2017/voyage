"""Portfolio optimization engine using Riskfolio-Lib."""

import pandas as pd
import numpy as np
from typing import Optional


def _returns_from_prices(prices_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a returns DataFrame from a dict of price series (aligned by date)."""
    prices_df = pd.DataFrame(prices_dict)
    prices_df = prices_df.dropna()
    returns = prices_df.pct_change().dropna()
    return returns


def equal_weight(assets: list[str]) -> dict[str, float]:
    """Simple equal-weight allocation."""
    w = 1.0 / len(assets)
    return {a: round(w, 6) for a in assets}


def risk_parity(prices_dict: dict[str, pd.Series],
                constraints: Optional[dict] = None) -> dict[str, float]:
    """Risk Parity optimization via Riskfolio-Lib."""
    import riskfolio as rp
    returns = _returns_from_prices(prices_dict)
    assets = list(returns.columns)

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")

    w = port.rp_optimization(
        model="Classic",
        rm="MV",
        hist=True,
        rf=0.02 / 252,
        b=None,
    )

    if w is None:
        raise ValueError("Risk parity optimization failed to converge")

    weights = {assets[i]: round(float(w.iloc[i, 0]), 6) for i in range(len(assets))}

    if constraints:
        weights = _apply_constraints(weights, constraints)

    return weights


def mean_variance(prices_dict: dict[str, pd.Series],
                  risk_free_rate: float = 0.02,
                  constraints: Optional[dict] = None) -> dict[str, float]:
    """Mean-Variance Optimization (max Sharpe) via Riskfolio-Lib."""
    import riskfolio as rp
    returns = _returns_from_prices(prices_dict)
    assets = list(returns.columns)

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")

    w = port.optimization(
        model="Classic",
        rm="MV",
        obj="Sharpe",
        hist=True,
        rf=risk_free_rate / 252,
        l=0,
    )

    if w is None:
        raise ValueError("MVO optimization failed to converge")

    weights = {assets[i]: round(float(w.iloc[i, 0]), 6) for i in range(len(assets))}

    if constraints:
        weights = _apply_constraints(weights, constraints)

    return weights


def hierarchical_risk_parity(prices_dict: dict[str, pd.Series],
                              constraints: Optional[dict] = None) -> dict[str, float]:
    """Hierarchical Risk Parity via Riskfolio-Lib."""
    import riskfolio as rp
    returns = _returns_from_prices(prices_dict)
    assets = list(returns.columns)

    port = rp.HCPortfolio(returns=returns)

    w = port.optimization(
        model="HRP",
        rm="MV",
        rf=0.02 / 252,
    )

    if w is None:
        raise ValueError("HRP optimization failed to converge")

    weights = {assets[i]: round(float(w.iloc[i, 0]), 6) for i in range(len(assets))}

    if constraints:
        weights = _apply_constraints(weights, constraints)

    return weights


def _apply_constraints(weights: dict[str, float], constraints: dict) -> dict[str, float]:
    """Apply min/max weight constraints with iterative clipping + renormalization.

    Freezes assets that hit a bound, then redistributes the remaining
    weight among unfrozen assets until all constraints are satisfied.
    """
    min_w = constraints.get("min_weight", 0.0)
    max_w = constraints.get("max_weight", 1.0)

    w = dict(weights)
    frozen: dict[str, float] = {}

    for _ in range(len(w) * 2):
        free_keys = [k for k in w if k not in frozen]
        if not free_keys:
            break

        frozen_sum = sum(frozen.values())
        free_target = 1.0 - frozen_sum
        free_sum = sum(w[k] for k in free_keys)

        if free_sum > 0:
            scale = free_target / free_sum
            for k in free_keys:
                w[k] = w[k] * scale

        changed = False
        for k in list(free_keys):
            if w[k] < min_w:
                w[k] = min_w
                frozen[k] = min_w
                changed = True
            elif w[k] > max_w:
                w[k] = max_w
                frozen[k] = max_w
                changed = True

        if not changed:
            break

    return {k: round(v, 6) for k, v in w.items()}


def portfolio_expected_stats(prices_dict: dict[str, pd.Series],
                              weights: dict[str, float],
                              risk_free_rate: float = 0.02) -> dict:
    """Calculate expected return, volatility, Sharpe for a weighted portfolio."""
    returns = _returns_from_prices(prices_dict)
    w = np.array([weights[col] for col in returns.columns])

    mu = returns.mean().values * 252
    cov = returns.cov().values * 252

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    port_sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return {
        "expected_return": round(port_ret, 4),
        "expected_volatility": round(port_vol, 4),
        "expected_sharpe": round(port_sharpe, 4),
    }
