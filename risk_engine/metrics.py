"""Core portfolio risk math: VaR, Expected Shortfall, volatility, Sharpe.

Notation used throughout
------------------------
- r_p,t  : portfolio log return on day t (weighted sum of asset log returns)
- μ_p    : sample mean of r_p
- σ_p    : sample standard deviation of r_p
- α      : confidence level, e.g. 0.95 or 0.99
- z_α    : standard-normal quantile of the *left tail* (1 − α)
           For 95% this is Φ^{-1}(0.05) ≈ −1.645

Value at Risk is reported as a **positive loss** (both in return space and
in currency). A 1-day 95% VaR of 2% on a $100,000 book means: historically /
under normality we would not expect to lose more than $2,000 on a typical
day, 95% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from risk_engine.data_loader import TRADING_DAYS_PER_YEAR, align_weights

# Assumed annual risk-free rate used in the Sharpe ratio (2.0%).
RISK_FREE_RATE = 0.02


@dataclass(frozen=True)
class PortfolioMetrics:
    """All headline risk and performance statistics for the current book."""

    weights: pd.Series
    portfolio_log_returns: pd.Series
    mean_daily_return: float
    daily_volatility: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    confidence: float
    parametric_var_pct: float
    historical_var_pct: float
    cvar_pct: float
    parametric_var_currency: float
    historical_var_currency: float
    cvar_currency: float
    portfolio_value: float

    @property
    def var_threshold_return(self) -> float:
        """Signed daily return that corresponds to Historical VaR (usually negative)."""
        return -self.historical_var_pct

    @property
    def cvar_threshold_return(self) -> float:
        """Signed daily return equal to −CVaR (mean of the tail)."""
        return -self.cvar_pct

    @property
    def parametric_var_10d_pct(self) -> float:
        """10-day Parametric VaR scaled by sqrt(10)."""
        return float(self.parametric_var_pct * np.sqrt(10.0))

    @property
    def historical_var_10d_pct(self) -> float:
        """10-day Historical VaR scaled by sqrt(10)."""
        return float(self.historical_var_pct * np.sqrt(10.0))

    @property
    def cvar_10d_pct(self) -> float:
        """10-day CVaR scaled by sqrt(10)."""
        return float(self.cvar_pct * np.sqrt(10.0))

    @property
    def parametric_var_10d_currency(self) -> float:
        """10-day Parametric VaR in currency units."""
        return float(self.parametric_var_currency * np.sqrt(10.0))

    @property
    def historical_var_10d_currency(self) -> float:
        """10-day Historical VaR in currency units."""
        return float(self.historical_var_currency * np.sqrt(10.0))

    @property
    def cvar_10d_currency(self) -> float:
        """10-day CVaR in currency units."""
        return float(self.cvar_currency * np.sqrt(10.0))


def scale_var_horizon(var_metric: float, days: float = 10.0) -> float:
    """Scale a 1-day risk metric to a multi-day horizon using the square-root-of-time rule: VaR_T = VaR_1d * sqrt(T)."""
    return float(var_metric * np.sqrt(days))


def portfolio_log_returns(asset_log_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Build a single portfolio return series: r_p,t = Σ_i w_i r_i,t.

    Using a weighted sum of *log* returns is the usual daily approximation
    for a constantly rebalanced book. Weights must already sum to 1.
    """
    aligned = weights.reindex(asset_log_returns.columns).fillna(0.0).astype(float)
    return asset_log_returns.dot(aligned)


def parametric_var(mean_daily: float, daily_vol: float, confidence: float) -> float:
    """Variance-covariance (normal) 1-day VaR as a positive loss *return*.

    Formula
        VaR_α = −(μ_p + z_α · σ_p)

    where z_α = Φ^{-1}(1 − α) is negative, so the expression yields a
    positive number whenever the left-tail loss exceeds the mean.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError("Confidence level must be between 0.5 and 1.0, e.g. 0.95.")
    # Left-tail z-score, e.g. 95% → −1.64485, 99% → −2.32635
    z_alpha = float(norm.ppf(1.0 - confidence))
    var_loss = -(mean_daily + z_alpha * daily_vol)
    return float(max(var_loss, 0.0))


def historical_var(portfolio_returns: pd.Series, confidence: float) -> float:
    """Historical-simulation 1-day VaR as a positive loss *return*.

    Rank-order the actual daily portfolio returns and read the
    (1 − α) empirical percentile. Example: at 95% confidence we take the
    5th percentile. If that percentile is −1.8%, Historical VaR is 1.8%.
    """
    if portfolio_returns.empty:
        raise ValueError("Cannot compute Historical VaR without a return series.")
    tail_probability = 1.0 - confidence
    # Linear interpolation between nearby order statistics.
    quantile = float(np.quantile(portfolio_returns.to_numpy(dtype=float), tail_probability, method="linear"))
    return float(max(-quantile, 0.0))


def expected_shortfall(portfolio_returns: pd.Series, confidence: float) -> float:
    """Conditional VaR (Expected Shortfall) as a positive loss *return*.

    CVaR_α is the average of all daily portfolio returns that are *worse*
    than the Historical VaR threshold (the left-tail mean), then flipped
    to a positive loss. If no observations sit strictly beyond the
    threshold (tiny samples), we include returns at the VaR quantile.
    """
    if portfolio_returns.empty:
        raise ValueError("Cannot compute CVaR without a return series.")
    var_loss = historical_var(portfolio_returns, confidence)
    threshold = -var_loss  # signed return, typically negative
    tail = portfolio_returns[portfolio_returns <= threshold]
    if tail.empty:
        tail = portfolio_returns[portfolio_returns <= portfolio_returns.quantile(1.0 - confidence)]
    if tail.empty:
        return var_loss
    return float(max(-float(tail.mean()), 0.0))


def annualized_return(mean_daily: float) -> float:
    """Scale the mean daily log return to a 252-trading-day year: μ × 252."""
    return float(mean_daily * TRADING_DAYS_PER_YEAR)


def annualized_volatility(daily_vol: float) -> float:
    """Scale daily σ to annual: σ × √252."""
    return float(daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(ann_return: float, ann_vol: float, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """(Portfolio excess return) / (annualized volatility).

    Higher Sharpe means more return earned per unit of risk. A zero
    volatility book is defined as Sharpe = 0 to avoid division by zero.
    """
    if ann_vol <= 0 or not np.isfinite(ann_vol):
        return 0.0
    return float((ann_return - risk_free_rate) / ann_vol)


def to_currency(loss_return: float, portfolio_value: float) -> float:
    """Convert a fractional loss (e.g. 0.02) into currency units."""
    return float(loss_return * portfolio_value)


def compute_portfolio_metrics(
    asset_log_returns: pd.DataFrame,
    weights: pd.Series | dict[str, float] | list[float],
    *,
    portfolio_value: float,
    confidence: float = 0.95,
    risk_free_rate: float = RISK_FREE_RATE,
) -> PortfolioMetrics:
    """Run the full 1-day risk pack on a cleaned log-return panel."""
    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be a positive number.")
    w = align_weights(list(asset_log_returns.columns), weights)
    r_p = portfolio_log_returns(asset_log_returns, w)
    mu = float(r_p.mean())
    sigma = float(r_p.std(ddof=1))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Portfolio volatility could not be estimated from the return window.")

    pvar = parametric_var(mu, sigma, confidence)
    hvar = historical_var(r_p, confidence)
    cvar = expected_shortfall(r_p, confidence)
    ann_ret = annualized_return(mu)
    ann_vol = annualized_volatility(sigma)
    sharpe = sharpe_ratio(ann_ret, ann_vol, risk_free_rate)

    return PortfolioMetrics(
        weights=w,
        portfolio_log_returns=r_p,
        mean_daily_return=mu,
        daily_volatility=sigma,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        confidence=confidence,
        parametric_var_pct=pvar,
        historical_var_pct=hvar,
        cvar_pct=cvar,
        parametric_var_currency=to_currency(pvar, portfolio_value),
        historical_var_currency=to_currency(hvar, portfolio_value),
        cvar_currency=to_currency(cvar, portfolio_value),
        portfolio_value=float(portfolio_value),
    )
