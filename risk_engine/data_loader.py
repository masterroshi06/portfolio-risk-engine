"""Fetch and clean multi-asset daily price and return series.

All prices are pulled from Yahoo Finance via ``yfinance``. Daily risk
calculations in this project use **logarithmic returns**:

    r_t = ln(P_t / P_{t-1})

Log returns are additive over time and are the standard input for
parametric (variance-covariance) VaR.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252
# Drop a ticker if more than this share of its Close series is missing.
MAX_MISSING_SHARE = 0.30


class DataLoadError(ValueError):
    """Raised when market data cannot be loaded or is unusable."""


@dataclass(frozen=True)
class PortfolioData:
    """Cleaned market history for a set of portfolio holdings."""

    prices: pd.DataFrame
    log_returns: pd.DataFrame
    mean_daily_returns: pd.Series
    covariance: pd.DataFrame
    annualized_volatility: pd.Series
    dropped_tickers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)


def _as_date(value: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def lookback_start(lookback_years: int, end: date | None = None) -> date:
    """Return a calendar start date for a 1/3/5-year lookback window."""
    end = end or date.today()
    return end - timedelta(days=int(lookback_years * 365.25) + 7)


COMMON_US_TICKERS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "BRK.B", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "WMT", "PG", "MA", "LLY", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "ABT", "DHR",
    "LIN", "DIS", "NFLX", "ADBE", "NKE", "TXN", "PM", "AMD", "VZ", "BMY", "CMCSA",
    "INTC", "QCOM", "WFC", "HON", "COP", "UPS", "AMAT", "IBM", "BA", "GE", "CAT",
    "SPY", "QQQ", "IWM", "TLT", "EEM", "VTI", "VOO", "GLD", "SLV", "USO", "UNG",
    "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLY", "VNQ"
}

INDIAN_STOCKS_SET = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "TATAMOTORS", "JIOFIN",
    "BHARTIARTL", "ITC", "LT", "KOTAKBANK", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "BAJAJFINSV", "WIPRO", "ADANIENT", "ADANIPORTS",
    "POWERGRID", "NTPC", "ONGC", "COALINDIA", "TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM",
    "TECHM", "DIVISLAB", "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "BRITANNIA", "NESTLEIND",
    "ULTRACEMCO", "APOLLOHOSP", "INDUSINDBK", "BPCL", "IOC", "VEDL", "ZOMATO", "PAYTM",
    "NYKAA", "DMART", "HAL", "BEL", "BHEL", "IRCTC", "PFC", "RECLTD", "YESBANK", "SUZLON",
    "IDEA", "TRENT", "IRFC", "RVNL", "SJVN", "NHPC", "MAZDOCK", "COCHINSHIP"
}


def normalize_ticker_symbol(raw: str) -> str:
    """Normalize a raw ticker string: deduplicate, strip, uppercase, and append .NS for Indian stocks without suffix."""
    sym = str(raw).strip().upper()
    if not sym:
        return ""
    # Indices, currencies, or symbols already bearing an exchange suffix (.NS, .BO, etc.)
    if sym.startswith("^") or "." in sym or "=" in sym or "-" in sym:
        return sym
    # If explicitly in Indian set, or not in common US tickers, append .NS
    if sym in INDIAN_STOCKS_SET or sym not in COMMON_US_TICKERS:
        return f"{sym}.NS"
    return sym


def normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Deduplicate, uppercase, and auto-suffix ticker symbols entered by a user."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        symbol = normalize_ticker_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned


def _extract_close_prices(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Handle both single-ticker (flat columns) and multi-ticker (MultiIndex) downloads."""
    if raw is None or raw.empty:
        raise DataLoadError(
            "No price data was returned. Check ticker symbols and the lookback window."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        if "Close" in level0:
            prices = raw["Close"].copy()
        elif "Adj Close" in level0:
            prices = raw["Adj Close"].copy()
        elif "Close" in level1:
            prices = raw.xs("Close", axis=1, level=1).copy()
        elif "Adj Close" in level1:
            prices = raw.xs("Adj Close", axis=1, level=1).copy()
        else:
            raise DataLoadError("Yahoo Finance response did not include Close prices.")
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close"
        if close_col not in raw.columns:
            raise DataLoadError("Yahoo Finance response did not include Close prices.")
        label = tickers[0] if len(tickers) == 1 else "ASSET"
        prices = raw[[close_col]].rename(columns={close_col: label})

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    # Keep only requested tickers that actually arrived.
    keep = [t for t in tickers if t in prices.columns]
    extra = [c for c in prices.columns if c not in keep]
    if extra and not keep:
        keep = list(prices.columns)
    prices = prices.loc[:, keep]
    return prices.astype(float)


def _clean_prices(
    prices: pd.DataFrame,
    requested: Sequence[str],
    *,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Drop dead tickers, fill short gaps, and align a common trading calendar."""
    warnings: list[str] = []
    missing_requested = [t for t in requested if t not in prices.columns]
    dropped: list[str] = list(missing_requested)
    if missing_requested:
        warnings.append(
            "No data returned for: " + ", ".join(missing_requested) + ". Those tickers were skipped."
        )

    usable: list[str] = []
    for ticker in prices.columns:
        series = prices[ticker]
        missing_share = float(series.isna().mean())
        if series.dropna().empty or missing_share > MAX_MISSING_SHARE:
            dropped.append(str(ticker))
            warnings.append(
                f"{ticker} was dropped because more than {MAX_MISSING_SHARE:.0%} of daily prices are missing."
            )
            continue
        usable.append(str(ticker))

    if not usable:
        raise DataLoadError(
            "None of the requested tickers produced usable daily prices. "
            "Verify symbols (e.g. AAPL, MSFT) and try a shorter lookback."
        )

    cleaned = prices[usable].copy()
    # Fill isolated holidays / halted sessions, then require a complete cross-section.
    cleaned = cleaned.ffill(limit=5)
    before = len(cleaned)
    cleaned = cleaned.dropna(how="any")
    dropped_rows = before - len(cleaned)
    if dropped_rows:
        warnings.append(
            f"Dropped {dropped_rows} trading day(s) with incomplete cross-asset prices "
            "so the covariance matrix stays well-defined."
        )

    if len(cleaned) < min_observations:
        raise DataLoadError(
            f"Only {len(cleaned)} overlapping trading days remain after cleaning. "
            f"Need at least {min_observations} days. Try fewer tickers or a longer window."
        )

    return cleaned, tuple(dict.fromkeys(dropped)), tuple(warnings)


def _download_prices(symbols: Sequence[str], start: str, end: str, timeout: int) -> pd.DataFrame:
    """Call yfinance with kwargs tolerated across recent package versions."""
    kwargs: dict = dict(
        tickers=list(symbols),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    try:
        return yf.download(**kwargs, threads=True, timeout=timeout)
    except TypeError:
        return yf.download(**kwargs)


def load_portfolio_data(
    tickers: Sequence[str],
    *,
    lookback_years: int = 3,
    start: date | str | None = None,
    end: date | str | None = None,
    timeout: int = 30,
    min_observations: int = 30,
) -> PortfolioData:
    """Download adjusted closes and compute log returns plus covariance.

    Parameters
    ----------
    tickers:
        Yahoo Finance symbols (equities, ETFs, some indices).
    lookback_years:
        Used when ``start`` is omitted (1, 3, or 5 years typical).
    start, end:
        Optional explicit calendar bounds. ``end`` defaults to today.
    timeout:
        Per-request timeout passed through to yfinance.
    """
    symbols = normalize_tickers(tickers)
    if not symbols:
        raise DataLoadError("Enter at least one ticker symbol.")

    end_date = _as_date(end) if end else date.today()
    start_date = _as_date(start) if start else lookback_start(lookback_years, end_date)
    if start_date >= end_date:
        raise DataLoadError("Start date must be before end date.")

    try:
        raw = _download_prices(
            symbols,
            start_date.isoformat(),
            (end_date + timedelta(days=1)).isoformat(),
            timeout,
        )
    except Exception as exc:  # noqa: BLE001 — fallback to per-ticker fetch if batch fails
        frames = {}
        for s in symbols:
            try:
                single = _download_prices(
                    [s],
                    start_date.isoformat(),
                    (end_date + timedelta(days=1)).isoformat(),
                    timeout,
                )
                if single is not None and not single.empty:
                    frames[s] = single
            except Exception:
                pass
        if not frames:
            raise DataLoadError(f"Market data download failed for all symbols: {exc}") from exc
        raw = pd.concat(frames.values(), axis=1)

    prices = _extract_close_prices(raw, symbols)
    prices, dropped, warnings = _clean_prices(prices, symbols, min_observations=min_observations)

    # r_t = ln(P_t / P_{t-1}); first observation is undefined and dropped.
    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise DataLoadError("Could not compute daily log returns from the downloaded prices.")

    mean_daily = log_returns.mean()
    covariance = log_returns.cov(ddof=1)
    annualized_vol = log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return PortfolioData(
        prices=prices,
        log_returns=log_returns,
        mean_daily_returns=mean_daily,
        covariance=covariance,
        annualized_volatility=annualized_vol,
        dropped_tickers=dropped,
        warnings=warnings,
    )


def fetch_fx_rate(pair: str = "USDINR=X", timeout: int = 20) -> float | None:
    """Return the latest FX rate for currency conversion, or None if unavailable."""
    try:
        raw = yf.download(
            tickers=pair,
            period="5d",
            auto_adjust=True,
            progress=False,
            timeout=timeout,
        )
        if raw is None or raw.empty:
            return None
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        value = float(close.dropna().iloc[-1])
        return value if np.isfinite(value) and value > 0 else None
    except Exception:  # noqa: BLE001
        return None


def align_weights(tickers: Sequence[str], weights: Sequence[float] | dict[str, float]) -> pd.Series:
    """Map user weights onto the tickers that actually loaded, then renormalize to 1.0."""
    if isinstance(weights, dict):
        series = pd.Series({str(k).upper(): float(v) for k, v in weights.items()}, dtype=float)
    else:
        if len(weights) != len(tickers):
            raise DataLoadError("Weight vector length must match the number of loaded tickers.")
        series = pd.Series(np.asarray(weights, dtype=float), index=list(tickers))

    series = series.reindex(list(tickers)).fillna(0.0)
    series = series.clip(lower=0.0)
    total = float(series.sum())
    if total <= 0:
        series = pd.Series(1.0 / len(tickers), index=list(tickers))
    else:
        series = series / total
    return series
