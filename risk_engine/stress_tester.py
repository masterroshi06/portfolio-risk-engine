"""Historical macro stress tests on the current portfolio mix.

Each scenario replays a well-known crisis window. We download (or reuse)
adjusted closes for that exact calendar range, apply the user's *current*
weights as a constant-mix book, and report:

1. Cumulative portfolio return over the window (the full path P&L).
2. Maximum peak-to-trough drawdown along that path.
3. Absolute currency loss vs. the starting portfolio value.

Assets that did not trade in a given window (IPO later, delisted, etc.)
are skipped and remaining weights are renormalized so the test still runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from risk_engine.data_loader import (
    DataLoadError,
    PortfolioData,
    align_weights,
    load_portfolio_data,
)
from risk_engine.metrics import portfolio_log_returns


@dataclass(frozen=True)
class CrisisScenario:
    key: str
    name: str
    start: date
    end: date
    description: str


# Pre-configured windows from the product spec.
CRISIS_SCENARIOS: dict[str, CrisisScenario] = {
    "covid_2020": CrisisScenario(
        key="covid_2020",
        name="2020 COVID Market Crash",
        start=date(2020, 2, 19),
        end=date(2020, 3, 23),
        description=(
            "Global risk-off as COVID-19 lockdowns began: equities sold off violently "
            "while high-quality duration (e.g. Treasuries) typically bid up."
        ),
    ),
    "tech_2022": CrisisScenario(
        key="tech_2022",
        name="2022 Tech Sell-Off & Inflation Shock",
        start=date(2022, 1, 3),
        end=date(2022, 12, 28),
        description=(
            "Aggressive Fed hiking cycle, inflation spike, and a deep drawdown in "
            "long-duration growth / technology names."
        ),
    ),
    "gfc_2008": CrisisScenario(
        key="gfc_2008",
        name="2008 Global Financial Crisis",
        start=date(2007, 10, 9),
        end=date(2009, 3, 9),
        description=(
            "Banking-system collapse through the Lehman weekend into the March 2009 "
            "equity trough — the deepest post-war drawdown for most risk assets."
        ),
    ),
}


@dataclass(frozen=True)
class StressResult:
    scenario: CrisisScenario
    weights_used: pd.Series
    skipped_tickers: tuple[str, ...]
    cumulative_return: float
    max_drawdown: float
    currency_pnl: float
    currency_max_drawdown: float
    nav_path: pd.Series
    asset_total_returns: pd.Series
    warnings: tuple[str, ...]


def _max_drawdown(nav: pd.Series) -> float:
    """Largest peak-to-trough decline of a NAV series (negative number)."""
    peaks = nav.cummax()
    dd = nav / peaks - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def run_stress_test(
    tickers: list[str],
    weights: pd.Series | dict[str, float],
    scenario: CrisisScenario,
    *,
    portfolio_value: float,
    preloaded: PortfolioData | None = None,
    timeout: int = 30,
) -> StressResult:
    """Replay one crisis window for the current ticker/weight mix."""
    data, warnings = _load_window(tickers, scenario, preloaded=preloaded, timeout=timeout)

    available = list(data.prices.columns)
    skipped = tuple(t for t in tickers if t not in available)
    w = align_weights(available, weights)

    # Daily constant-mix P&L from log returns, then exponentiate to a NAV path.
    # NAV starts at 1.0 on the first close; each subsequent step is exp(r_p).
    r_p = portfolio_log_returns(data.log_returns, w)
    if r_p.empty:
        raise DataLoadError(f"No overlapping returns inside '{scenario.name}'.")
    nav_index = [data.prices.index[0], *list(r_p.index)]
    nav_values = np.concatenate([[1.0], np.exp(np.cumsum(r_p.to_numpy(dtype=float)))])
    nav = pd.Series(nav_values, index=pd.DatetimeIndex(nav_index), name="nav")
    nav = nav / float(nav.iloc[0])

    cumulative = float(nav.iloc[-1] - 1.0)
    mdd = _max_drawdown(nav)

    first = data.prices.iloc[0]
    last = data.prices.iloc[-1]
    asset_total = (last / first) - 1.0
    asset_total = asset_total.reindex(w.index).astype(float)

    extra = list(data.warnings)
    if skipped:
        extra.append(
            "No (or insufficient) prices in this window for: "
            + ", ".join(skipped)
            + ". Remaining weights were renormalized to 100%."
        )

    return StressResult(
        scenario=scenario,
        weights_used=w,
        skipped_tickers=skipped,
        cumulative_return=cumulative,
        max_drawdown=mdd,
        currency_pnl=float(cumulative * portfolio_value),
        currency_max_drawdown=float(mdd * portfolio_value),
        nav_path=nav,
        asset_total_returns=asset_total,
        warnings=tuple(extra),
    )


def run_all_stress_tests(
    tickers: list[str],
    weights: pd.Series | dict[str, float],
    *,
    portfolio_value: float,
    timeout: int = 30,
) -> dict[str, StressResult | Exception]:
    """Run every pre-configured scenario; capture per-scenario failures."""
    results: dict[str, StressResult | Exception] = {}
    for key, scenario in CRISIS_SCENARIOS.items():
        try:
            results[key] = run_stress_test(
                tickers,
                weights,
                scenario,
                portfolio_value=portfolio_value,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 — dashboard renders the error per card
            results[key] = exc
    return results


def _load_window(
    tickers: list[str],
    scenario: CrisisScenario,
    *,
    preloaded: PortfolioData | None,
    timeout: int,
) -> tuple[PortfolioData, list[str]]:
    """Prefer a dedicated download for the crisis calendar, not the UI lookback.

    A 1-year lookback never contains 2008, so stress tests always fetch their
    own window. ``preloaded`` is unused for now but kept for API stability.
    """
    _ = preloaded
    data = load_portfolio_data(
        tickers,
        start=scenario.start,
        end=scenario.end,
        timeout=timeout,
        min_observations=5,
    )
    return data, []
