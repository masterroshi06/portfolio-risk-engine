"""Multi-Asset Portfolio Risk & Market Stress-Testing Engine.

Public API for data ingestion, VaR/CVaR metrics, and historical stress tests.
"""

from risk_engine.data_loader import (
    PortfolioData,
    fetch_fx_rate,
    load_portfolio_data,
)
from risk_engine.metrics import (
    RISK_FREE_RATE,
    PortfolioMetrics,
    compute_portfolio_metrics,
    scale_var_horizon,
)
from risk_engine.stress_tester import (
    CRISIS_SCENARIOS,
    StressResult,
    run_stress_test,
    run_all_stress_tests,
)

__all__ = [
    "PortfolioData",
    "PortfolioMetrics",
    "StressResult",
    "CRISIS_SCENARIOS",
    "RISK_FREE_RATE",
    "fetch_fx_rate",
    "load_portfolio_data",
    "compute_portfolio_metrics",
    "scale_var_horizon",
    "run_stress_test",
    "run_all_stress_tests",
]
