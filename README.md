# Multi-Asset Portfolio Risk & Market Stress-Testing Engine

Interactive Streamlit app for **1-day Value at Risk (VaR)**, **Expected Shortfall (CVaR)**, and **historical crisis replay** on a multi-asset book (equities and ETFs). Built for portfolio managers and risk analysts; deploys on [Streamlit Community Cloud](https://streamlit.io/cloud) from this repository root.

## What it does

- Pulls adjusted daily closes from Yahoo Finance (`yfinance`)
- Builds logarithmic returns \( r_t = \ln(P_t / P_{t-1}) \)
- Estimates **parametric VaR**, **historical VaR**, and **CVaR** at 95% or 99%
- Reports annualized return, volatility (\(\sigma \sqrt{252}\)), and Sharpe (2% risk-free rate)
- Replays **COVID 2020**, **2022 inflation / tech sell-off**, and **GFC 2008** with the current weights
- Exports a one-page CSV risk summary

VaR is explained in the UI as: *the maximum expected loss over a 1-day period with X% confidence.*  
CVaR is: *the average loss expected on the absolute worst-case days beyond the VaR threshold.*

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # optional
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). No JSON files or CLI flags are required to use the dashboard.

## Using the dashboard

1. Enter Yahoo tickers (default `AAPL, MSFT, NVDA, TLT`).
2. Move the weight sliders — they are renormalized to 100% automatically.
3. Set notional (default **$100,000**), lookback (1 / 3 / 5 years), and confidence (95% / 99%).
4. Click **Run analysis**.
5. Inspect KPI cards, the return histogram (VaR / CVaR lines), the correlation heatmap, and the NAV / drawdown path.
6. Pick a crisis in the stress-test module for cumulative loss and a per-asset breakdown.
7. Download **portfolio_risk_summary.csv**.

Display currency can be USD or INR. INR monetary figures use the latest `USDINR=X` spot when Yahoo returns it.

## Project layout

```
portfolio-risk-engine/
├── app.py                      # Streamlit dashboard (Cloud entrypoint)
├── requirements.txt
├── .env.example
├── .streamlit/config.toml      # Dark theme
└── risk_engine/
    ├── data_loader.py          # Prices, log returns, covariance
    ├── metrics.py              # VaR, CVaR, Sharpe
    └── stress_tester.py        # Historical crisis windows
```

Imports are package-relative (`from risk_engine...`) so they work locally and on Streamlit Cloud as long as `app.py` stays at the repo root.

## Risk formulas (short)

Parametric (variance-covariance) 1-day VaR, reported as a **positive loss**:

\[
\mathrm{VaR}_\alpha = -(\mu_p + z_\alpha \cdot \sigma_p)
\]

where \( z_\alpha = \Phi^{-1}(1-\alpha) \) is the left-tail normal quantile.

Historical VaR is the \((1-\alpha)\) empirical percentile of realized portfolio log returns (sign-flipped to a loss). CVaR is the mean of observations at or below that threshold.

## Configuration

Optional keys (see `.env.example`):

| Key | Meaning | Default |
| --- | --- | --- |
| `YFINANCE_TIMEOUT` | Download timeout (seconds) | `30` |
| `DEFAULT_CURRENCY` | Initial sidebar currency | `USD` |
| `RISK_FREE_RATE` | Sharpe risk-free rate (decimal) | `0.02` |

Never commit a real `.env`. Streamlit Cloud secrets can hold the same names.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. At [share.streamlit.io](https://share.streamlit.io), **New app** → this repository.
3. Main file: `app.py` (repo root). Python 3.10+ (3.11 or 3.12 recommended).
4. Deploy. Yahoo Finance must be reachable from Cloud; if a ticker is missing, the app skips it and renormalizes weights.

## Disclaimer

This is a research / education tool, not investment advice and not a regulatory VaR model (no liquidity adjustment, no overlapping-window statistics, no stressed VaR backtesting suite). Crisis results assume a **constant-mix** book and will skip names that did not trade in that window.

## License

Use and modify freely for coursework, demos, and internal research.
