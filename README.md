# Multi-Asset Portfolio Risk & Market Stress-Testing Engine

Interactive Streamlit dashboard implementing **1-day Value at Risk (VaR)**, **Expected Shortfall (CVaR)**, **10-day Basel horizon scaling (√10 rule)**, and **audit-ready reporting** across Indian and global assets. Built for quantitative risk evaluation, academic projects, and portfolio analysis.

---

## Features

- **Multi-asset ingestion & search:** Pulls adjusted daily closes and live pricing from Yahoo Finance (`yfinance`) with smart ticker resolution and automatic `.NS` suffixing for Indian stocks.
- **Flexible allocation modes:** Allocate via percentage sliders (auto-balanced to 100%) or exact absolute capital amounts with strict budget validation.
- **Tail-risk modeling:** Calculates Parametric (Variance-Covariance) VaR, Historical Simulation VaR, and Conditional VaR (Expected Shortfall) at 95% or 99% confidence.
- **Multi-horizon scaling:** Applies the square-root-of-time rule to scale 1-day risk metrics to 10-day regulatory holding horizons.
- **Performance analytics:** Computes annualized return, annualized volatility, and Sharpe ratio against a configurable risk-free rate.
- **Interactive visualizations:** Plotly charts for return distribution histograms with VaR cutoff lines, cumulative performance, and underwater (peak-to-trough) drawdown curves.
- **Executive audit reporting:** Exports risk audit summaries to CSV or custom-formatted PDF.

---

## Quick Start (Local)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

Open the URL Streamlit prints in your terminal (typically `http://localhost:8501`).

---

## Using the Dashboard

### Tab 1: Portfolio Setup

1. Search and select assets from major benchmarks, or type custom tickers (e.g., `RELIANCE`, `TCS`, `AAPL`). Indian stocks automatically get the `.NS` suffix.
2. Choose your display currency (INR or USD), lookback window (1, 3, or 5 years), total portfolio capital, and confidence level (95% or 99%).
3. Configure weights via **Percentage** sliders or **Absolute** amounts.
4. Click **Run Risk Analysis**.

### Tab 2: Risk Analytics

- Inspect headline KPI cards for 1-day and 10-day tail risk.
- View the interactive return distribution, cumulative valuation, and drawdown charts.
- Expand the advanced sections for the asset return correlation matrix (ρ) and the mathematical methodology.
- Download audit-ready reports as CSV or PDF.

---

## Project Layout

```text
portfolio-risk-engine/
├── app.py                  # Streamlit dashboard entrypoint
├── requirements.txt        # Python dependencies
├── .env.example            # Environment configuration template
├── .streamlit/
│   └── config.toml         # Dark theme styling
└── risk_engine/
    ├── data_loader.py      # Data ingestion, normalization, and FX rates
    ├── metrics.py          # VaR, CVaR, Sharpe, and horizon scaling calculations
    └── stress_tester.py    # Historical crisis replay utilities
```

---

## Risk Formulas

**Parametric VaR (Variance-Covariance)**

$$\text{VaR}_\alpha = -(\mu_p + z_\alpha \cdot \sigma_p)$$

where $z_\alpha$ is the left-tail quantile of the standard normal distribution.

**Historical Simulation VaR**

Non-parametric empirical quantile of realized portfolio log returns.

**Expected Shortfall (CVaR)**

Conditional expectation of losses beyond the VaR threshold:

$$\text{CVaR}_\alpha = -\mathbb{E}\left[r_{p,t} \mid r_{p,t} \le -\text{VaR}_\alpha\right]$$

**10-Day Horizon Scaling**

$$\text{VaR}_{10\text{-Day}} = \text{VaR}_{1\text{-Day}} \times \sqrt{10}$$

---

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, set the **Main file path** to `app.py`, and click **Deploy**.

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. You may use, modify, and share this software freely under the terms of the GPLv3. See the [LICENSE](LICENSE) file for details.