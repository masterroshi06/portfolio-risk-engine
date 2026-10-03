"""Multi-Asset Portfolio Risk & Market Stress-Testing Engine.

Academic and student-friendly risk management dashboard implementing
Value at Risk (VaR), Conditional VaR (Expected Shortfall), multi-horizon
scaling, and historical macro crisis replay across Indian & global assets.
"""

from __future__ import annotations

import io
import os
import urllib.request
from datetime import date
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from fpdf import FPDF

from risk_engine.data_loader import (
    DataLoadError,
    align_weights,
    fetch_fx_rate,
    load_portfolio_data,
    lookback_start,
    normalize_ticker_symbol,
    normalize_tickers,
)
from risk_engine.metrics import (
    RISK_FREE_RATE as DEFAULT_RF,
    compute_portfolio_metrics,
)
from risk_engine.stress_tester import CRISIS_SCENARIOS, run_stress_test

load_dotenv()

# --- Page Configuration (Must be first Streamlit command) ---
st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def _env_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback

def _env_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback

RISK_FREE_RATE = _env_float("RISK_FREE_RATE", DEFAULT_RF)
YFINANCE_TIMEOUT = _env_int("YFINANCE_TIMEOUT", 30)
DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "INR").strip().upper()
if DEFAULT_CURRENCY not in {"INR", "USD"}:
    DEFAULT_CURRENCY = "INR"

# --- UI Styling & Mappings ---
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22, 30, 46, 0.45)",
    font=dict(color="#E2E8F0", family="'Inter', -apple-system, sans-serif"),
    margin=dict(l=54, r=28, t=48, b=54),
)
ACCENT_ACADEMIC = "#38BDF8" 
LOSS_MUTED = "#F87171"       
WARN_MUTED = "#FBBF24"       

TICKER_READABLE_NAMES: dict[str, str] = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "TATAMOTORS.NS": "Tata Motors",
    "SBIN.NS": "State Bank of India",
    "JIOFIN.NS": "Jio Financial",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "LT.NS": "Larsen & Toubro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "HCLTECH.NS": "HCL Tech",
    "MARUTI.NS": "Maruti Suzuki",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com",
    "META": "Meta Platforms",
    "TSLA": "Tesla Inc.",
    "TLT": "US 20+ Yr Treasury ETF",
    "SPY": "S&P 500 ETF",
}

POPULAR_TICKERS = [
    "^NSEI", "^BSESN", "^GSPC", "^IXIC",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "JIOFIN.NS", "TATAMOTORS.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "TLT", "SPY"
]

DEFAULT_TICKERS = ["^NSEI", "^GSPC", "RELIANCE.NS", "AAPL"]
LOOKBACK_MAP = {"1 Year": 1, "3 Years": 3, "5 Years": 5}

def clean_name(ticker: str) -> str:
    sym = ticker.strip().upper()
    if sym in TICKER_READABLE_NAMES:
        return TICKER_READABLE_NAMES[sym]
    
    clean = sym.replace("^", "").replace(".NS", "").replace(".BO", "")
    return clean.replace("-", " ").title()

def smart_indian_suffix(tickers: list[str]) -> list[str]:
    indian_majors = {"RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "TATAMOTORS", "SBIN", "JIOFIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH"}
    processed = []
    for t in tickers:
        clean_t = t.strip().upper()
        if clean_t in indian_majors:
            processed.append(f"{clean_t}.NS")
        else:
            processed.append(clean_t)
    return processed

def _inject_css() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stHeader"] { background-color: transparent !important; }
            .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1380px; }
            .hero-badge {
                display: inline-block; color: #38BDF8; font-size: 0.8rem; letter-spacing: 0.1em;
                text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem;
                background: rgba(56, 189, 248, 0.1); padding: 4px 10px; border-radius: 4px;
                position: relative; z-index: 10;
            }
            .setup-card {
                background-color: #161E2E; border: 1px solid #263346;
                border-radius: 12px; padding: 22px 24px; margin-bottom: 1.4rem;
            }
            div[data-testid="stMetric"] {
                background-color: #161E2E; border: 1px solid #263346;
                border-radius: 10px; padding: 14px 18px;
            }
            .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #263346; margin-bottom: 1.5rem; }
            .stTabs [aria-selected="true"] { color: #38BDF8 !important; border-bottom: 3px solid #38BDF8; }
            .math-card { background: #161E2E; border: 1px solid #263346; border-radius: 10px; padding: 18px 22px; margin-bottom: 1.2rem; }
            .math-card h4 { color: #38BDF8; margin-top: 0; margin-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def format_money(value: float, currency: str) -> str:
    symbol = "₹" if currency == "INR" else "$"
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol}{abs(value):,.0f}"

def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"

def generate_pdf_report(metrics, tickers, lookback_label, currency, fx, portfolio_val) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    font_path = "Roboto-Regular.ttf"
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            pass 

    try:
        pdf.add_font("Roboto", style="", fname=font_path)
        pdf.set_font("Roboto", size=16)
        font_name = "Roboto"
    except Exception:
        pdf.set_font("Arial", 'B', 16)
        font_name = "Arial"
        
    def render_money(val):
        formatted = format_money(val, currency)
        return formatted.replace("₹", "INR ") if font_name == "Arial" else formatted

    pdf.cell(0, 10, "Portfolio Risk & Stress-Testing Audit Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font(font_name, size=12)
    pdf.cell(0, 10, f"Date: {date.today().isoformat()}", ln=True)
    pdf.set_font(font_name, size=11)
    pdf.cell(0, 8, f"Capital: {render_money(portfolio_val)}", ln=True)
    pdf.cell(0, 8, f"Confidence Level: {metrics.confidence:.0%}", ln=True)
    pdf.cell(0, 8, f"Lookback Window: {lookback_label}", ln=True)
    
    pdf.ln(5)
    pdf.set_font(font_name, size=12)
    pdf.cell(0, 10, "1-Day Tail-Risk Metrics", ln=True)
    pdf.set_font(font_name, size=11)
    pdf.cell(0, 8, f"Parametric VaR: {render_money(metrics.parametric_var_currency)} ({format_pct(metrics.parametric_var_pct)})", ln=True)
    pdf.cell(0, 8, f"Historical VaR: {render_money(metrics.historical_var_currency)} ({format_pct(metrics.historical_var_pct)})", ln=True)
    pdf.cell(0, 8, f"Expected Shortfall (CVaR): {render_money(metrics.cvar_currency)} ({format_pct(metrics.cvar_pct)})", ln=True)
    pdf.cell(0, 8, f"Annualized Volatility: {format_pct(metrics.annualized_volatility)}", ln=True)
    
    pdf.ln(5)
    pdf.set_font(font_name, size=12)
    pdf.cell(0, 10, "10-Day Horizon Risk Scaling", ln=True)
    pdf.set_font(font_name, size=11)
    pdf.cell(0, 8, f"10-Day Parametric VaR: {render_money(metrics.parametric_var_10d_currency)}", ln=True)
    pdf.cell(0, 8, f"10-Day Expected Shortfall: {render_money(metrics.cvar_10d_currency)}", ln=True)
    pdf.cell(0, 8, f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}", ln=True)

    pdf.ln(5)
    pdf.set_font(font_name, size=12)
    pdf.cell(0, 10, "Asset Allocation Weights", ln=True)
    pdf.set_font(font_name, size=11)
    for sym in tickers:
        w = metrics.weights.get(sym, 0)
        pdf.cell(0, 8, f"{clean_name(sym)} ({sym}): {w*100:.2f}%", ln=True)

    return bytes(pdf.output())

@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_live_quotes(tickers: tuple[str, ...], timeout: int = 15) -> dict[str, dict]:
    quotes: dict[str, dict] = {}
    for raw_symbol in tickers:
        symbol = normalize_ticker_symbol(raw_symbol)
        try:
            t = yf.Ticker(symbol)
            fast = t.fast_info
            last_price = getattr(fast, "last_price", None)
            if not last_price or not np.isfinite(last_price):
                last_price = getattr(fast, "regular_market_previous_close", None)
            
            cur = getattr(fast, "currency", None) or ("INR" if ".NS" in symbol or ".BO" in symbol else "USD")
            prev_close = getattr(fast, "regular_market_previous_close", None) or getattr(fast, "previous_close", None)
            
            pct_change = (last_price - prev_close) / prev_close if last_price and prev_close else None
            
            quotes[raw_symbol] = {
                "symbol": symbol,
                "price": float(last_price) if last_price else None,
                "currency": str(cur).upper(),
                "pct_change": float(pct_change) if pct_change else None,
            }
        except Exception:
            quotes[raw_symbol] = {"symbol": symbol, "price": None, "currency": "INR", "pct_change": None}
    return quotes

@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_market_data(tickers: tuple[str, ...], lookback_years: int, timeout: int) -> dict:
    data = load_portfolio_data(list(tickers), lookback_years=lookback_years, timeout=timeout)
    return {
        "prices": data.prices,
        "log_returns": data.log_returns,
        "covariance": data.covariance,
        "dropped": data.dropped_tickers,
        "warnings": data.warnings,
    }

@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fx_usd_inr(timeout: int) -> float | None:
    return fetch_fx_rate("USDINR=X", timeout=timeout)

@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_stress_test(tickers: tuple[str, ...], weights: tuple[tuple[str, float], ...], scenario_key: str, portfolio_value: float, timeout: int):
    weight_series = pd.Series({k: v for k, v in weights}, dtype=float)
    return run_stress_test(list(tickers), weight_series, CRISIS_SCENARIOS[scenario_key], portfolio_value=portfolio_value, timeout=timeout)

# --- Chart Builders ---
def build_histogram(returns: pd.Series, var_ret: float, cvar_ret: float, confidence: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns * 100, nbinsx=65, opacity=0.85, name="Daily Returns",
        marker=dict(color="rgba(56, 189, 248, 0.65)", line=dict(color=ACCENT_ACADEMIC, width=0.8)),
        hovertemplate="Return: %{x:.2f}%<br>Days: %{y}<extra></extra>"
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(255, 255, 255, 0.25)")
    fig.add_vline(x=var_ret * 100, line_dash="dash", line_color=LOSS_MUTED, line_width=2.2, annotation_text=f"VaR: {var_ret * 100:.2f}%")
    fig.add_vline(x=cvar_ret * 100, line_dash="dash", line_color=WARN_MUTED, line_width=2.2, annotation_text=f"CVaR: {cvar_ret * 100:.2f}%")
    fig.update_layout(
        **PLOTLY_LAYOUT, title="Tail-Risk Return Distribution",
        xaxis=dict(title="Daily Return (%)", tickangle=-45, automargin=True, gridcolor="#263346"),
        yaxis=dict(title="Frequency", automargin=True, gridcolor="#263346"),
        showlegend=False, height=400
    )
    return fig

def build_cumulative_chart(prices: pd.DataFrame, weights: pd.Series, baseline_value: float, currency: str) -> go.Figure:
    rets = np.log(prices / prices.shift(1)).dropna()
    w = weights.reindex(rets.columns).fillna(0.0)
    rp = rets.dot(w)
    nav = np.exp(rp.cumsum()) / np.exp(rp.cumsum()).iloc[0]
    portfolio_dollars = nav * baseline_value

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_dollars.index, y=portfolio_dollars, name="Valuation",
        line=dict(color=ACCENT_ACADEMIC, width=2.4)
    ))
    fig.add_hline(y=baseline_value, line_dash="dot", line_color="rgba(255, 255, 255, 0.35)")
    fig.update_layout(
        **PLOTLY_LAYOUT, title=f"Cumulative Performance ({currency})",
        xaxis=dict(title="Date", tickangle=-45, automargin=True, gridcolor="#263346"),
        yaxis=dict(title=f"Valuation ({currency})", automargin=True, gridcolor="#263346"),
        showlegend=False, height=400
    )
    return fig

def build_drawdown_chart(prices: pd.DataFrame, weights: pd.Series) -> go.Figure:
    rets = np.log(prices / prices.shift(1)).dropna()
    w = weights.reindex(rets.columns).fillna(0.0)
    nav = np.exp(rets.dot(w).cumsum()) / np.exp(rets.dot(w).cumsum()).iloc[0]
    dd = (nav / nav.cummax() - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", fillcolor="rgba(248, 113, 113, 0.22)", line=dict(color=LOSS_MUTED, width=1.5)))
    fig.update_layout(
        **PLOTLY_LAYOUT, title="Peak-to-Trough Drawdown (%)",
        xaxis=dict(title="Date", tickangle=-45, automargin=True, gridcolor="#263346"),
        yaxis=dict(title="Drawdown (%)", automargin=True, gridcolor="#263346"),
        showlegend=False, height=280
    )
    return fig

def build_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    renamed = matrix.rename(index=clean_name, columns=clean_name)
    fig = px.imshow(renamed, text_auto=".2f", color_continuous_scale="Tealrose", aspect="auto")
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=400, xaxis=dict(tickangle=-45, automargin=True))
    fig.update_coloraxes(colorbar_title="")
    return fig

# --- Main App Execution ---
def main() -> None:
    _inject_css()

    st.markdown('<div class="hero-badge">Academic Quantitative Risk Engine</div>', unsafe_allow_html=True)
    st.markdown('<h1>Portfolio Risk &amp; Market Stress-Testing Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8; margin-bottom:1.5rem;">Tail-risk evaluation, multi-horizon scaling, and macro crisis replay.</p>', unsafe_allow_html=True)

    tab_setup, tab_analytics = st.tabs(["⚙️ 1. Portfolio Setup", "📊 2. Risk Analytics"])

    with tab_setup:
        st.markdown('<div class="setup-card"><h3>Asset &amp; Capital Settings</h3>', unsafe_allow_html=True)
        col_left, col_right = st.columns([1.3, 0.7])
        
        with col_left:
            raw_selected = st.multiselect(
                "Select Portfolio Assets", options=POPULAR_TICKERS, default=DEFAULT_TICKERS,
                format_func=clean_name, accept_new_options=True,
                help="Search via dropdown or type custom tickers (e.g. RELIANCE, JIOFIN, AAPL). Indian stocks auto-append .NS."
            )
            
            tickers = normalize_tickers(smart_indian_suffix(raw_selected)) or list(DEFAULT_TICKERS)
            
            with st.expander("📌 Current Asset Prices (Live / Latest Close)", expanded=False):
                quotes = fetch_live_quotes(tuple(tickers), YFINANCE_TIMEOUT)
                q_cols = st.columns(min(len(tickers), 4))
                for idx, sym in enumerate(tickers):
                    q = quotes.get(sym, {})
                    p, c = q.get("price"), q.get("currency", "USD")
                    chg = q.get("pct_change")
                    sym_c = "₹" if c == "INR" else "$"
                    q_cols[idx % 4].metric(clean_name(sym), f"{sym_c}{p:,.2f}" if p else "N/A", f"{chg:+.2%}" if chg else None)

        with col_right:
            c1, c2 = st.columns(2)
            currency = c1.selectbox("Display Currency", ["INR", "USD"], index=0 if DEFAULT_CURRENCY == "INR" else 1)
            lookback_label = c2.selectbox("Lookback Window", list(LOOKBACK_MAP.keys()), index=1)
            
            c3, c4 = st.columns(2)
            curr_symbol = "₹" if currency == "INR" else "$"
            portfolio_value = c3.number_input(f"Total Capital ({curr_symbol})", min_value=1_000.0, value=1_000_000.0 if currency == "INR" else 100_000.0, step=50000.0)
            confidence = 0.95 if c4.selectbox("Confidence Level", ["95%", "99%"]) == "95%" else 0.99
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="setup-card"><h3>Asset Allocation</h3>', unsafe_allow_html=True)
        allocation_mode = st.radio("Allocation Mode", ["Percentage (%)", f"Absolute Amount ({curr_symbol})"], horizontal=True)
        
        allocation_valid, weight_series = True, pd.Series(dtype=float)
        
        if "Percentage" in allocation_mode:
            raw_weights = {}
            slider_cols = st.columns(3)
            
            for sym in tickers:
                if f"pct_{sym}" not in st.session_state:
                    st.session_state[f"pct_{sym}"] = 100.0 / len(tickers)
            
            for i, sym in enumerate(tickers):
                raw_weights[sym] = slider_cols[i % 3].slider(clean_name(sym), 0.0, 100.0, step=0.5, key=f"pct_{sym}")
            
            total_pct = sum(raw_weights.values())
            if abs(total_pct - 100.0) < 0.05:
                st.success("✅ Allocation balanced exactly to 100.0%")
                weight_series = align_weights(tickers, raw_weights)
            else:
                allocation_valid = False
                st.error(f"⚠️ Weights sum to {total_pct:.1f}%. Must equal exactly 100.0%.")
                
                def balance_callback():
                    if total_pct > 0:
                        new_weights = {s: float(round((raw_weights[s] / total_pct) * 100.0, 1)) for s in tickers}
                        remainder = round(100.0 - sum(new_weights.values()), 1)
                        if remainder != 0 and tickers:
                            new_weights[tickers[-1]] = round(new_weights[tickers[-1]] + remainder, 1)
                        for s in tickers:
                            st.session_state[f"pct_{s}"] = new_weights[s]
                            
                st.button("⚖️ Auto-Balance Sliders", on_click=balance_callback, use_container_width=True)

        else:
            raw_amounts = {}
            input_cols = st.columns(3)
            for i, sym in enumerate(tickers):
                raw_amounts[sym] = input_cols[i % 3].number_input(f"{clean_name(sym)} ({curr_symbol})", min_value=0.0, max_value=float(portfolio_value), step=1000.0, key=f"amt_{sym}")
            
            total_allocated = sum(raw_amounts.values())
            if total_allocated > portfolio_value + 1e-4:
                allocation_valid = False
                st.error(f"❌ Budget Exceeded! You have allocated {format_money(total_allocated, currency)} of a {format_money(portfolio_value, currency)} max.")
            elif total_allocated <= 0:
                allocation_valid = False
                st.error("❌ Allocation must be greater than zero.")
            else:
                st.success(f"✅ Allocated: {format_money(total_allocated, currency)} | Cash Buffer: {format_money(portfolio_value - total_allocated, currency)}")
                weight_series = pd.Series(raw_amounts) / total_allocated
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Run Risk Analysis", type="primary", use_container_width=True):
            if not allocation_valid:
                st.error("Please correct portfolio allocations above.")
            else:
                with st.spinner("Running Monte Carlo simulations..."):
                    try:
                        payload = cached_market_data(tuple(tickers), LOOKBACK_MAP[lookback_label], YFINANCE_TIMEOUT)
                        
                        # --- STRICT VALIDATION BLOCK ---
                        if payload.get("dropped"):
                            dropped_str = ", ".join(payload["dropped"])
                            st.error(f"❌ **Invalid Asset(s) Detected:** Cannot fetch data for `{dropped_str}`. Please remove or correct them in the Portfolio Setup before proceeding.")
                        else:
                            live_tickers = list(payload["prices"].columns)
                            live_weights = align_weights(live_tickers, weight_series)
                            
                            metrics = compute_portfolio_metrics(
                                payload["log_returns"], 
                                live_weights, 
                                portfolio_value=float(portfolio_value), 
                                confidence=confidence, 
                                risk_free_rate=RISK_FREE_RATE
                            )
                            st.session_state.report = {
                                "live_tickers": live_tickers, "lookback_label": lookback_label,
                                "currency": currency, "metrics": metrics, "prices": payload["prices"], "log_returns": payload["log_returns"],
                                "covariance": payload.get("covariance", payload["log_returns"].cov()), "portfolio_value": float(portfolio_value), "fx": None
                            }
                            st.toast("Analysis complete! View Tab 2.", icon="📊")
                    except Exception as e:
                        traceback.print_exc()
                        st.error("⚠️ Connection Error: Unable to fetch market data. Check your terminal for details.")

    with tab_analytics:
        if "report" not in st.session_state:
            st.info("👈 Please configure your portfolio in Tab 1 and click **Run Risk Analysis**.")
        else:
            r = st.session_state.report
            m = r["metrics"]
            val, curr, conf = r["portfolio_value"], r["currency"], f"{m.confidence:.0%}"

            st.markdown(f"<div style='background:#161E2E; padding:10px; border-radius:8px; margin-bottom:1rem; color:#CBD5E1;'><b>Capital:</b> {format_money(val, curr)} &nbsp;|&nbsp; <b>Confidence:</b> {conf} &nbsp;|&nbsp; <b>Window:</b> {r['lookback_label']}</div>", unsafe_allow_html=True)
            
            st.markdown("##### 1-Day Horizon Tail-Risk")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Valuation", format_money(val, curr))
            k2.metric(f"Parametric VaR", format_money(m.parametric_var_currency, curr), format_pct(m.parametric_var_pct), delta_color="inverse")
            k3.metric(f"Historical VaR", format_money(m.historical_var_currency, curr), format_pct(m.historical_var_pct), delta_color="inverse")
            k4.metric("CVaR (Expected Shortfall)", format_money(m.cvar_currency, curr), format_pct(m.cvar_pct), delta_color="inverse")
            k5.metric("Ann. Volatility", format_pct(m.annualized_volatility))

            st.plotly_chart(build_histogram(m.portfolio_log_returns, m.var_threshold_return, m.cvar_threshold_return, m.confidence), use_container_width=True)
            st.plotly_chart(build_cumulative_chart(r["prices"], m.weights, val, curr), use_container_width=True)
            
            st.markdown("---")
            pdf_bytes = generate_pdf_report(m, r["live_tickers"], r["lookback_label"], curr, r["fx"], val)
            st.download_button(
                label="📥 Download Risk Audit Report (PDF)",
                data=pdf_bytes,
                file_name="portfolio_risk_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            with st.expander("Advanced: Correlation Matrix & Math Methodology"):
                st.plotly_chart(build_heatmap(r["log_returns"].corr(), "Asset Correlation Matrix (ρ)"), use_container_width=True)
                st.markdown(r"""
                <div class="math-card">
                    <h4>Quantitative Formulas</h4>
                    <p><b>Parametric VaR:</b> $\text{VaR}_\alpha = -(\mu_p + z_\alpha \sigma_p)$</p>
                    <p><b>Expected Shortfall:</b> $\text{CVaR}_\alpha = -\mathbb{E}[r_{p,t} \mid r_{p,t} \le -\text{VaR}_\alpha]$</p>
                    <p><b>Square-Root-of-Time (10-Day VaR):</b> $\text{VaR}_{10\text{-Day}} = \text{VaR}_{1\text{-Day}} \times \sqrt{10}$</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()