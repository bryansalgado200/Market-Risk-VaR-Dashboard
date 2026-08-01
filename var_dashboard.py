"""
Interactive Portfolio VaR Dashboard
Run with: streamlit run var_dashboard.py

Lets you change tickers and weights live and see VaR recalculate.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

st.set_page_config(page_title="Portfolio VaR Dashboard", layout="wide")
st.title("Interactive Portfolio VaR Dashboard")
st.caption("Adjust your holdings and weights on the left, results update automatically.")

# ---------------------------------------------------------------------------
# SIDEBAR — user inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Portfolio Setup")

default_tickers = "VTI, VXUS, SCHD, XLF, BND, JQUA, KO, JPM, SPY"
ticker_input = st.sidebar.text_area(
    "Tickers (comma-separated)", value=default_tickers, height=80
)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

st.sidebar.subheader("Weights (%)")
st.sidebar.caption("Should add up to 100 — the app will normalize automatically if not.")

weights = {}
default_weight = round(100 / len(tickers), 1) if tickers else 0
for ticker in tickers:
    weights[ticker] = st.sidebar.slider(
        ticker, min_value=0.0, max_value=100.0, value=default_weight, step=0.5
    )

total_weight = sum(weights.values())
if total_weight == 0:
    st.sidebar.error("Set at least one weight above 0.")
    st.stop()
normalized_weights = {t: w / total_weight for t, w in weights.items()}

st.sidebar.subheader("Model Settings")
lookback_years = st.sidebar.slider("Lookback period (years)", 1, 10, 5)
confidence = st.sidebar.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)
portfolio_value = st.sidebar.number_input(
    "Assumed portfolio value ($)", min_value=1000, value=100_000, step=1000
)

# ---------------------------------------------------------------------------
# DATA — cached so re-running with same tickers doesn't re-download
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_prices(tickers, years):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    prices = yf.download(tickers, start=start, end=end)["Close"]
    return prices.dropna()


with st.spinner("Downloading price data..."):
    prices = load_prices(tickers, lookback_years)

if prices.empty:
    st.error("No data returned — check your tickers and try again.")
    st.stop()

daily_returns = prices.pct_change().dropna()
weight_vector = np.array([normalized_weights[t] for t in daily_returns.columns])
portfolio_returns = daily_returns @ weight_vector

# ---------------------------------------------------------------------------
# VaR CALCULATIONS
# ---------------------------------------------------------------------------

def historical_var(returns, confidence, value):
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    return -cutoff * value, cutoff


def parametric_var(returns, confidence, value):
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    cutoff = mu + z * sigma
    return -cutoff * value, cutoff


def monte_carlo_var(returns, confidence, value, n_sims=50_000):
    mu, sigma = returns.mean(), returns.std()
    sims = np.random.normal(mu, sigma, n_sims)
    cutoff = np.percentile(sims, (1 - confidence) * 100)
    return -cutoff * value, cutoff


hist_var, hist_cutoff = historical_var(portfolio_returns, confidence, portfolio_value)
param_var, param_cutoff = parametric_var(portfolio_returns, confidence, portfolio_value)
mc_var, mc_cutoff = monte_carlo_var(portfolio_returns, confidence, portfolio_value)

# ---------------------------------------------------------------------------
# DISPLAY — metrics
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
col1.metric("Historical VaR", f"${hist_var:,.0f}", f"{hist_cutoff:.2%}")
col2.metric("Parametric VaR", f"${param_var:,.0f}", f"{param_cutoff:.2%}")
col3.metric("Monte Carlo VaR", f"${mc_var:,.0f}", f"{mc_cutoff:.2%}")

st.caption(
    f"At {int(confidence*100)}% confidence, your portfolio is not expected to lose more than "
    f"the amount shown on a given day, based on {lookback_years} years of history."
)

# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------

breaches = portfolio_returns < hist_cutoff
breach_rate = breaches.mean()
expected_rate = 1 - confidence

st.subheader("Backtest")
bcol1, bcol2 = st.columns(2)
bcol1.metric("Observed breach rate", f"{breach_rate:.2%}", f"expected ~{expected_rate:.2%}")
bcol2.metric("Breach days", f"{int(breaches.sum())} / {len(portfolio_returns)}")

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------

st.subheader("Return Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(portfolio_returns, bins=100, alpha=0.6, color="steelblue", label="Daily returns")
ax1.axvline(hist_cutoff, color="red", linestyle="--", label=f"Historical VaR ({hist_cutoff:.2%})")
ax1.axvline(param_cutoff, color="green", linestyle="--", label=f"Parametric VaR ({param_cutoff:.2%})")
ax1.axvline(mc_cutoff, color="orange", linestyle="--", label=f"Monte Carlo VaR ({mc_cutoff:.2%})")
ax1.set_xlabel("Daily Return")
ax1.set_ylabel("Frequency")
ax1.legend()
st.pyplot(fig1)

st.subheader("Rolling 1-Year VaR")
rolling_window = 252
rolling_var = portfolio_returns.rolling(rolling_window).apply(
    lambda x: -np.percentile(x, (1 - confidence) * 100) * portfolio_value
)
fig2, ax2 = plt.subplots(figsize=(10, 4))
rolling_var.plot(ax=ax2)
ax2.set_ylabel("VaR ($)")
st.pyplot(fig2)

st.subheader("Current Weights")
weight_df = pd.DataFrame(
    {"Ticker": list(normalized_weights.keys()), "Weight": [f"{w:.1%}" for w in normalized_weights.values()]}
)
st.table(weight_df)
