"""
Portfolio Value at Risk (VaR) Calculator
Holdings: VTI, VXUS, SCHD, XLF, BND, JQUA, KO, JPM, SPY

Implements three VaR methodologies:
  1. Historical simulation
  2. Variance-covariance (parametric, normal distribution)
  3. Monte Carlo simulation

Also backtests the historical VaR estimate against realized losses.

Requires: pip install yfinance pandas numpy matplotlib scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# 1. CONFIG — edit these to match your actual weights
# ---------------------------------------------------------------------------

TICKERS = ["VTI", "VXUS", "SCHD", "XLF", "BND", "JQUA", "KO", "JPM", "SPY"]

# Equal-weighted by default (11.11% each). Replace with your real dollar
# weights once you know them — e.g. {"VTI": 0.25, "VXUS": 0.10, ...}
WEIGHTS = {ticker: 1 / len(TICKERS) for ticker in TICKERS}

LOOKBACK_YEARS = 5
CONFIDENCE_LEVEL = 0.95          # i.e. 95% VaR
PORTFOLIO_VALUE = 100_000        # assumed $ value, purely for dollar VaR output
N_SIMULATIONS = 100_000          # Monte Carlo paths


# ---------------------------------------------------------------------------
# 2. DATA
# ---------------------------------------------------------------------------

def get_price_data(tickers, years):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    prices = yf.download(tickers, start=start, end=end)["Close"]
    prices = prices.dropna()
    return prices


def get_portfolio_returns(prices, weights):
    daily_returns = prices.pct_change().dropna()
    weight_vector = np.array([weights[t] for t in daily_returns.columns])
    portfolio_returns = daily_returns @ weight_vector
    return daily_returns, portfolio_returns


# ---------------------------------------------------------------------------
# 3. VaR METHODS
# ---------------------------------------------------------------------------

def historical_var(portfolio_returns, confidence, value):
    cutoff = np.percentile(portfolio_returns, (1 - confidence) * 100)
    return -cutoff * value, cutoff


def parametric_var(portfolio_returns, confidence, value):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = norm.ppf(1 - confidence)
    cutoff = mu + z * sigma
    return -cutoff * value, cutoff


def monte_carlo_var(portfolio_returns, confidence, value, n_sims):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    simulated_returns = np.random.normal(mu, sigma, n_sims)
    cutoff = np.percentile(simulated_returns, (1 - confidence) * 100)
    return -cutoff * value, cutoff, simulated_returns


# ---------------------------------------------------------------------------
# 4. BACKTEST — how often did realized losses exceed the historical VaR?
# ---------------------------------------------------------------------------

def backtest_var(portfolio_returns, var_cutoff, confidence):
    breaches = portfolio_returns < var_cutoff
    breach_rate = breaches.mean()
    expected_rate = 1 - confidence
    return breaches, breach_rate, expected_rate


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    print("Downloading price data...")
    prices = get_price_data(TICKERS, LOOKBACK_YEARS)
    daily_returns, portfolio_returns = get_portfolio_returns(prices, WEIGHTS)

    hist_var, hist_cutoff = historical_var(portfolio_returns, CONFIDENCE_LEVEL, PORTFOLIO_VALUE)
    param_var, param_cutoff = parametric_var(portfolio_returns, CONFIDENCE_LEVEL, PORTFOLIO_VALUE)
    mc_var, mc_cutoff, mc_returns = monte_carlo_var(
        portfolio_returns, CONFIDENCE_LEVEL, PORTFOLIO_VALUE, N_SIMULATIONS
    )

    print(f"\n--- {int(CONFIDENCE_LEVEL*100)}% 1-Day VaR on ${PORTFOLIO_VALUE:,.0f} portfolio ---")
    print(f"Historical simulation : ${hist_var:,.2f}")
    print(f"Parametric (var-cov)  : ${param_var:,.2f}")
    print(f"Monte Carlo           : ${mc_var:,.2f}")

    breaches, breach_rate, expected_rate = backtest_var(
        portfolio_returns, hist_cutoff, CONFIDENCE_LEVEL
    )
    print(f"\n--- Backtest (historical VaR) ---")
    print(f"Observed breach rate : {breach_rate:.2%}  (expected ~{expected_rate:.2%})")
    print(f"Number of breaches   : {breaches.sum()} out of {len(portfolio_returns)} days")

    # --- Plot: return distribution with VaR thresholds ---
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_returns, bins=100, alpha=0.6, label="Historical daily returns", color="steelblue")
    plt.axvline(hist_cutoff, color="red", linestyle="--", label=f"Historical VaR ({hist_cutoff:.2%})")
    plt.axvline(param_cutoff, color="green", linestyle="--", label=f"Parametric VaR ({param_cutoff:.2%})")
    plt.axvline(mc_cutoff, color="orange", linestyle="--", label=f"Monte Carlo VaR ({mc_cutoff:.2%})")
    plt.title(f"Portfolio Daily Return Distribution — {int(CONFIDENCE_LEVEL*100)}% VaR Comparison")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("var_comparison.png", dpi=150)
    print("\nSaved chart to var_comparison.png")

    # --- Plot: rolling 1-year VaR over time (stretch goal, historical method) ---
    rolling_window = 252
    rolling_var = portfolio_returns.rolling(rolling_window).apply(
        lambda x: -np.percentile(x, (1 - CONFIDENCE_LEVEL) * 100) * PORTFOLIO_VALUE
    )
    plt.figure(figsize=(10, 5))
    rolling_var.plot()
    plt.title(f"Rolling {rolling_window}-Day Historical VaR (${PORTFOLIO_VALUE:,.0f} portfolio)")
    plt.ylabel("VaR ($)")
    plt.tight_layout()
    plt.savefig("rolling_var.png", dpi=150)
    print("Saved chart to rolling_var.png")


if __name__ == "__main__":
    main()
    