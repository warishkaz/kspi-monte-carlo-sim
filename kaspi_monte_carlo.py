"""
Monte Carlo Simulation for Kaspi.kz (KSPI) Stock Price
=========================================================

Simulates future price paths using Geometric Brownian Motion (GBM),
calibrated from historical daily returns, plus risk metrics (VaR, ES)
and visualizations.

DATA INPUT
----------
Option A (recommended): Provide a CSV with historical daily prices.
    Required columns: 'Date', 'Close'
    e.g. exported from Yahoo Finance, Investing.com, or KASE.
    Set CSV_PATH below to your file.

Option B (fallback / demo): If no CSV is found, the script generates
    a synthetic-but-realistic historical price series for KSPI using
    parameters roughly in line with its public trading history
    (high growth, elevated volatility). This lets you test the full
    pipeline immediately; swap in real data whenever you have it.

USAGE
-----
    python kaspi_monte_carlo.py

Adjust the CONFIG block below to change simulation horizon, number
of paths, model choice, etc.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------- CONFIG ------------------------------
CSV_PATH        = "kaspi_prices.csv"   # set to your own historical data file
TICKER_LABEL    = "KSPI (Kaspi.kz)"

N_SIMULATIONS   = 10_000               # number of Monte Carlo paths
HORIZON_DAYS    = 252                  # trading days to simulate forward (~1 year)
TRADING_DAYS_YR = 252

MODEL           = "gbm"                # "gbm" | "bootstrap" | "jump_diffusion"

# Jump-diffusion extra params (only used if MODEL == "jump_diffusion")
JUMP_INTENSITY  = 8      # expected number of jumps per year
JUMP_MEAN       = -0.02  # average jump size (log-return)
JUMP_STD        = 0.06   # jump size volatility

CONFIDENCE      = 0.95   # for VaR / Expected Shortfall
RANDOM_SEED     = 42

OUTPUT_DIR      = "/mnt/user-data/outputs"
# ---------------------------------------------------------------------

np.random.seed(RANDOM_SEED)


# --------------------------- DATA LOADING -----------------------------
def load_price_history(csv_path: str) -> pd.DataFrame:
    """Load historical prices from CSV, or fall back to a synthetic series."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        print(f"Loaded {len(df)} rows of real historical data from '{csv_path}'.")
        return df[["Date", "Close"]]

    print(f"No CSV found at '{csv_path}' — generating a synthetic demo "
          f"price history instead (replace with real KSPI data when ready).")
    return _generate_synthetic_history()


def _generate_synthetic_history(n_days: int = 750, start_price: float = 90.0) -> pd.DataFrame:
    """
    Generate a plausible synthetic daily price series for demo purposes.
    Parameters are set in the ballpark of a high-growth, high-volatility
    emerging-market fintech stock -- NOT real KSPI data.
    """
    mu_annual, sigma_annual = 0.25, 0.42          # illustrative drift/vol
    dt = 1 / TRADING_DAYS_YR
    mu_daily = mu_annual * dt
    sigma_daily = sigma_annual * np.sqrt(dt)

    shocks = np.random.normal(mu_daily, sigma_daily, n_days)
    # sprinkle in a few larger jumps to mimic earnings/news shocks
    jump_days = np.random.choice(n_days, size=max(3, n_days // 120), replace=False)
    shocks[jump_days] += np.random.normal(0, 0.06, len(jump_days))

    log_prices = np.log(start_price) + np.cumsum(shocks)
    prices = np.exp(log_prices)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame({"Date": dates, "Close": prices})


# ------------------------ PARAMETER ESTIMATION -------------------------
def estimate_parameters(prices: pd.Series) -> dict:
    log_returns = np.log(prices / prices.shift(1)).dropna()

    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    return {
        "log_returns": log_returns,
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annual": mu_daily * TRADING_DAYS_YR,
        "sigma_annual": sigma_daily * np.sqrt(TRADING_DAYS_YR),
    }


# ------------------------------ MODELS ----------------------------------
def simulate_gbm(s0, mu, sigma, n_days, n_sims):
    dt = 1
    z = np.random.normal(size=(n_days, n_sims))
    daily_log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.log(s0) + np.cumsum(daily_log_returns, axis=0)
    paths = np.exp(log_paths)
    return np.vstack([np.full(n_sims, s0), paths])


def simulate_bootstrap(s0, historical_log_returns, n_days, n_sims):
    sampled = np.random.choice(
        historical_log_returns, size=(n_days, n_sims), replace=True
    )
    log_paths = np.log(s0) + np.cumsum(sampled, axis=0)
    paths = np.exp(log_paths)
    return np.vstack([np.full(n_sims, s0), paths])


def simulate_jump_diffusion(s0, mu, sigma, n_days, n_sims,
                             jump_intensity, jump_mean, jump_std):
    dt = 1
    lam_daily = jump_intensity / TRADING_DAYS_YR

    z = np.random.normal(size=(n_days, n_sims))
    diffusion = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

    n_jumps = np.random.poisson(lam_daily, size=(n_days, n_sims))
    jump_sizes = np.random.normal(jump_mean, jump_std, size=(n_days, n_sims))
    jump_component = n_jumps * jump_sizes

    daily_log_returns = diffusion + jump_component
    log_paths = np.log(s0) + np.cumsum(daily_log_returns, axis=0)
    paths = np.exp(log_paths)
    return np.vstack([np.full(n_sims, s0), paths])


# ------------------------------ ANALYSIS ---------------------------------
def compute_risk_metrics(terminal_prices, s0, confidence):
    returns = terminal_prices / s0 - 1
    var_pct = np.percentile(returns, (1 - confidence) * 100)
    es_pct = returns[returns <= var_pct].mean()

    return {
        "mean_terminal": terminal_prices.mean(),
        "median_terminal": np.median(terminal_prices),
        "std_terminal": terminal_prices.std(),
        "p5": np.percentile(terminal_prices, 5),
        "p25": np.percentile(terminal_prices, 25),
        "p75": np.percentile(terminal_prices, 75),
        "p95": np.percentile(terminal_prices, 95),
        "prob_gain": (terminal_prices > s0).mean(),
        "VaR_pct": var_pct,
        "ES_pct": es_pct,
        "VaR_price": s0 * (1 + var_pct),
        "ES_price": s0 * (1 + es_pct),
    }


# ------------------------------ PLOTTING ----------------------------------
def plot_results(hist_df, paths, terminal_prices, metrics, s0, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_days = paths.shape[0] - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Fan chart: sample of simulated paths ---
    ax = axes[0]
    sample_idx = np.random.choice(paths.shape[1], size=min(200, paths.shape[1]), replace=False)
    ax.plot(paths[:, sample_idx], color="steelblue", alpha=0.04)

    pct_bands = [5, 25, 50, 75, 95]
    colors = ["#c6dbef", "#6baed6", "#08519c", "#6baed6", "#c6dbef"]
    for p, c in zip(pct_bands, colors):
        ax.plot(np.percentile(paths, p, axis=1), color=c, linewidth=2,
                 label=f"{p}th pct" if p != 50 else "Median")

    ax.axhline(s0, color="black", linestyle="--", linewidth=1, label="Start price")
    ax.set_title(f"{TICKER_LABEL} — {n_days}-Day Monte Carlo Simulation\n"
                 f"({paths.shape[1]:,} paths, model={MODEL})")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # --- Histogram of terminal prices ---
    ax2 = axes[1]
    ax2.hist(terminal_prices, bins=80, color="steelblue", alpha=0.75, edgecolor="white")
    ax2.axvline(s0, color="black", linestyle="--", label="Start price")
    ax2.axvline(metrics["median_terminal"], color="darkorange", label="Median")
    ax2.axvline(metrics["VaR_price"], color="crimson", label=f"VaR {int(CONFIDENCE*100)}%")
    ax2.set_title(f"Distribution of Simulated Prices\nafter {n_days} Trading Days")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "kaspi_monte_carlo_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")
    plt.close(fig)

    # --- Historical price chart used for calibration ---
    fig2, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(hist_df["Date"], hist_df["Close"], color="darkslateblue")
    ax3.set_title(f"{TICKER_LABEL} — Historical Prices Used for Calibration")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "kaspi_historical_prices.png")
    plt.savefig(hist_path, dpi=150)
    print(f"Saved plot to {hist_path}")
    plt.close(fig2)

    return fig_path, hist_path


# --------------------------------- MAIN ------------------------------------
def main():
    hist_df = load_price_history(CSV_PATH)
    s0 = hist_df["Close"].iloc[-1]

    params = estimate_parameters(hist_df["Close"])
    print("\n--- Calibrated parameters (from historical daily log-returns) ---")
    print(f"  Daily mean return:      {params['mu_daily']:.5f}")
    print(f"  Daily volatility:       {params['sigma_daily']:.5f}")
    print(f"  Annualized return:      {params['mu_annual']:.2%}")
    print(f"  Annualized volatility:  {params['sigma_annual']:.2%}")
    print(f"  Current (last) price:   {s0:.2f}")

    if MODEL == "gbm":
        paths = simulate_gbm(s0, params["mu_daily"], params["sigma_daily"],
                              HORIZON_DAYS, N_SIMULATIONS)
    elif MODEL == "bootstrap":
        paths = simulate_bootstrap(s0, params["log_returns"].values,
                                    HORIZON_DAYS, N_SIMULATIONS)
    elif MODEL == "jump_diffusion":
        paths = simulate_jump_diffusion(s0, params["mu_daily"], params["sigma_daily"],
                                         HORIZON_DAYS, N_SIMULATIONS,
                                         JUMP_INTENSITY, JUMP_MEAN, JUMP_STD)
    else:
        raise ValueError(f"Unknown MODEL: {MODEL}")

    terminal_prices = paths[-1]
    metrics = compute_risk_metrics(terminal_prices, s0, CONFIDENCE)

    print(f"\n--- Simulation results ({HORIZON_DAYS} trading days ahead, "
          f"{N_SIMULATIONS:,} paths, model='{MODEL}') ---")
    print(f"  Mean terminal price:      {metrics['mean_terminal']:.2f}")
    print(f"  Median terminal price:    {metrics['median_terminal']:.2f}")
    print(f"  Std dev:                  {metrics['std_terminal']:.2f}")
    print(f"  5th / 95th percentile:    {metrics['p5']:.2f} / {metrics['p95']:.2f}")
    print(f"  25th / 75th percentile:   {metrics['p25']:.2f} / {metrics['p75']:.2f}")
    print(f"  P(price > today):         {metrics['prob_gain']:.1%}")
    print(f"  {int(CONFIDENCE*100)}% VaR (return):        {metrics['VaR_pct']:.2%}  "
          f"(price: {metrics['VaR_price']:.2f})")
    print(f"  {int(CONFIDENCE*100)}% Expected Shortfall:  {metrics['ES_pct']:.2%}  "
          f"(price: {metrics['ES_price']:.2f})")

    plot_results(hist_df, paths, terminal_prices, metrics, s0, OUTPUT_DIR)

    # Save terminal price distribution + summary to CSV for further analysis
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"terminal_price": terminal_prices}).to_csv(
        os.path.join(OUTPUT_DIR, "kaspi_simulated_terminal_prices.csv"), index=False
    )
    print(f"\nSaved raw terminal-price distribution to "
          f"{OUTPUT_DIR}/kaspi_simulated_terminal_prices.csv")


if __name__ == "__main__":
    main()