"""
Monte Carlo Simulation for Kaspi.kz (KSPI) — Linear Regression Approach
=========================================================================

Instead of GBM (random-walk) assumptions, this version:
  1. Fits a LINEAR REGRESSION to log(price) vs. time to estimate the
     underlying trend (drift).
  2. Looks at the regression RESIDUALS (actual - trend) to measure how
     much prices wobble around that trend.
  3. Runs Monte Carlo by projecting the trend forward and adding random
     noise drawn from (or resembling) those residuals to each simulated day.

This is a different modeling philosophy than GBM: GBM treats returns as
a random walk with drift; here we assume prices revert toward a fitted
trend line, with noise layered on top. Useful for comparing forecasts
or when you want an explicit trend-following model.

DATA INPUT
----------
Same as the GBM script: put a CSV named CSV_PATH with 'Date','Close'
columns in this folder. If missing, synthetic demo data is generated.

USAGE
-----
    python kaspi_linear_regression_mc.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------- CONFIG ------------------------------
CSV_PATH        = "kaspi_prices.csv"
TICKER_LABEL    = "KSPI (Kaspi.kz)"

N_SIMULATIONS   = 10_000
HORIZON_DAYS    = 252
TRADING_DAYS_YR = 252

NOISE_MODE      = "bootstrap"   # "bootstrap" (resample real residuals) | "normal" (fit a Gaussian to residuals)

CONFIDENCE      = 0.95
RANDOM_SEED     = 42

OUTPUT_DIR      = "output"      # change to "/mnt/user-data/outputs" if running in Claude's sandbox
# ---------------------------------------------------------------------

np.random.seed(RANDOM_SEED)


# --------------------------- DATA LOADING -----------------------------
def load_price_history(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        print(f"Loaded {len(df)} rows of real historical data from '{csv_path}'.")
        return df[["Date", "Close"]]

    print(f"No CSV found at '{csv_path}' — generating synthetic demo data instead.")
    return _generate_synthetic_history()


def _generate_synthetic_history(n_days: int = 750, start_price: float = 90.0) -> pd.DataFrame:
    mu_annual, sigma_annual = 0.25, 0.42
    dt = 1 / TRADING_DAYS_YR
    mu_daily = mu_annual * dt
    sigma_daily = sigma_annual * np.sqrt(dt)

    shocks = np.random.normal(mu_daily, sigma_daily, n_days)
    jump_days = np.random.choice(n_days, size=max(3, n_days // 120), replace=False)
    shocks[jump_days] += np.random.normal(0, 0.06, len(jump_days))

    log_prices = np.log(start_price) + np.cumsum(shocks)
    prices = np.exp(log_prices)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame({"Date": dates, "Close": prices})


# ------------------------ LINEAR REGRESSION FIT -------------------------
def fit_trend_model(df: pd.DataFrame) -> dict:
    """
    Fit log(price) = a + b * t  via linear regression, where t is the
    trading-day index. Returns the fitted model plus residual stats.
    """
    t = np.arange(len(df)).reshape(-1, 1)
    log_price = np.log(df["Close"].values)

    model = LinearRegression()
    model.fit(t, log_price)

    fitted = model.predict(t)
    residuals = log_price - fitted

    slope = model.coef_[0]          # daily trend in log-price terms
    intercept = model.intercept_
    r2 = model.score(t, log_price)

    # IMPORTANT: residual *levels* are autocorrelated (a stock can drift
    # far from a straight trend line for long stretches). Using their
    # std directly as day-to-day noise and cumulatively summing it wildly
    # overstates volatility. What we actually want for a day-step random
    # walk is the day-to-day CHANGE in the residual (roughly equivalent
    # to detrended daily log-returns).
    residual_diffs = np.diff(residuals)

    return {
        "model": model,
        "slope_daily": slope,
        "intercept": intercept,
        "r2": r2,
        "residuals": residuals,
        "residual_diffs": residual_diffs,
        "daily_noise_std": residual_diffs.std(),
        "fitted_log_price": fitted,
        "n_obs": len(df),
    }


# ------------------------------ SIMULATION -------------------------------
def simulate_regression_mc(s0, last_t, slope, daily_noise_std, residual_diffs,
                            n_days, n_sims, noise_mode):
    """
    Each simulated day, the price moves by the regression's daily trend
    (slope) plus a random day-to-day noise term. Summed cumulatively,
    this gives a random walk that drifts along the fitted trend's slope
    with realistic day-to-day wobble (calibrated from how much the
    actual price deviates from the trend line day over day).
    """
    if noise_mode == "normal":
        noise = np.random.normal(0, daily_noise_std, size=(n_days, n_sims))
    elif noise_mode == "bootstrap":
        noise = np.random.choice(residual_diffs, size=(n_days, n_sims), replace=True)
    else:
        raise ValueError(f"Unknown NOISE_MODE: {noise_mode}")

    daily_log_change = slope + noise
    log_paths = np.log(s0) + np.cumsum(daily_log_change, axis=0)
    paths = np.exp(log_paths)

    future_t = np.arange(last_t + 1, last_t + 1 + n_days)
    trend_log_price = np.log(s0) + slope * (future_t - last_t)
    return np.vstack([np.full(n_sims, s0), paths]), trend_log_price


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
def plot_results(hist_df, fit, paths, terminal_prices, metrics, s0, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_days = paths.shape[0] - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Fan chart with trend line ---
    ax = axes[0]
    sample_idx = np.random.choice(paths.shape[1], size=min(200, paths.shape[1]), replace=False)
    ax.plot(paths[:, sample_idx], color="seagreen", alpha=0.04)

    for p, c in zip([5, 25, 50, 75, 95],
                     ["#c7e9c0", "#74c476", "#006d2c", "#74c476", "#c7e9c0"]):
        ax.plot(np.percentile(paths, p, axis=1), color=c, linewidth=2,
                 label=f"{p}th pct" if p != 50 else "Median")

    ax.axhline(s0, color="black", linestyle="--", linewidth=1, label="Start price")
    ax.set_title(f"{TICKER_LABEL} — Linear-Regression-Trend Monte Carlo\n"
                 f"({paths.shape[1]:,} paths, noise={NOISE_MODE}, R²={fit['r2']:.3f})")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # --- Histogram of terminal prices ---
    ax2 = axes[1]
    ax2.hist(terminal_prices, bins=80, color="seagreen", alpha=0.75, edgecolor="white")
    ax2.axvline(s0, color="black", linestyle="--", label="Start price")
    ax2.axvline(metrics["median_terminal"], color="darkorange", label="Median")
    ax2.axvline(metrics["VaR_price"], color="crimson", label=f"VaR {int(CONFIDENCE*100)}%")
    ax2.set_title(f"Distribution of Simulated Prices\nafter {n_days} Trading Days")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "kaspi_linreg_mc_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")
    plt.close(fig)

    # --- Historical prices with fitted trend line overlaid ---
    fig2, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(hist_df["Date"], hist_df["Close"], color="darkslateblue", label="Actual price")
    ax3.plot(hist_df["Date"], np.exp(fit["fitted_log_price"]), color="crimson",
              linewidth=2, label="Linear regression trend")
    ax3.set_title(f"{TICKER_LABEL} — Historical Prices & Fitted Log-Linear Trend "
                  f"(R²={fit['r2']:.3f})")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "kaspi_linreg_trend_fit.png")
    plt.savefig(hist_path, dpi=150)
    print(f"Saved plot to {hist_path}")
    plt.close(fig2)

    return fig_path, hist_path


# --------------------------------- MAIN ------------------------------------
def main():
    hist_df = load_price_history(CSV_PATH)
    s0 = hist_df["Close"].iloc[-1]
    last_t = len(hist_df) - 1

    fit = fit_trend_model(hist_df)

    print("\n--- Linear regression fit (log-price vs. time) ---")
    print(f"  Daily trend (slope):     {fit['slope_daily']:.5f}  "
          f"(~{fit['slope_daily']*TRADING_DAYS_YR:.2%} annualized)")
    print(f"  R² (fit quality):        {fit['r2']:.4f}")
    print(f"  Daily noise std:         {fit['daily_noise_std']:.5f}")
    print(f"  Observations used:       {fit['n_obs']}")
    print(f"  Current (last) price:    {s0:.2f}")

    paths, trend_log_price = simulate_regression_mc(
        s0, last_t, fit["slope_daily"], fit["daily_noise_std"], fit["residual_diffs"],
        HORIZON_DAYS, N_SIMULATIONS, NOISE_MODE
    )

    terminal_prices = paths[-1]
    metrics = compute_risk_metrics(terminal_prices, s0, CONFIDENCE)

    print(f"\n--- Simulation results ({HORIZON_DAYS} trading days ahead, "
          f"{N_SIMULATIONS:,} paths, noise_mode='{NOISE_MODE}') ---")
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

    plot_results(hist_df, fit, paths, terminal_prices, metrics, s0, OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"terminal_price": terminal_prices}).to_csv(
        os.path.join(OUTPUT_DIR, "kaspi_linreg_terminal_prices.csv"), index=False
    )
    print(f"\nSaved raw terminal-price distribution to "
          f"{OUTPUT_DIR}/kaspi_linreg_terminal_prices.csv")


if __name__ == "__main__":
    main()