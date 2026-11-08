"""
Heston model: VIX futures, VIX calls and the convexity adjustment (Daniel).

Run from the project root:   python3.11 heston/heston_vix.py
Reads data/market_history.csv, data/vix_futures_all.csv and data/risk_free.csv.
Writes to report/figures/: heston_futures_curve.pdf/.png, heston_convexity.pdf/.png,
heston_futures_table.tex and heston_calls_table.tex.

Only the variance v is simulated: under Heston the VIX is an exact function of v,
so the stock price and rho are not needed.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from params import (FIGURES_DIR, FUTURES_FILE, HESTON_GATHERAL, HESTON_N_PATHS, MARKET_FILE,
                    RANDOM_SEED, RISK_FREE_FILE, VIX_CALL_STRIKES)

DAY = 1 / 365
TAU = 30 / 365
CURVE_DATE = "2026-11-03"
MAX_DAYS = 180
CALL_DAYS = {30: "DGS1MO", 91: "DGS3MO"}
CHECK_DAYS = [30, 91, 180]
MARKET_BLUE = "#2a78d6"
MODEL_ORANGE = "#eb6834"
GREY = "#52514e"


def b_factor(kappa):
    """Weight of today's variance in VIX^2 (the rest goes on theta)."""
    return (1 - np.exp(-kappa * TAU)) / (kappa * TAU)


def vix_from_variance(v, kappa, theta):
    """Exact Heston VIX, in points, from the instantaneous variance."""
    v_plus = np.maximum(v, 0)
    return 100 * np.sqrt(theta + (v_plus - theta) * b_factor(kappa))


def variance_from_vix(vix, kappa, theta):
    """Inverse of vix_from_variance: the v that gives this VIX."""
    return theta + ((vix / 100) ** 2 - theta) / b_factor(kappa)


def simulate_variance(v0, kappa, theta, sigma, n_paths, n_days, seed):
    """Full-truncation Euler with daily steps; row d holds v on day d for every path."""
    rng = np.random.default_rng(seed)
    v = np.empty((n_days + 1, n_paths))
    v[0] = v0
    for day in range(n_days):
        v_plus = np.maximum(v[day], 0)
        z = rng.standard_normal(n_paths)
        v[day + 1] = v[day] + kappa * (theta - v_plus) * DAY + sigma * np.sqrt(v_plus * DAY) * z
    return v


def exact_mean_variance(v0, kappa, theta, t):
    return theta + (v0 - theta) * np.exp(-kappa * t)


def exact_mean_vix_squared(v0, kappa, theta, t):
    """E[VIX_T^2] in points squared; VIX^2 is linear in v, so its mean is exact."""
    mean_v = exact_mean_variance(v0, kappa, theta, t)
    return 100 ** 2 * (theta + (mean_v - theta) * b_factor(kappa))


def call_price(vix_t, strike, rate, t):
    payoff = np.maximum(vix_t - strike, 0)
    discount = np.exp(-rate * t)
    return discount * payoff.mean(), discount * payoff.std(ddof=1) / np.sqrt(len(payoff))


def continuous_rate(bey):
    """Treasury bond-equivalent yield to a continuously compounded rate, as in vix_calc.py."""
    return 2 * np.log(1 + bey / 2)


def style(ax):
    ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=1)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def main():
    market = pd.read_csv(MARKET_FILE, index_col="date", parse_dates=True)
    rates = pd.read_csv(RISK_FREE_FILE, index_col="date", parse_dates=True)
    futures = pd.read_csv(FUTURES_FILE, parse_dates=["trade_date", "expiry"])
    futures = futures[futures["settle"] > 0]
    curve = futures[futures["trade_date"] == CURVE_DATE].sort_values("expiry")
    market_days = curve["days_to_expiry"].to_numpy()
    market_prices = curve["settle"].to_numpy()
    spot_vix = market.loc[CURVE_DATE, "VIX"]

    kappa, theta, sigma = HESTON_GATHERAL["kappa"], HESTON_GATHERAL["theta"], HESTON_GATHERAL["sigma"]
    v0 = variance_from_vix(spot_vix, kappa, theta)
    print(f"Spot VIX on {CURVE_DATE}: {spot_vix:.2f}")
    print(f"Gatheral: kappa={kappa}, theta={theta}, sigma={sigma}, rho={HESTON_GATHERAL['rho']}")
    print(f"B = {b_factor(kappa):.4f};  v0 matched to spot VIX = {v0:.4f} (instead of {HESTON_GATHERAL['v0']:.4f})")
    print(f"Feller: 2*kappa*theta = {2 * kappa * theta:.4f} vs sigma^2 = {sigma ** 2:.4f}")

    v = simulate_variance(v0, kappa, theta, sigma, HESTON_N_PATHS, MAX_DAYS, RANDOM_SEED)
    vix = vix_from_variance(v, kappa, theta)
    model_curve = vix.mean(axis=1)
    model_se = vix.std(axis=1, ddof=1) / np.sqrt(HESTON_N_PATHS)
    days = np.arange(MAX_DAYS + 1)
    t = days * DAY

    table = pd.DataFrame({
        "expiry": curve["expiry"].dt.strftime("%Y-%m-%d").to_numpy(),
        "days": market_days,
        "market": market_prices,
        "model": model_curve[market_days],
        "se": model_se[market_days],
    })
    table["market_minus_model"] = table["market"] - table["model"]
    print("\nVIX futures on 3 Nov 2026 (VIX points)")
    print(table.round(3).to_string(index=False))

    print("\nVIX calls (VIX points, then $ per contract at $100 per point)")
    calls = {}
    for n_days, column in CALL_DAYS.items():
        rate = continuous_rate(rates.loc[CURVE_DATE, column])
        for strike in VIX_CALL_STRIKES:
            price, se = call_price(vix[n_days], strike, rate, n_days * DAY)
            calls[(strike, n_days)] = price
            print(f"  K={strike:>2}, T={n_days:>2} days: {price:.3f} (s.e. {se:.3f})  ${100 * price:,.0f}")

    exact_vix2 = exact_mean_vix_squared(v0, kappa, theta, t)
    convexity = np.sqrt(exact_vix2) - model_curve
    print("\nConvexity adjustment sqrt(E[VIX^2]) - E[VIX]")
    for n_days in [30, 91, 180]:
        print(f"  {n_days:>3} days: {convexity[n_days]:.3f}")

    print("\nChecks")
    print(f"  VIX at T=0: {vix[0, 0]:.4f}")
    print(f"  smallest convexity adjustment: {convexity[1:].min():.3f}")
    print(f"  largest futures standard error: {model_se.max():.4f}")
    checks = []
    for n_days in CHECK_DAYS:
        simulated = v[n_days].mean()
        exact = exact_mean_variance(v0, kappa, theta, n_days * DAY)
        two_se = 2 * v[n_days].std(ddof=1) / np.sqrt(HESTON_N_PATHS)
        checks.append((n_days, simulated, exact, 100 * (simulated - exact) / exact, 100 * two_se / exact))
        print(f"  day {n_days}: mean v {simulated:.5f} vs exact {exact:.5f}, "
              f"difference {checks[-1][3]:+.2f}%, random error (2 s.e.) {checks[-1][4]:.2f}%")

    v_double = simulate_variance(v0, kappa, theta, sigma, 2 * HESTON_N_PATHS, MAX_DAYS, RANDOM_SEED + 1)
    vix_double = vix_from_variance(v_double, kappa, theta)
    futures_change = np.abs(vix_double[market_days].mean(axis=1) - model_curve[market_days]).max()
    calls_change = max(abs(call_price(vix_double[n_days], strike, continuous_rate(rates.loc[CURVE_DATE, column]),
                                      n_days * DAY)[0] - calls[(strike, n_days)])
                       for n_days, column in CALL_DAYS.items() for strike in VIX_CALL_STRIKES)
    print(f"  doubling the paths: largest change {futures_change:.3f} (futures), {calls_change:.3f} (calls)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": "#0b0b0b",
                         "xtick.color": GREY, "ytick.color": GREY})

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(days, model_curve, color=MODEL_ORANGE, lw=2, zorder=3, label="Heston model")
    ax.plot(market_days, market_prices, "o", color=MARKET_BLUE, ms=7, zorder=4, label="Market settlements, 3 Nov 2026")
    ax.plot(0, spot_vix, "o", color=MARKET_BLUE, ms=7, mfc="white", zorder=4, label=f"Spot VIX ({spot_vix:.2f})")
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("VIX futures price (points)")
    ax.set_xlim(-3, MAX_DAYS + 3)
    style(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "heston_futures_curve")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(days[1:], convexity[1:], color=MODEL_ORANGE, lw=2, zorder=3)
    for n_days in [30, 91, 180]:
        ax.plot(n_days, convexity[n_days], "o", color=MODEL_ORANGE, ms=6, zorder=4)
        ax.annotate(f"{convexity[n_days]:.2f}", (n_days, convexity[n_days]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel(r"$\sqrt{E[\mathrm{VIX}_T^2]} - E[\mathrm{VIX}_T]$ (points)")
    ax.set_ylim(0, convexity.max() * 1.2)
    style(ax)
    save(fig, "heston_convexity")

    rows = [f"{r.expiry} & {r.days} & {r.market:.2f} & {r.model:.2f} & {r.market_minus_model:+.2f} \\\\"
            for r in table.itertuples()]
    (FIGURES_DIR / "heston_futures_table.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"\textbf{Expiry} & \textbf{Days} & \textbf{Market} & \textbf{Model} & \textbf{Market $-$ model}\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    rows = [f"{k} & {calls[(k, 30)]:.2f} & \\${100 * calls[(k, 30)]:,.0f} & {calls[(k, 91)]:.2f} & \\${100 * calls[(k, 91)]:,.0f} \\\\"
            for k in VIX_CALL_STRIKES]
    (FIGURES_DIR / "heston_calls_table.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{1 month (30 days)}} & \multicolumn{2}{c}{\textbf{3 months (91 days)}}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Strike} & \textbf{Points} & \textbf{Per contract} & \textbf{Points} & \textbf{Per contract}\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    rows = [f"{d} & {sim:.5f} & {exact:.5f} & {diff:+.2f}\\% & {err:.2f}\\% \\\\" for d, sim, exact, diff, err in checks]
    (FIGURES_DIR / "heston_checks_table.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"\textbf{Day} & \textbf{Simulated $\EQ[v_T]$} & \textbf{Exact $\EQ[v_T]$} & \textbf{Difference} & \textbf{Random error (2 s.e.)}\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))
    (FIGURES_DIR / "heston_doubling.tex").write_text(
        f"{futures_change:.3f} points for futures and {calls_change:.3f} points for calls")
    print(f"\nSaved figures and tables to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
