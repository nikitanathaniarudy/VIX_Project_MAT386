"""
Figures for the hedging-backtest section.

Reads data/hedging_backtest_daily.csv and data/hedging_backtest_episodes.csv,
written by hedging_backtest.py, and writes PDF + PNG to report/figures/.

    hedge_ratio_rolling   rolling n* and actual contracts held
    hedging_equity        equity curves and drawdowns
    hedging_roll_cost     cumulative roll cost vs futures P&L, annual roll cost
    hedging_crises        unhedged vs hedged returns in 2018 and 2020 episodes

Run from the project root:
    python report/make_hedging_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
EVENTS = [("2018-02-05", "Volmageddon"), ("2020-03-16", "COVID")]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def tidy(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


def mark_events(ax, labels=False):
    for d, lab in EVENTS:
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.7, ls=":")
        if labels:
            left = lab == "Volmageddon"
            ax.annotate(
                lab,
                xy=(pd.Timestamp(d), 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(-4 if left else 4, -2),
                textcoords="offset points",
                fontsize=8,
                color=INK,
                va="top",
                ha="right" if left else "left",
            )


def year_axis(ax, every=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(every))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main():
    d = pd.read_csv(DATA / "hedging_backtest_daily.csv", parse_dates=["date"]).set_index("date")
    ep = pd.read_csv(DATA / "hedging_backtest_episodes.csv")

    # ---------------------------------------------------------------- Figure 1
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(6.8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    a1.step(d.index, d["n_per_m_252"], where="post", color=ORANGE, lw=1.5,
            label="252-day window")
    a1.step(d.index, d["n_per_m_504"], where="post", color=AQUA, lw=1.3, ls="--",
            label="504-day window")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_ylabel("$n^*$ per \\$1M SPY")
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    tidy(a1)
    mark_events(a1, labels=True)

    a2.step(d.index, d["N_252"], where="post", color=ORANGE, lw=1.5,
            label="252-day window")
    a2.step(d.index, d["N_504"], where="post", color=AQUA, lw=1.3, ls="--",
            label="504-day window")
    a2.set_ylabel("Contracts held ($N_t$)")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a2)
    mark_events(a2)

    year_axis(a2)
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, "hedge_ratio_rolling")

    # ---------------------------------------------------------------- Figure 2
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(6.8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]}
    )

    series = [
        ("spy", "Unhedged SPY", MUTED, "-"),
        ("total_252", "Rolling hedge (252d)", ORANGE, "-"),
        ("total_504", "Rolling hedge (504d)", AQUA, "--"),
    ]
    for col, lab, color, ls in series:
        a1.plot(d.index, d[col] / 1e6, color=color, lw=1.3, ls=ls, label=lab)
    a1.set_yscale("log")
    a1.set_ylabel("Portfolio value (\\$m, log scale)")
    a1.set_yticks([1.0, 1.5, 2.0, 3.0, 4.0])
    a1.set_yticklabels(["1.0", "1.5", "2.0", "3.0", "4.0"])
    a1.yaxis.set_minor_formatter(mtick.NullFormatter())
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a1)
    mark_events(a1, labels=True)

    for col, lab, color, ls in series:
        dd = d[col] / d[col].cummax() - 1
        a2.plot(d.index, dd * 100, color=color, lw=1.3, ls=ls, label=lab)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("Drawdown (\\%)")
    a2.yaxis.set_major_formatter(mtick.PercentFormatter())
    tidy(a2)
    mark_events(a2)

    year_axis(a2)
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, "hedging_equity")

    # ---------------------------------------------------------------- Figure 3
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(6.8, 4.8), gridspec_kw={"height_ratios": [1.3, 1]}
    )

    a1.plot(d.index, d["roll_cost_252"].cumsum() / 1e6, color=ORANGE, lw=1.5,
            label="Cumulative roll cost")
    a1.plot(d.index, d["fut_pnl_252"].cumsum() / 1e6, color=BLUE, lw=1.5,
            label="Cumulative futures P&L")
    a1.plot(d.index, d["gross_price_pnl_252"].cumsum() / 1e6, color=AQUA, lw=1.2,
            ls="--", label="Gross price change before roll")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_ylabel("\\$m")
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(a1)

    annual = d.groupby(d.index.year)["roll_cost_252"].sum() / 1e6
    a2.bar(annual.index, annual.values, color=ORANGE, width=0.8)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("Roll cost (\\$m/year)")
    a2.set_xticks(range(annual.index.min(), annual.index.max() + 1, 2))
    tidy(a2)

    top_year = annual.idxmax()
    a2.annotate(
        f"\\${annual.max():.2f}m",
        xy=(top_year, annual.max()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=INK,
    )

    fig.tight_layout()
    save(fig, "hedging_roll_cost")

    # ---------------------------------------------------------------- Figure 4
    order = ["2018 Q1", "Volmageddon", "2020 Q1", "COVID crash"]
    labels = [
        "2018 Q1\nJan-Mar",
        "Volmageddon\n1-9 Feb 2018",
        "2020 Q1\nJan-Mar",
        "COVID crash\n19 Feb-23 Mar 2020",
    ]
    base = ep[ep["window"] == 252].set_index("episode").loc[order]
    h252 = ep[ep["window"] == 252].set_index("episode").loc[order, "hedged_ret"]
    h504 = ep[ep["window"] == 504].set_index("episode").loc[order, "hedged_ret"]
    unhedged = base["unhedged_ret"]

    x = np.arange(len(order))
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(x - width, unhedged * 100, width, color=MUTED, label="Unhedged SPY")
    ax.bar(x, h252 * 100, width, color=ORANGE, label="Rolling hedge (252d)")
    ax.bar(x + width, h504 * 100, width, color=AQUA, label="Rolling hedge (504d)")

    for i, vals in enumerate(zip(unhedged * 100, h252 * 100, h504 * 100)):
        for offset, v in zip((-width, 0, width), vals):
            if abs(v) < 0.05:
                y, va, dy, text = 0.6, "bottom", 0, "0.0%"
            elif v >= 0:
                y, va, dy, text = v, "bottom", 2, f"{v:.1f}%"
            else:
                y, va, dy, text = v, "top", -2, f"{v:.1f}%"
            ax.annotate(
                text,
                xy=(i + offset, y),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=7.5,
                color=INK,
            )

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Return over episode (\\%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0, 1.01, 1, 0.1),
        mode="expand",
        ncol=3,
    )
    tidy(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "hedging_crises")


if __name__ == "__main__":
    main()
