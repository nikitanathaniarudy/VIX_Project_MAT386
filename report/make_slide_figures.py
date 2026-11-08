"""
Slide-sized charts for the Canva deck. Each PNG is drawn at the aspect ratio of
the box it fills on its slide, with larger type than the report figures.

Run from the project root:   python3.11 report/make_slide_figures.py
Writes report/figures/slides/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "report" / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0})


def fig_for(w_px, h_px):
    """Figure whose aspect matches a w_px x h_px slide box (100 px = 1 inch)."""
    return plt.figure(figsize=(w_px / 100, h_px / 100))


def tidy(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=250, facecolor="white")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


m = pd.read_csv(DATA / "market_history.csv", index_col=0, parse_dates=True)
fm = pd.read_csv(DATA / "vix_front_month.csv", index_col=0, parse_dates=True)
etp_path = DATA / "etp_history.csv"
etp = pd.read_csv(etp_path, index_col=0, parse_dates=True) if etp_path.exists() else None

# ---- Slide 3: VIX vs S&P 500 (box 820 x 600)
fig = fig_for(820, 600)
a1, a2 = fig.subplots(2, 1, sharex=True)
a1.plot(m.index, m["SPX"], color=BLUE, lw=1.6)
a1.set_ylabel("S&P 500")
a2.plot(m.index, m["VIX"], color=ORANGE, lw=1.1)
a2.set_ylabel("VIX")
for d, lab in [("2018-02-05", "Feb 2018"), ("2020-03-16", "Mar 2020")]:
    for ax in (a1, a2):
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.9, ls=":")
a2.annotate("COVID\nMar 2020", xy=(pd.Timestamp("2020-03-16"), 82.7), xytext=(10, -4),
            textcoords="offset points", fontsize=13, va="top", color=INK)
a2.annotate("Volmageddon\nFeb 2018", xy=(pd.Timestamp("2018-02-05"), 37), xytext=(pd.Timestamp("2017-10-01"), 78),
            fontsize=13, va="top", ha="right", color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
for ax in (a1, a2):
    tidy(ax)
a2.xaxis.set_major_locator(mdates.YearLocator(4))
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.align_ylabels()
fig.tight_layout()
save(fig, "s03_vix_vs_spx")

# ---- Slide 4: VIXY decay strip (box 1520 x 250)
if etp is not None and "VIXY" in etp:
    v = etp["VIXY"].dropna()
    fig = fig_for(1520, 250)
    ax = fig.subplots()
    ax.plot(v.index, v, color=BLUE, lw=1.8)
    ax.set_yscale("log")
    ax.set_ylabel("VIXY (log)")
    tidy(ax)
    loss = 1 - v.iloc[-1] / v.iloc[0]
    ax.text(0.99, 0.88, f"VIXY, split-adjusted: down {loss:.3%} since {v.index[0]:%b %Y}",
            transform=ax.transAxes, ha="right", va="top", fontsize=15, color=INK)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save(fig, "s04_vixy_decay")

# ---- Slide 5: Volmageddon (box 880 x 610)
start, end = "2018-01-15", "2018-03-15"
spot, fut = m.loc[start:end, "VIX"], fm.loc[start:end, "front_settle"]
has_svxy = etp is not None and "SVXY" in etp and etp["SVXY"].loc[start:end].notna().sum() > 10
fig = fig_for(880, 610)
axes = fig.subplots(2 if has_svxy else 1, 1, sharex=True, gridspec_kw={"height_ratios": [1.4, 1]} if has_svxy else None)
a1 = axes[0] if has_svxy else axes
a1.plot(spot.index, spot, color=ORANGE, lw=2, label="VIX (spot)")
a1.plot(fut.index, fut, color=BLUE, lw=2, label="Front-month future")
a1.axvline(pd.Timestamp("2018-02-05"), color=MUTED, lw=0.9, ls=":")
a1.set_ylabel("VIX points")
a1.legend(frameon=False, fontsize=13, loc="upper right")
tidy(a1)
if has_svxy:
    s = etp["SVXY"].loc[start:end]
    a2 = axes[1]
    a2.plot(s.index, s / s.iloc[0] * 100, color=AQUA, lw=2)
    a2.axvline(pd.Timestamp("2018-02-05"), color=MUTED, lw=0.9, ls=":")
    a2.set_ylabel("SVXY\n(15 Jan = 100)")
    low = (s / s.iloc[0] * 100).loc["2018-02-06":].min()
    a2.text(pd.Timestamp("2018-02-14"), 55, f"inverse ETF fell to {low:.0f}", fontsize=13, color=INK)
    tidy(a2)
(axes[-1] if has_svxy else a1).xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.align_ylabels()
fig.tight_layout()
save(fig, "s05_volmageddon")

# ---- Slide 6: the strike strip used in our VIX (box 900 x 230)
k = pd.read_csv(DATA / "vix_calc_strikes.csv")
near = k[k.term == "near"].sort_values("strike")
fig = fig_for(900, 230)
ax = fig.subplots()
for opt, col, lab in [("put", ORANGE, "OTM puts"), ("call", BLUE, "OTM calls"), ("put/call avg", INK, "K₀")]:
    x = near[near.option == opt]
    ax.vlines(x.strike, 0, x.mid, color=col, lw=1.4, label=lab)
k0 = near.loc[near.option == "put/call avg", "strike"].iloc[0]
ax.axvline(k0, color=INK, lw=1, ls="--")
ax.set_yscale("log")
ax.set_ylabel("mid quote ($)", fontsize=13)
ax.set_xlabel(f"SPX strike  (near term, {near.expiry.iloc[0]}; {len(near)} strikes)", fontsize=13)
ax.legend(frameon=False, fontsize=12, loc="upper left", ncol=3)
ax.set_ylim(0.05, near["mid"].max() * 6)
tidy(ax)
fig.tight_layout()
save(fig, "s06_strike_strip")

# ---- Slide 8: futures curve on a calm day vs a crisis day (box 1120 x 200)
f = pd.read_csv(DATA / "vix_futures_all.csv", parse_dates=["trade_date", "expiry"])
f = f[f.settle > 0]
fig = fig_for(1120, 200)
ax = fig.subplots()
for d, col, lab in [("2017-11-03", BLUE, "Calm: 3 Nov 2017 (contango)"), ("2020-03-16", ORANGE, "Crisis: 16 Mar 2020 (backwardation)")]:
    x = f[f.trade_date == d].sort_values("expiry")
    ax.plot(x.days_to_expiry, x.settle, color=col, lw=2.2, marker="o", ms=4, label=lab)
ax.set_xlim(0, 265)
ax.set_ylim(0, 80)
ax.legend(frameon=False, fontsize=13, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_xlabel("days to expiry", fontsize=13)
ax.set_ylabel("VIX future", fontsize=13)
tidy(ax)
fig.tight_layout()
save(fig, "s08_term_structure")

# ---- Slide 12: cumulative share of VIX^2 by strike (box 1608 x 280)
fig = fig_for(1608, 280)
ax = fig.subplots()
for term, col in [("near", BLUE), ("next", ORANGE)]:
    x = k[(k.term == term)].sort_values("strike")
    fwd = x.loc[x.option == "put/call avg", "strike"].iloc[0]
    share = x.contribution.cumsum() / x.contribution.sum()
    ax.plot(x.strike / fwd, share * 100, color=col, lw=2.2, label=f"{term} term ({x.expiry.iloc[0]})")
ax.axvline(1, color=MUTED, lw=0.9, ls=":")
ax.text(1.005, 8, "at the money", fontsize=12, color=MUTED)
ax.set_xlabel("strike / K₀", fontsize=13)
ax.set_ylabel("% of VIX²", fontsize=13)
ax.set_xlim(0.45, 1.25)
ax.legend(frameon=False, fontsize=12, loc="upper left")
tidy(ax)
fig.tight_layout()
save(fig, "s12_strike_contributions")

# ---- Slide 17: cumulative P&L of one long front-month contract (box 870 x 610)
fig = fig_for(870, 610)
ax = fig.subplots()
pnl = fm["cum_points"] * 1000 / 1000  # in $ thousands per contract
ax.plot(fm.index, pnl, color=BLUE, lw=1.8)
ax.axhline(0, color=MUTED, lw=0.9)
ax.set_ylabel("cumulative P&L, $ thousands\n(one contract, $1,000 per point)")
ax.text(pd.Timestamp("2023-01-01"), -127, f"{fm['cum_points'].iloc[-1]:.0f} points ≈ −${-fm['cum_points'].iloc[-1]:.0f}k", ha="center", va="center", fontsize=14, color=INK)
for d, lab in [("2018-02-05", "Feb 2018"), ("2020-03-16", "Mar 2020")]:
    ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.9, ls=":")
    ax.text(pd.Timestamp(d), ax.get_ylim()[1], lab, fontsize=12, color=MUTED, ha="center", va="bottom")
tidy(ax)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
save(fig, "s17_long_futures_pnl")
