"""
Figures for the Discussion section (Josh's Figures 1-3).

Run from the project root:   python3.11 report/make_discussion_figures.py
                             python3.11 report/make_discussion_figures.py --fetch   (refresh data from Yahoo first)

  Figure 1  S&P 500 and VIX, Jan 2005 - Nov 2026        data/discussion_history.csv
  Figure 2  VXX decay vs a mean-reverting VIX, 2018-2026 data/discussion_history.csv
  Figure 3  Volmageddon, Jan-Mar 2018                    data/market_history.csv, data/vix_front_month.csv,
                                                         data/etp_history.csv (SVXY panel)

data/discussion_history.csv holds daily closes of the S&P 500, the VIX and VXX (split-adjusted).
From 2010 the S&P 500 and VIX columns are copied from data/market_history.csv. The 2005-2009 values
and the VXX column were read back from the vector paths of Josh's original Figures 1 and 2 (Yahoo
Finance data); they match market_history.csv where the two overlap (median error 0.003 VIX points).
Run with --fetch on a machine with internet access to replace the file with a fresh Yahoo download.

Writes PDF + PNG to report/figures/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NAVY, RED = "#1e3a8a", "#b91c1c"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def tidy(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


def fetch_history():
    """Download S&P 500, VIX and VXX closes from Yahoo Finance into data/discussion_history.csv."""
    import yfinance as yf
    d = yf.download(["^GSPC", "^VIX", "VXX"], start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    d = d.dropna(how="all").rename(columns={"^GSPC": "SPX", "^VIX": "VIX"})[["SPX", "VIX", "VXX"]]
    if d.empty:
        raise RuntimeError("empty download")
    d.index.name = "date"
    d.to_csv(DATA / "discussion_history.csv")
    print("downloaded", DATA / "discussion_history.csv")


if "--fetch" in sys.argv:
    fetch_history()

h = pd.read_csv(DATA / "discussion_history.csv", index_col=0, parse_dates=True)
m = pd.read_csv(DATA / "market_history.csv", index_col=0, parse_dates=True)
fm = pd.read_csv(DATA / "vix_front_month.csv", index_col=0, parse_dates=True)
etp_path = DATA / "etp_history.csv"
etp = pd.read_csv(etp_path, index_col=0, parse_dates=True) if etp_path.exists() else None

# ---------------------------------------------------------------- Figures 1 and 2 (Josh's design)
# Same layout, colours, titles and labels as Josh's originals; only the label positions differ,
# so that no text overlaps the title, the lines or the panel borders.
JOSH = {"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#475569",
        "axes.labelcolor": "#0b0b0b", "xtick.color": "#0b0b0b", "ytick.color": "#0b0b0b"}
J_NAVY, J_RED, J_BLUE, J_LINERED = "#1e3a8a", "#b91c1c", "#4338ca", "#dc2626"
BOX_RED = dict(boxstyle="round,pad=0.25", fc="#fdecec", ec=J_LINERED, lw=0.7)
BOX_GREY = dict(boxstyle="round,pad=0.3", fc="white", ec="#94a3b8", lw=0.8)
BOX_BLUE = dict(boxstyle="round,pad=0.35", fc="#eef0ff", ec="#6366f1", lw=0.8)


def josh_grid(ax):
    ax.grid(True, which="major", color="#e5e7eb", lw=0.5, ls="--")


with plt.rc_context(JOSH):
    # ------------------------------------------------------------ Figure 1
    spx, vix = h["SPX"].dropna(), h["VIX"].dropna()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    a1.plot(spx.index, spx, color=J_NAVY, lw=1.0, label="S&P 500 Index (SPX)")
    a1.set_ylabel("S&P 500 Level")
    a1.legend(loc="upper left", fontsize=8)
    a1.set_title("S&P 500 Index and Cboe Volatility Index (VIX), 2005–Present", fontsize=10, fontweight="bold")
    a2.plot(vix.index, vix, color=J_RED, lw=0.7, label="Cboe VIX")
    a2.axhline(vix.mean(), color="#475569", lw=0.9, ls=":", label=f"Long-run Mean (~{vix.mean():.1f})")
    a2.set_ylabel("VIX Index Points")
    a2.set_ylim(0, 112)                      # headroom so the labels sit inside the panel
    a2.legend(loc="upper center", bbox_to_anchor=(0.40, 1.0), fontsize=8)
    events = [("2008-11-20", "2008 GFC", 97), ("2018-02-05", "Feb 2018\n'Volmageddon'", 62),
              ("2020-03-16", "Mar 2020 COVID", 97)]
    for d, lab, ybox in events:
        t = pd.Timestamp(d)
        for ax in (a1, a2):
            ax.axvline(t, color=J_LINERED, lw=0.7, ls="--", alpha=0.7)
        peak = vix.loc[t - pd.Timedelta(days=5): t + pd.Timedelta(days=5)].max()
        text = f"{lab}\n(VIX: {peak:.1f})" if peak > 50 else lab
        a2.annotate(text, xy=(t, peak), xytext=(t, ybox), ha="center", va="center", fontsize=7,
                    fontweight="bold", bbox=BOX_RED,
                    arrowprops=dict(arrowstyle="->", color=J_LINERED, lw=0.7, shrinkA=0, shrinkB=1))
    for ax in (a1, a2):
        josh_grid(ax)
    a2.xaxis.set_major_locator(mdates.YearLocator(3, month=1, day=1))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a2.set_xlim(spx.index[0], spx.index[-1])
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, "vix_vs_spx")

    # ------------------------------------------------------------ Figure 2
    v = h["VXX"].dropna()
    vx = h["VIX"].loc[v.index[0]:].dropna()
    fig, ax1 = plt.subplots(figsize=(8.0, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(v.index, v, color=J_BLUE, lw=1.4, label="VXX (Split-Adj. Price, Log Scale)", zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(v.min() * 0.7, v.max() * 3.5)   # headroom above the COVID spike for its label
    ax1.set_ylabel("VXX Price (USD, Log Scale)", color=J_BLUE, fontweight="bold")
    ax1.tick_params(axis="y", colors=J_BLUE)
    ax2.plot(vx.index, vx, color=J_LINERED, lw=0.9, alpha=0.75, label="Spot VIX (Linear Scale)")
    ax2.set_ylim(0, 90)
    ax2.set_ylabel("Spot VIX Index Level", color=J_LINERED, fontweight="bold")
    ax2.tick_params(axis="y", colors=J_LINERED)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.set_title("VXX Structural Decay vs. Mean-Reverting Spot VIX (2018–Present)", fontsize=11,
                  fontweight="bold")
    peak_day = v.idxmax()
    ax1.annotate(f"COVID Spike\n(\\${v.max():,.0f})", xy=(peak_day, v.max()), xytext=(28, 4),
                 textcoords="offset points", ha="left", va="center", fontsize=8, bbox=BOX_GREY,
                 arrowprops=dict(arrowstyle="->", color="#475569", lw=0.8))
    loss = v.iloc[-1] / v.iloc[0] - 1
    ax1.annotate(f"{v.index[0]:%b %Y}: \\${v.iloc[0]:,.0f}\n{v.index[-1]:%b %Y}: \\${v.iloc[-1]:.2f}\n"
                 f"Total Return: {loss:.1%}", xy=(v.index[-1], v.iloc[-1]), xytext=(0.80, 0.69),
                 textcoords="axes fraction", ha="center", va="center", fontsize=8, bbox=BOX_BLUE,
                 arrowprops=dict(arrowstyle="->", color=J_BLUE, lw=0.8))
    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    josh_grid(ax1)
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlim(v.index[0], v.index[-1] + pd.Timedelta(days=20))
    fig.tight_layout()
    save(fig, "vxx_vs_vix")

# ---------------------------------------------------------------- Figure 3
start, end = "2018-01-15", "2018-03-15"
spot = m.loc[start:end, "VIX"]
fut = fm.loc[start:end, "front_settle"]
has_svxy = etp is not None and "SVXY" in etp and etp["SVXY"].loc[start:end].notna().sum() > 10
fig, axes = plt.subplots(2 if has_svxy else 1, 1, figsize=(6.6, 4.4 if has_svxy else 3.0), sharex=True)
a1 = axes[0] if has_svxy else axes
a1.plot(spot.index, spot, color=ORANGE, lw=1.4, label="VIX (spot)")
a1.plot(fut.index, fut, color=BLUE, lw=1.4, label="Front-month VIX future")
a1.axvline(pd.Timestamp("2018-02-05"), color=MUTED, lw=0.7, ls=":")
d0, d1 = pd.Timestamp("2018-02-02"), pd.Timestamp("2018-02-05")
a1.annotate(f"future {fut[d0]:.1f} → {fut[d1]:.1f}\n(+{fut[d1] / fut[d0] - 1:.0%} in one day)",
            xy=(d1, fut[d1]), xytext=(pd.Timestamp("2018-02-17"), 33), fontsize=8, color=INK, va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a1.set_ylabel("VIX points")
a1.legend(frameon=False, fontsize=8, loc="upper left")
tidy(a1)
if has_svxy:
    s = etp["SVXY"].loc[start:end]
    a2 = axes[1]
    a2.plot(s.index, s / s.iloc[0] * 100, color=AQUA, lw=1.4)
    a2.axvline(pd.Timestamp("2018-02-05"), color=MUTED, lw=0.7, ls=":")
    a2.set_ylabel("SVXY (inverse ETF)\nindexed, 15 Jan = 100")
    tidy(a2)
(axes[-1] if has_svxy else a1).xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.align_ylabels()
fig.tight_layout()
save(fig, "volmageddon")
