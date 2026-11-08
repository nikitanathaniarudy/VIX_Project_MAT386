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

# ---------------------------------------------------------------- Figure 1
spx, vix = h["SPX"].dropna(), h["VIX"].dropna()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 4.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
a1.plot(spx.index, spx, color=NAVY, lw=1.0)
a1.set_ylabel("S&P 500")
a1.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
a2.plot(vix.index, vix, color=RED, lw=0.7)
a2.axhline(vix.mean(), color=MUTED, lw=0.9, ls="--", label=f"average VIX ({vix.mean():.1f})")
a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.42, 1.0))
a2.set_ylabel("VIX")
a2.set_ylim(0, 100)
events = [("2008-11-20", "Nov 2008\nfinancial crisis", -6, "right"),
          ("2018-02-05", "Feb 2018\nVolmageddon", -6, "right"),
          ("2020-03-16", "Mar 2020\nCOVID", 6, "left")]
for d, lab, dx, ha in events:
    t = pd.Timestamp(d)
    for ax in (a1, a2):
        ax.axvline(t, color=MUTED, lw=0.7, ls=":")
    peak = vix.loc[t - pd.Timedelta(days=5): t + pd.Timedelta(days=5)].max()
    y = max(peak, 45) + 4
    a2.annotate(f"{lab}\n(VIX {peak:.1f})" if peak > 50 else lab, xy=(t, y), xytext=(dx, 0),
                textcoords="offset points", fontsize=8, color=INK, ha=ha, va="center")
for ax in (a1, a2):
    tidy(ax)
a2.xaxis.set_major_locator(mdates.YearLocator(3, month=1, day=1))
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
a2.set_xlim(pd.Timestamp("2005-01-01"), vix.index[-1] + pd.Timedelta(days=60))
fig.align_ylabels()
fig.tight_layout()
save(fig, "vix_vs_spx")

# ---------------------------------------------------------------- Figure 2
v = h["VXX"].dropna()
vx = h["VIX"].loc[v.index[0]:].dropna()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
a1.plot(v.index, v, color=BLUE, lw=1.1)
a1.set_yscale("log")
a1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"\\${x:,.0f}"))
a1.set_ylabel("VXX price\n(split-adjusted, log scale)")
peak_day = v.idxmax()
a1.annotate(f"COVID spike, {peak_day:%d %b %Y}: \\${v.max():,.0f}", xy=(peak_day, v.max()), xytext=(10, -2),
            textcoords="offset points", fontsize=8, color=INK, va="top")
loss = 1 - v.iloc[-1] / v.iloc[0]
a1.text(0.03, 0.06, f"{v.index[0]:%b %Y}: \\${v.iloc[0]:,.0f}   →   {v.index[-1]:%b %Y}: \\${v.iloc[-1]:,.2f}"
        f"   (−{loss:.1%})", transform=a1.transAxes, fontsize=8.5, color=INK)
a2.plot(vx.index, vx, color=RED, lw=0.8)
a2.axhline(vx.mean(), color=MUTED, lw=0.9, ls="--", label=f"average VIX ({vx.mean():.1f})")
a2.legend(frameon=False, fontsize=8, loc="upper right")
a2.set_ylabel("VIX")
for ax in (a1, a2):
    tidy(ax)
a2.xaxis.set_major_locator(mdates.YearLocator(1))
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.align_ylabels()
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
