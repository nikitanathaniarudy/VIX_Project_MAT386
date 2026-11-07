"""
Figures for the Discussion section (Josh's Figures 1-3).

Run from the project root:   python3.11 report/make_discussion_figures.py

  Figure 1  VIX and S&P 500, 2010-2026                 data/market_history.csv
  Figure 2  VIXY (long VIX-futures ETF) decay vs VIX    data/etp_history.csv  (downloaded on first run)
  Figure 3  Volmageddon, Jan-Mar 2018                   data/market_history.csv, data/vix_front_month.csv,
                                                        data/etp_history.csv (SVXY panel, if available)

Writes PDF + PNG to report/figures/. Figure 2 and the SVXY panel need Yahoo Finance (yfinance);
if the download fails, those parts are skipped and the script says so.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e6e5e0", "#f0efec"
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


def load_etps():
    """VIXY and SVXY daily closes (split-adjusted); downloaded once and cached."""
    path = DATA / "etp_history.csv"
    if not path.exists():
        try:
            import yfinance as yf
            d = yf.download(["VIXY", "SVXY"], start="2011-01-01", auto_adjust=True, progress=False)["Close"]
            d = d.dropna(how="all")
            if d.empty:
                raise RuntimeError("empty download")
            d.index.name = "date"
            d.to_csv(path)
            print("downloaded", path)
        except Exception as err:  # noqa: BLE001
            print(f"Could not download VIXY/SVXY ({type(err).__name__}: {err}); skipping Figure 2 and the SVXY panel.")
            return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


m = pd.read_csv(DATA / "market_history.csv", index_col=0, parse_dates=True)
fm = pd.read_csv(DATA / "vix_front_month.csv", index_col=0, parse_dates=True)
etp = load_etps()
EVENTS = [("2018-02-05", "Volmageddon\nFeb 2018"), ("2020-03-16", "COVID\nMar 2020")]

# ---------------------------------------------------------------- Figure 1
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
a1.plot(m.index, m["SPX"], color=BLUE, lw=1.2)
a1.set_ylabel("S&P 500")
a2.plot(m.index, m["VIX"], color=ORANGE, lw=1.0)
a2.set_ylabel("VIX")
a2.axhline(m["VIX"].mean(), color=MUTED, lw=0.8, ls="--", label=f"average {m['VIX'].mean():.1f}")
a2.legend(frameon=False, fontsize=8, loc="upper right")
for d, lab in EVENTS:
    for ax in (a1, a2):
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.7, ls=":")
    v = m.loc[:d, "VIX"].iloc[-1] if d != "2020-03-16" else m.loc["2020-03-16", "VIX"]
    a2.annotate(lab, xy=(pd.Timestamp(d), v), xytext=(8, -2), textcoords="offset points", fontsize=8, color=INK, va="top")
for ax in (a1, a2):
    tidy(ax)
a2.xaxis.set_major_locator(mdates.YearLocator(2))
a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.align_ylabels()
fig.tight_layout()
save(fig, "vix_vs_spx")

# ---------------------------------------------------------------- Figure 2
if etp is not None and "VIXY" in etp:
    v = etp["VIXY"].dropna()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
    a1.plot(v.index, v, color=BLUE, lw=1.2)
    a1.set_yscale("log")
    a1.set_ylabel("VIXY price\n(split-adjusted, log scale)")
    vix = m["VIX"].loc[v.index.min():]
    a2.plot(vix.index, vix, color=ORANGE, lw=1.0)
    a2.set_ylabel("VIX")
    loss = 1 - v.iloc[-1] / v.iloc[0]
    a1.text(0.98, 0.9, f"{loss:.1%} lost since {v.index[0]:%b %Y}", transform=a1.transAxes, ha="right", fontsize=9, color=INK)
    for ax in (a1, a2):
        tidy(ax)
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, "vixy_vs_vix")

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
