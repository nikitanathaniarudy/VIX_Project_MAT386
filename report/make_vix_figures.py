"""
Figure for the VIX computation section: cumulative share of VIX^2 by strike.

Run from the project root:   python3.11 report/make_vix_figures.py
Reads data/vix_calc_strikes.csv (written by vix_calc.py), writes report/figures/vix_strike_contributions.pdf
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / "data" / "vix_calc_strikes.csv")
FWD = {"near": 7498.0438, "next": 7501.4055}          # forwards printed by vix_calc.py
COLORS = {"near": "#2a78d6", "next": "#eb6834"}
LABELS = {"near": "Near term (4 Dec 2026)", "next": "Next term (11 Dec 2026)"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.axvspan(0.40, 1.0, color="#f0efec", zorder=0)
ax.text(0.70, 0.70, "out-of-the-money puts", ha="center", fontsize=9, color="#52514e")
ax.text(1.12, 0.50, "calls", ha="center", fontsize=9, color="#52514e")
for term in ["near", "next"]:
    g = t[t.term == term].sort_values("strike")
    share = g.contribution.cumsum() / g.contribution.sum()
    x = g.strike / FWD[term]
    ax.plot(x, share, lw=2, color=COLORS[term], label=LABELS[term], zorder=3)
ax.axvline(1.0, color="#52514e", lw=1, ls="--", zorder=2)
ax.axvline(0.9, color="#52514e", lw=1, ls=":", zorder=2)
ax.annotate("about 30% of VIX$^2$ comes from\nstrikes more than 10% below $F$",
            xy=(0.9, 0.30), xytext=(0.45, 0.42), fontsize=9, color="#0b0b0b",
            arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))
ax.text(1.008, 0.40, "$K_0 \\approx F$", fontsize=9, color="#0b0b0b")
ax.set_xlim(0.40, 1.22)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Strike / forward ($K/F$)")
ax.set_ylabel("Cumulative share of VIX$^2$")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=1)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
out = ROOT / "report" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "vix_strike_contributions.pdf")
fig.savefig(out / "vix_strike_contributions.png", dpi=200)
print("saved", out / "vix_strike_contributions.pdf")
