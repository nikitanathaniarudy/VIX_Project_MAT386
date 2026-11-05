"""
Check vix_calc.py against the worked example in the Cboe VIX White Paper (2019),
"The VIX Index Calculation: Step-by-Step" (pages 5-9) and Appendix 1 (pages 13-16).

The option tables from Appendix 1 are in references/cboe_example_near.csv and
references/cboe_example_next.csv (strike, call bid/ask, put bid/ask).

Run:   python3.11 check_cboe_example.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from params import MINUTES_PER_YEAR
from vix_calc import expiry_variance, interpolate_30d

REF = Path("references")

# Inputs stated in the white paper (page 5): calculation at 9:46 a.m. ET
MINUTES = {"near": 854 + 510 + 34_560,      # standard SPX, expires at 9:30 a.m. ET
           "next": 854 + 900 + 44_640}      # SPX Weekly, expires at 4:00 p.m. ET
RATES = {"near": 0.000305, "next": 0.000286}

# Cboe's published answers (pages 5-9)
EXPECTED = {
    "near": {"T": 0.0683486, "F": 1962.89996, "K0": 1960, "sigma2": 0.01846292},
    "next": {"T": 0.0882686, "F": 1962.40006, "K0": 1960, "sigma2": 0.01882101},
    "vix": 13.685821,
}


def load_chain(path: Path) -> pd.DataFrame:
    ch = pd.read_csv(path).set_index("strike").sort_index()
    for side in ["c", "p"]:
        b, a = ch[f"{side}_bid"], ch[f"{side}_ask"]
        ok = (b > 0) & (a >= b)
        ch[f"{side}_mid"] = np.where(ok, (b + a) / 2, np.nan)
    return ch


def check(label: str, ours: float, theirs: float, tol: float) -> bool:
    ok = abs(ours - theirs) <= tol
    print(f"  {label:<8} ours {ours:>14.8f}   Cboe {theirs:>14.8f}   {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    results, all_ok = {}, True
    for term in ["near", "next"]:
        ch = load_chain(REF / f"cboe_example_{term}.csv")
        T = MINUTES[term] / MINUTES_PER_YEAR
        r = expiry_variance(ch, T, RATES[term])
        r.update({"minutes": MINUTES[term], "T": T})
        results[term] = r

        exp = EXPECTED[term]
        print(f"{term.upper()}-TERM ({r['n_puts']} puts, 1 at K0, {r['n_calls']} calls)")
        all_ok &= check("T", T, exp["T"], 5e-7)
        all_ok &= check("F", r["F"], exp["F"], 5e-5)
        all_ok &= check("K0", r["K0"], exp["K0"], 0)
        all_ok &= check("sigma^2", r["sigma2"], exp["sigma2"], 5e-8)
        print()

    vix, (w1, w2) = interpolate_30d(results["near"], results["next"])
    print(f"30-day weights: near {w1:.6f}, next {w2:.6f}")
    all_ok &= check("VIX", vix, EXPECTED["vix"], 5e-5)
    print("\nAll checks passed: vix_calc.py reproduces Cboe's worked example."
          if all_ok else "\nSome checks failed - see MISMATCH lines above.")


if __name__ == "__main__":
    main()
