"""
Compute the VIX from an SPX option snapshot using the Cboe methodology, and compare it with the
official VIX recorded at the same moment.

Run:   python3.11 vix_calc.py                                   (uses the newest snapshot)
       python3.11 vix_calc.py data/options/spx_options_20261105_1141.csv

Outputs:
  printout of every intermediate number (T, R, F, K0, sigma^2, VIX)
  data/vix_calc_strikes.csv    every strike used, with its contribution (for the strike-contribution chart)

Steps (Cboe Volatility Index Mathematics Methodology):
  1. Pick the near- and next-term Friday expirations with more than 23 and fewer than 37 days to go.
  2. Time to expiry T in minutes / 525,600; risk-free rate R interpolated for each expiry.
  3. Forward F = K* + e^(RT) (C - P) at the strike K* where |call mid - put mid| is smallest.
  4. K0 = first strike at or below F.
  5. Out-of-the-money puts below K0 and calls above K0 (average of both at K0); skip zero bids and
     stop after two zero bids in a row.
  6. sigma^2 = (2/T) sum( dK/K^2 e^(RT) Q(K) ) - (1/T) (F/K0 - 1)^2
  7. Interpolate the two variances to 30 days; VIX = 100 * sqrt(annualized 30-day variance).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from params import MINUTES_PER_YEAR, OPTIONS_DIR, RISK_FREE_FILE, VIX_MAX_DAYS, VIX_MIN_DAYS

TZ = "America/New_York"
N30 = 30 * 1440          # minutes in 30 days
N365 = 365 * 1440        # minutes in a year (= MINUTES_PER_YEAR)


# ------------------------------------------------------------------ loading

def latest_snapshot() -> Path:
    files = sorted(OPTIONS_DIR.glob("spx_options_*.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshot found in {OPTIONS_DIR}. Run snapshot_spx_options.py first.")
    return files[-1]


def load_snapshot(path: Path) -> tuple[pd.DataFrame, pd.Timestamp, float, float]:
    o = pd.read_csv(path)
    snap = pd.Timestamp(o["snapshot_time"].iloc[0]).tz_convert(TZ)
    spot = float(o["spot"].iloc[0])
    official = float(o["official_vix"].iloc[0])
    o["expiration"] = pd.to_datetime(o["expiration"])
    for c in ["strike", "bid", "ask"]:
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o[["bid", "ask"]] = o[["bid", "ask"]].fillna(0.0)
    return o, snap, spot, official


def is_third_friday(d: pd.Timestamp) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def settlement_time(exp: pd.Timestamp) -> pd.Timestamp:
    """Standard monthly SPX (3rd Friday) settles AM, 9:30 ET; weekly SPXW settles PM, 4:00 ET."""
    hour, minute = (9, 30) if is_third_friday(exp) else (16, 0)
    return pd.Timestamp(exp.year, exp.month, exp.day, hour, minute, tz=TZ)


def risk_free_rate(snap: pd.Timestamp, days: float) -> float:
    """Treasury 1M and 3M yields (bond-equivalent), linearly interpolated to `days`,
    converted to a continuously compounded rate as Cboe does."""
    rf = pd.read_csv(RISK_FREE_FILE, index_col=0, parse_dates=True)
    row = rf.loc[rf.index <= snap.tz_localize(None).normalize()].iloc[-1]
    y1, y3 = row["DGS1MO"], row["DGS3MO"]
    bey = float(np.interp(days, [30, 91], [y1, y3]))      # flat outside 30-91 days
    apy = (1 + bey / 2) ** 2 - 1
    return float(np.log(1 + apy))


# ------------------------------------------------------------------ one expiry

def chain_for(o: pd.DataFrame, exp: pd.Timestamp) -> pd.DataFrame:
    """Calls and puts for one expiry side by side, one row per strike, using the right root."""
    root = "SPX" if is_third_friday(exp) else "SPXW"
    sub = o[(o["expiration"] == exp) & (o["root"] == root)]
    if sub.empty:                                        # fall back to whatever root is there
        sub = o[o["expiration"] == exp]
    calls = sub[sub["type"] == "call"].set_index("strike")[["bid", "ask"]].add_prefix("c_")
    puts = sub[sub["type"] == "put"].set_index("strike")[["bid", "ask"]].add_prefix("p_")
    ch = calls.join(puts, how="outer").sort_index()
    ch = ch[~ch.index.duplicated()]
    for side in ["c", "p"]:
        b, a = ch[f"{side}_bid"], ch[f"{side}_ask"]
        ok = (b > 0) & (a >= b)
        ch[f"{side}_mid"] = np.where(ok, (b + a) / 2, np.nan)
    return ch


def select_otm(side_bids: pd.Series, side_mids: pd.Series, strikes_in_order) -> list:
    """Walk away from K0; skip zero bids; stop after two zero bids in a row."""
    chosen, zeros = [], 0
    for k in strikes_in_order:
        bid = side_bids.get(k, np.nan)
        if not (bid > 0) or np.isnan(side_mids.get(k, np.nan)):
            zeros += 1
            if zeros == 2:
                break
            continue
        zeros = 0
        chosen.append(k)
    return chosen


def expiry_variance(ch: pd.DataFrame, T: float, R: float) -> dict:
    growth = np.exp(R * T)

    both = ch.dropna(subset=["c_mid", "p_mid"])
    if both.empty:
        raise ValueError("No strike has both a call and a put quote - cannot find the forward.")
    diff = (both["c_mid"] - both["p_mid"]).abs()
    k_star = diff.idxmin()
    F = k_star + growth * (both.at[k_star, "c_mid"] - both.at[k_star, "p_mid"])

    below = ch.index[ch.index <= F]
    if below.empty:
        raise ValueError("No strike at or below the forward.")
    K0 = below.max()

    puts = select_otm(ch["p_bid"], ch["p_mid"], sorted(ch.index[ch.index < K0], reverse=True))
    calls = select_otm(ch["c_bid"], ch["c_mid"], sorted(ch.index[ch.index > K0]))

    rows = [(k, "put", ch.at[k, "p_mid"]) for k in puts]
    k0_mid = np.nanmean([ch.at[K0, "p_mid"], ch.at[K0, "c_mid"]])
    rows.append((K0, "put/call avg", k0_mid))
    rows += [(k, "call", ch.at[k, "c_mid"]) for k in calls]
    tab = pd.DataFrame(rows, columns=["strike", "option", "mid"]).sort_values("strike").reset_index(drop=True)

    k = tab["strike"].to_numpy()
    dk = np.empty_like(k)
    dk[1:-1] = (k[2:] - k[:-2]) / 2
    dk[0], dk[-1] = k[1] - k[0], k[-1] - k[-2]
    tab["dK"] = dk
    tab["contribution"] = dk / k**2 * growth * tab["mid"].to_numpy()

    sigma2 = 2 / T * tab["contribution"].sum() - (F / K0 - 1) ** 2 / T
    return {"F": F, "K_star": k_star, "K0": K0, "sigma2": sigma2, "table": tab,
            "n_puts": len(puts), "n_calls": len(calls)}


# ------------------------------------------------------------------ main

def compute_vix(o: pd.DataFrame, snap: pd.Timestamp) -> dict:
    exps = sorted(o["expiration"].unique())
    cands = []
    for e in exps:
        e = pd.Timestamp(e)
        if e.weekday() != 4:                 # VIX uses Friday expirations only
            continue
        minutes = (settlement_time(e) - snap).total_seconds() / 60
        if VIX_MIN_DAYS * 1440 < minutes < VIX_MAX_DAYS * 1440:
            cands.append((e, minutes))
    if len(cands) < 2:
        raise ValueError(f"Need two Friday expirations 23-37 days out; found {[str(c[0].date()) for c in cands]}.")
    (e1, n1), (e2, n2) = cands[0], cands[1]

    out = {}
    for label, e, n in [("near", e1, n1), ("next", e2, n2)]:
        T = n / MINUTES_PER_YEAR
        R = risk_free_rate(snap, n / 1440)
        res = expiry_variance(chain_for(o, e), T, R)
        res.update({"expiry": e, "minutes": n, "T": T, "R": R})
        out[label] = res

    out["vix"], out["weights"] = interpolate_30d(out["near"], out["next"])
    return out


def interpolate_30d(near: dict, nxt: dict) -> tuple[float, tuple[float, float]]:
    """Step 3: weight the two variances to exactly 30 days and turn into a VIX level."""
    n1, n2 = near["minutes"], nxt["minutes"]
    w1 = (n2 - N30) / (n2 - n1)
    w2 = (N30 - n1) / (n2 - n1)
    var30 = (near["T"] * near["sigma2"] * w1 + nxt["T"] * nxt["sigma2"] * w2) * N365 / N30
    return 100 * np.sqrt(var30), (w1, w2)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_snapshot()
    o, snap, spot, official = load_snapshot(path)
    res = compute_vix(o, snap)

    print(f"Snapshot: {path.name}   taken {snap:%Y-%m-%d %H:%M} ET   SPX {spot:,.2f}\n")
    for label in ["near", "next"]:
        r = res[label]
        print(f"{label.upper()}-TERM  expiry {r['expiry']:%a %d %b %Y}")
        print(f"  minutes to settlement : {r['minutes']:,.0f}   ({r['minutes']/1440:.2f} days)")
        print(f"  T                     : {r['T']:.6f} years")
        print(f"  R (continuous)        : {r['R']:.6f}")
        print(f"  strike used for F     : {r['K_star']:,.0f}")
        print(f"  forward F             : {r['F']:,.4f}")
        print(f"  K0                    : {r['K0']:,.0f}")
        print(f"  strikes used          : {r['n_puts']} puts, 1 at K0, {r['n_calls']} calls "
              f"({r['table']['strike'].min():,.0f} to {r['table']['strike'].max():,.0f})")
        print(f"  sigma^2               : {r['sigma2']:.6f}   (= {100*np.sqrt(r['sigma2']):.2f}% vol)\n")

    w1, w2 = res["weights"]
    print(f"30-day weights: near {w1:.4f}, next {w2:.4f}")
    print(f"\nCalculated VIX : {res['vix']:.2f}")
    print(f"Official VIX   : {official:.2f}")
    gap = res["vix"] - official
    print(f"Difference     : {gap:+.2f} points  "
          + ("-> within 0.5, success" if abs(gap) <= 0.5 else "-> bigger than 0.5, check the data (see notes)"))

    tabs = []
    for label in ["near", "next"]:
        t = res[label]["table"].copy()
        t.insert(0, "term", label)
        t.insert(1, "expiry", res[label]["expiry"].date())
        tabs.append(t)
    out = Path("data") / "vix_calc_strikes.csv"
    pd.concat(tabs).to_csv(out, index=False)
    print(f"\nPer-strike contributions saved to {out}")


if __name__ == "__main__":
    main()