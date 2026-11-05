"""
Take a snapshot of SPX options for the VIX calculation.

Saves every SPX/SPXW expiration 20-40 days out (the VIX uses the two that bracket 30 days),
plus the SPX level and the official VIX value at the same moment, so you can compare your
calculated VIX against Cboe's.

Output:  data/options/spx_options_<YYYYMMDD_HHMM>.csv

Run:     python3.11 snapshot_spx_options.py
Tonight: quotes will be thin/stale - fine for building the cleaning code.
Tomorrow: run again between ~11am and 3pm ET for the real snapshot.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

MIN_DAYS, MAX_DAYS = 20, 40
OUT_DIR = Path("data") / "options"
TZ = "America/New_York"


def last_price(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="5d", interval="1m")
    if hist.empty:
        hist = yf.Ticker(ticker).history(period="5d")
    return float(hist["Close"].dropna().iloc[-1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap_time = pd.Timestamp.now(tz=TZ)

    spot = last_price("^SPX")
    vix = last_price("^VIX")

    spx = yf.Ticker("^SPX")
    frames = []
    for exp in spx.options:
        expiry = pd.Timestamp(exp).tz_localize(TZ)
        dte = (expiry - snap_time).days
        if MIN_DAYS <= dte <= MAX_DAYS:
            chain = spx.option_chain(exp)
            frames.append(chain.calls.assign(expiration=exp, type="call"))
            frames.append(chain.puts.assign(expiration=exp, type="put"))

    if not frames:
        raise RuntimeError("No SPX expirations found 20-40 days out - try again in a minute.")

    opts = pd.concat(frames, ignore_index=True)
    opts["root"] = opts["contractSymbol"].str.extract(r"^([A-Z]+)", expand=False)  # SPX (AM) vs SPXW (PM)
    opts["snapshot_time"] = snap_time.isoformat()
    opts["spot"] = spot
    opts["official_vix"] = vix

    out = OUT_DIR / f"spx_options_{snap_time:%Y%m%d_%H%M}.csv"
    opts.to_csv(out, index=False)

    usable = (opts["bid"] > 0).sum()
    print(f"Snapshot time : {snap_time:%Y-%m-%d %H:%M} ET")
    print(f"SPX level     : {spot:,.2f}")
    print(f"Official VIX  : {vix:.2f}   <- compare your calculated VIX to this")
    print(f"Options saved : {len(opts):,}  ({usable:,} with a bid above zero)")
    print("\nContracts per expiration and root:")
    print(opts.groupby(["expiration", "root"]).size().to_string())
    print(f"\nSaved to {out}")
    if usable < len(opts) * 0.3:
        print("\nNote: most bids are zero - normal outside market hours. "
              "Take the real snapshot tomorrow between ~11am and 3pm ET.")


if __name__ == "__main__":
    main()
