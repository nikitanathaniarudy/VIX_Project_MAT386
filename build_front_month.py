"""
Build the continuous front-month VIX futures series used by the hedging backtest.

Roll rule (from the decisions doc): hold the nearest monthly VX contract and switch to the
next one at the close, 5 trading days before the held contract expires.

Input:   data/vix_futures_all.csv      (from fetch_vix_futures.py)
Output:  data/vix_front_month.csv      one row per trading day:

    date              trading day
    held_expiry       contract held at the END of this day (after any roll)
    held_settle       its settle price today
    pnl_points        price change of the contract held OVERNIGHT into today
                      (the P&L per contract in VIX points; x 1,000 for dollars)
    return            pnl_points / yesterday's settle of that same contract
    is_roll_day       True if we switched contracts at today's close
    roll_gap          on roll days: next contract's price minus the old one
                      (positive = contango, i.e. we pay up to roll)
    front_settle      nearest unexpired contract (for term-structure charts)
    second_settle     second-nearest contract
    days_to_expiry    calendar days until the held contract expires
    cum_points        running total of pnl_points (a continuous price path)

Run:   python3.11 build_front_month.py
"""

import pandas as pd

from params import (BACKTEST_END, BACKTEST_START, FRONT_MONTH_FILE, FUTURES_FILE,
                    ROLL_DAYS_BEFORE_EXPIRY)


def load_futures() -> pd.DataFrame:
    fut = pd.read_csv(FUTURES_FILE, parse_dates=["trade_date", "expiry"])
    fut = fut[fut["settle"] > 0]
    # Drop the expiry-day row: the final settlement uses a special morning quote,
    # and we always roll out 5 days earlier anyway.
    fut = fut[fut["trade_date"] < fut["expiry"]]
    return fut


def build(fut: pd.DataFrame) -> pd.DataFrame:
    settle = fut.pivot_table(index="trade_date", columns="expiry", values="settle")
    dates = settle.index
    expiries = sorted(settle.columns)

    # Roll date for each contract = the trading day ROLL_DAYS_BEFORE_EXPIRY before its expiry
    roll_date = {}
    for exp in expiries:
        if exp > dates[-1]:
            roll_date[exp] = pd.Timestamp.max    # expires after our data ends: never rolled out of
            continue
        idx = dates.searchsorted(exp) - ROLL_DAYS_BEFORE_EXPIRY   # 5 trading days before expiry
        roll_date[exp] = dates[idx] if idx >= 0 else None          # None = too early to use

    rows = []
    held_prev = None
    for i, d in enumerate(dates):
        # Contract held at the end of day d: the nearest one whose roll date is still after d
        held = next((e for e in expiries if roll_date[e] is not None and roll_date[e] > d), None)

        pnl = ret = None
        if held_prev is not None and i > 0:
            today = settle.at[d, held_prev] if held_prev in settle.columns else None
            yday = settle.at[dates[i - 1], held_prev]
            if pd.notna(today) and pd.notna(yday):
                pnl = today - yday
                ret = pnl / yday

        is_roll = held_prev is not None and held != held_prev
        roll_gap = None
        if is_roll and held is not None:
            roll_gap = settle.at[d, held] - settle.at[d, held_prev]

        live = settle.loc[d].dropna()
        live = live[live.index > d]
        rows.append({
            "date": d,
            "held_expiry": held,
            "held_settle": settle.at[d, held] if held is not None else None,
            "pnl_points": pnl,
            "return": ret,
            "is_roll_day": is_roll,
            "roll_gap": roll_gap,
            "front_settle": live.iloc[0] if len(live) > 0 else None,
            "second_settle": live.iloc[1] if len(live) > 1 else None,
            "days_to_expiry": (held - d).days if held is not None else None,
        })
        held_prev = held

    out = pd.DataFrame(rows).set_index("date")
    out = out.loc[BACKTEST_START:BACKTEST_END]
    out["pnl_points"] = out["pnl_points"].astype(float)
    out["cum_points"] = out["pnl_points"].fillna(0).cumsum()
    return out


def main() -> None:
    fut = load_futures()
    out = build(fut)
    out.to_csv(FRONT_MONTH_FILE)

    rolls = out[out["is_roll_day"]]
    missing = out["pnl_points"].isna().sum()
    contango_share = (rolls["roll_gap"] > 0).mean()
    print(f"Saved {FRONT_MONTH_FILE}")
    print(f"Days: {len(out):,}   From {out.index.min():%Y-%m-%d} to {out.index.max():%Y-%m-%d}")
    print(f"Roll days: {len(rolls)} (expect about 12 per year)")
    print(f"Days with missing P&L: {missing} (should be 0 or 1)")
    print(f"Rolls in contango (next contract dearer): {contango_share:.0%}")
    print(f"Average roll gap: {rolls['roll_gap'].mean():+.2f} VIX points")
    print(f"Total P&L of 1 contract, long the whole period: {out['pnl_points'].sum():+.1f} points "
          f"(= ${out['pnl_points'].sum() * 1000:+,.0f})")
    print("\nSanity check - Volmageddon (5 Feb 2018):")
    print(out.loc["2018-02-01":"2018-02-07", ["held_expiry", "held_settle", "pnl_points", "front_settle"]]
          .to_string())


if __name__ == "__main__":
    main()