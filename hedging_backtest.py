"""
Rolling-n* VIX-futures hedge backtest.

Strategy
--------
Each month, estimate the variance-minimizing hedge ratio from the previous
`window` daily observations of SPY returns and front-month VIX-futures P&L:

    beta_t = Cov(r_SPY, dF) / Var(dF)
    N_t    = max(0, round(-beta_t * S_{t-1} / 1000))

where S_{t-1} is the SPY portfolio value at the previous close and 1000 is the
VX contract multiplier. N_t is held for the month and applied only to future
P&L (no look-ahead).

Accounting
----------
    S_t = S_{t-1} (1 + r_SPY,t)                  SPY portfolio
    C_t = C_{t-1} + N_t * 1000 * pnl_points_t    futures cash account
    Total_t = S_t + C_t

The futures P&L already includes the roll (vix_front_month.csv is rolled
5 trading days before expiry). The roll cost is measured separately as

    roll_cost_t = N_t * 1000 * roll_gap_t

on roll days, and zero otherwise.

Outputs
-------
    data/hedging_backtest_daily.csv    daily series for both windows
    data/hedging_backtest_summary.csv  performance and roll-cost summary
    data/hedging_backtest_episodes.csv crisis-episode returns
    data/hedging_roll_cost_annual.csv  annual roll cost / futures P&L

Run from the project root:
    python hedging_backtest.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from params import (
    BACKTEST_START,
    FRONT_MONTH_FILE,
    MARKET_FILE,
    PORTFOLIO_VALUE,
    TRADING_DAYS_PER_YEAR,
    VX_MULTIPLIER,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

WINDOWS = [252, 504]

EPISODES = {
    "2018 Q1": ("2018-01-02", "2018-03-29"),
    "Volmageddon": ("2018-02-01", "2018-02-09"),
    "2020 Q1": ("2020-01-02", "2020-03-31"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
}


def load_aligned() -> pd.DataFrame:
    """Load SPY and the rolled front-month futures series on one calendar.

    Futures dates that are not NYSE trading days are mapped forward to the next
    market date, so their P&L is not lost.  There are three such dates:
    2015-04-03, 2018-12-05 and 2025-01-09.
    """
    mkt = (
        pd.read_csv(MARKET_FILE, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )
    fm = (
        pd.read_csv(FRONT_MONTH_FILE, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )

    fm = fm[["pnl_points", "is_roll_day", "roll_gap"]].reset_index()
    fm = pd.merge_asof(
        fm.sort_values("date"),
        pd.DataFrame({"date": mkt.index}),
        on="date",
        direction="forward",
    )
    fm["date"] = fm["date"].fillna(mkt.index[-1])

    daily = fm.groupby("date").agg(
        fpnl=("pnl_points", "sum"),
        roll_gap=("roll_gap", "sum"),
        n_rolls=("is_roll_day", "sum"),
    )

    df = mkt.join(daily, how="inner")
    df = df.loc[BACKTEST_START : fm["date"].max()].copy()
    df["ret"] = df["SPY"].pct_change()
    df = df.dropna(subset=["ret", "fpnl"])
    df.index.name = "date"
    return df


def rolling_hedge_ratio(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Monthly rolling variance-minimizing hedge size, estimated only on past data."""
    rets = df["ret"].to_numpy()
    fpnl = df["fpnl"].to_numpy()
    dates = df.index

    beta = np.full(len(df), np.nan)
    n_per_m = np.full(len(df), np.nan)
    contracts = np.zeros(len(df), dtype=int)

    spy_value = float(PORTFOLIO_VALUE)
    last_month: tuple[int, int] | None = None

    for i, d in enumerate(dates):
        month = (d.year, d.month)
        if month != last_month:
            if i >= window:
                y = rets[i - window : i]
                x = fpnl[i - window : i]
                var_x = np.var(x, ddof=1)
                if var_x > 0:
                    b = np.cov(y, x, ddof=1)[0, 1] / var_x
                    beta[i] = b
                    n_per_m[i] = -b * PORTFOLIO_VALUE / VX_MULTIPLIER
                    contracts[i] = max(
                        0, int(round(-b * spy_value / VX_MULTIPLIER))
                    )
            last_month = month
        else:
            beta[i] = beta[i - 1]
            n_per_m[i] = n_per_m[i - 1]
            contracts[i] = contracts[i - 1]

        spy_value *= 1.0 + rets[i]

    return pd.DataFrame(
        {"beta": beta, "N": contracts, "n_per_m": n_per_m}, index=dates
    )


def run_strategy(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Run the rolling hedge and return daily accounting columns."""
    sig = rolling_hedge_ratio(df, window)

    spy = float(PORTFOLIO_VALUE)
    cash = 0.0
    rows = []

    for i, (d, row) in enumerate(df.iterrows()):
        n = int(sig["N"].iloc[i])
        spy *= 1.0 + row["ret"]
        fut_pnl = n * VX_MULTIPLIER * row["fpnl"]
        roll_cost = n * VX_MULTIPLIER * row["roll_gap"]
        cash += fut_pnl

        rows.append(
            {
                "date": d,
                "N": n,
                "beta": sig["beta"].iloc[i],
                "n_per_m": sig["n_per_m"].iloc[i],
                "spy": spy,
                "cash": cash,
                "total": spy + cash,
                "fut_pnl": fut_pnl,
                "roll_cost": roll_cost,
                "gross_price_pnl": fut_pnl + roll_cost,
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    out["hedged_ret"] = out["total"].pct_change()
    out["unhedged_ret"] = out["spy"].pct_change()
    # The first row is the first day's P&L on the initial $1M, not a missing return.
    out.loc[out.index[0], "hedged_ret"] = out["total"].iloc[0] / PORTFOLIO_VALUE - 1
    out.loc[out.index[0], "unhedged_ret"] = out["spy"].iloc[0] / PORTFOLIO_VALUE - 1
    return out


def performance(series: pd.Series, ret: pd.Series, label: str) -> dict:
    """Annualized performance statistics."""
    n_days = int(ret.notna().sum())
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (series.iloc[-1] / PORTFOLIO_VALUE) ** (1 / years) - 1
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    drawdown = series / series.cummax() - 1
    return {
        "strategy": label,
        "final_value": series.iloc[-1],
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "worst_day": ret.min(),
        "n_days": n_days,
    }


def episode_values(series: pd.Series, start: str, end: str) -> tuple[float, float]:
    """Values at the close before `start` and at the close on/before `end`."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    pre = series.loc[:start_ts]
    if len(pre) == 0:
        return np.nan, np.nan
    if pre.index[-1] == start_ts and len(pre) >= 2:
        v0 = float(pre.iloc[-2])
    else:
        v0 = float(pre.iloc[-1])
    v1 = float(series.loc[:end_ts].iloc[-1])
    return v0, v1


def main() -> None:
    df = load_aligned()

    daily = df[["SPY", "ret", "fpnl", "roll_gap", "n_rolls"]].copy()
    daily["spy"] = PORTFOLIO_VALUE * (1 + daily["ret"]).cumprod()

    summaries = []
    episode_rows = []
    annual_frames = []

    for window in WINDOWS:
        res = run_strategy(df, window)
        suffix = f"_{window}"

        daily[f"N{suffix}"] = res["N"]
        daily[f"beta{suffix}"] = res["beta"]
        daily[f"n_per_m{suffix}"] = res["n_per_m"]
        daily[f"cash{suffix}"] = res["cash"]
        daily[f"total{suffix}"] = res["total"]
        daily[f"fut_pnl{suffix}"] = res["fut_pnl"]
        daily[f"roll_cost{suffix}"] = res["roll_cost"]
        daily[f"gross_price_pnl{suffix}"] = res["gross_price_pnl"]
        daily[f"hedged_ret{suffix}"] = res["hedged_ret"]

        perf = performance(res["total"], res["hedged_ret"], f"Rolling n* ({window}d)")
        perf.update(
            {
                "window": window,
                "avg_N": res["N"].mean(),
                "min_N": int(res["N"].min()),
                "max_N": int(res["N"].max()),
                "median_N": res["N"].median(),
                "avg_n_per_m": res["n_per_m"].dropna().mean(),
                "avg_beta": res["beta"].dropna().mean(),
                "roll_cost_total": res["roll_cost"].sum(),
                "fut_pnl_total": res["fut_pnl"].sum(),
                "gross_price_pnl_total": res["gross_price_pnl"].sum(),
                "annual_roll_cost": res["roll_cost"].sum()
                / (perf["n_days"] / TRADING_DAYS_PER_YEAR),
                "avg_portfolio": res["total"].mean(),
                "roll_cost_pct_avg": res["roll_cost"].sum()
                / (perf["n_days"] / TRADING_DAYS_PER_YEAR)
                / res["total"].mean(),
            }
        )
        summaries.append(perf)

        for label, (start, end) in EPISODES.items():
            spy0, spy1 = episode_values(res["spy"], start, end)
            tot0, tot1 = episode_values(res["total"], start, end)
            episode_rows.append(
                {
                    "window": window,
                    "episode": label,
                    "start": start,
                    "end": end,
                    "unhedged_ret": spy1 / spy0 - 1.0,
                    "hedged_ret": tot1 / tot0 - 1.0,
                    "hedge_pnl": (tot1 - tot0) - (spy1 - spy0),
                    "N_at_start": int(
                        res["N"].loc[: pd.Timestamp(start)].iloc[-1]
                    ),
                }
            )

        annual = res.copy()
        annual["year"] = annual.index.year
        annual = annual.groupby("year").agg(
            roll_cost=("roll_cost", "sum"),
            fut_pnl=("fut_pnl", "sum"),
            gross_price_pnl=("gross_price_pnl", "sum"),
            avg_N=("N", "mean"),
            avg_portfolio=("total", "mean"),
        )
        annual["window"] = window
        annual_frames.append(annual.reset_index())

    unhedged_perf = performance(daily["spy"], daily["ret"], "Unhedged SPY")
    summaries.insert(0, unhedged_perf)
    summary = pd.DataFrame(summaries)
    summary["variance_reduction"] = np.nan
    unhedged_var = daily["ret"].var(ddof=1)
    for window in WINDOWS:
        mask = summary["strategy"] == f"Rolling n* ({window}d)"
        summary.loc[mask, "variance_reduction"] = (
            1 - daily[f"hedged_ret_{window}"].var(ddof=1) / unhedged_var
        )

    episodes = pd.DataFrame(episode_rows)
    annual = pd.concat(annual_frames, ignore_index=True)

    DATA.mkdir(exist_ok=True)
    daily.to_csv(DATA / "hedging_backtest_daily.csv", index_label="date")
    summary.to_csv(DATA / "hedging_backtest_summary.csv", index=False)
    episodes.to_csv(DATA / "hedging_backtest_episodes.csv", index=False)
    annual.to_csv(DATA / "hedging_roll_cost_annual.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("Saved:")
    print(f"  {DATA / 'hedging_backtest_daily.csv'}")
    print(f"  {DATA / 'hedging_backtest_summary.csv'}")
    print(f"  {DATA / 'hedging_backtest_episodes.csv'}")
    print(f"  {DATA / 'hedging_roll_cost_annual.csv'}")

    print("\n=== Performance ===")
    cols = [
        "strategy",
        "final_value",
        "cagr",
        "vol",
        "sharpe",
        "max_drawdown",
        "worst_day",
        "variance_reduction",
        "avg_N",
        "min_N",
        "max_N",
        "avg_n_per_m",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== Roll cost (252d) ===")
    r = summary[summary["strategy"] == "Rolling n* (252d)"].iloc[0]
    print(f"  cumulative roll cost: ${r['roll_cost_total']:,.0f}")
    print(f"  cumulative futures P&L: ${r['fut_pnl_total']:,.0f}")
    print(f"  gross price change: ${r['gross_price_pnl_total']:,.0f}")
    print(f"  annual roll cost: ${r['annual_roll_cost']:,.0f}")
    print(f"  annual roll cost / average portfolio: {r['roll_cost_pct_avg']:.2%}")
    print(f"  number of rolls: {int(daily['n_rolls'].sum())}")

    print("\n=== Episodes (252d) ===")
    print(
        episodes[episodes["window"] == 252]
        .drop(columns=["window"])
        .to_string(index=False, float_format=lambda x: f"{x:,.3f}")
    )


if __name__ == "__main__":
    main()
