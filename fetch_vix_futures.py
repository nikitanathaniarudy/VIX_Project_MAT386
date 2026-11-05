"""
Download daily settlement data for every MONTHLY VIX futures contract (VX) from Cboe
and combine them into one file for the backtest.

Output:
  data/vix_futures/raw/VX_<expiry>.csv   one file per contract, exactly as Cboe publishes it
  data/vix_futures_all.csv               all contracts stacked, clean column names

Run:   python3.11 fetch_vix_futures.py
       (safe to re-run: files already downloaded are skipped)
"""

import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# Contract months to download: Jan 2015 through Feb 2027
FIRST_MONTH = (2015, 1)
LAST_MONTH = (2027, 2)

OUT_DIR = Path("data")
RAW_DIR = OUT_DIR / "vix_futures" / "raw"

CDN = "https://cdn.cboe.com/"
FILE_PATH = "data/us/futures/market_statistics/historical_data/VX/VX_{expiry}.csv"
LIST_URL = "https://www.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/json,*/*",
}


# ---------------------------------------------------------------- expiry dates

def months_between(first, last):
    y, m = first
    while (y, m) <= last:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def third_friday(year, month):
    d = date(year, month, 15)                      # the 3rd Friday is always the 15th-21st
    return d + timedelta(days=(4 - d.weekday()) % 7)


def rule_expiry(year, month):
    """Monthly VX expiry: the Wednesday 30 days before the 3rd Friday of the NEXT month.
    (If a holiday interferes, Cboe moves it 1 business day earlier; handled when downloading.)"""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return third_friday(ny, nm) - timedelta(days=30)


def expiries_from_cboe_list(session):
    """Ask Cboe's own product list for the monthly contracts. Returns {(year, month): (expiry, url)}."""
    resp = session.get(LIST_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    found = {}
    for items in resp.json().values():
        for it in items:
            if it.get("duration_type") != "M":       # "M" = monthly, "W" = weekly
                continue
            exp = date.fromisoformat(it["expire_date"][:10])
            path = it.get("path") or FILE_PATH.format(expiry=exp.isoformat())
            found[(exp.year, exp.month)] = (exp, CDN + path.lstrip("/"))
    return found


# ---------------------------------------------------------------- downloading

def fetch_csv(session, url):
    resp = session.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    if "Trade Date" not in resp.text[:300]:            # got a web page instead of a CSV
        return None
    return resp.text


def download_contract(session, year, month, listed):
    """Download one contract month. Returns (expiry, status)."""
    if (year, month) in listed:
        candidates = [listed[(year, month)]]
    else:
        base = rule_expiry(year, month)
        # Try the rule date first, then up to 3 days earlier (holiday adjustments)
        candidates = [(base - timedelta(days=k), None) for k in range(4)]

    for exp, url in candidates:
        target = RAW_DIR / f"VX_{exp.isoformat()}.csv"
        if target.exists():
            return exp, "already had it"
        url = url or CDN + FILE_PATH.format(expiry=exp.isoformat())
        text = fetch_csv(session, url)
        time.sleep(0.3)                                  # be polite to Cboe's server
        if text:
            target.write_text(text)
            return exp, "downloaded"
    return None, "NOT FOUND"


# ---------------------------------------------------------------- combining

def combine_raw_files():
    frames = []
    for f in sorted(RAW_DIR.glob("VX_*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["expiry"] = pd.to_datetime(f.stem.replace("VX_", ""))
        frames.append(df)

    allf = pd.concat(frames, ignore_index=True)
    allf = allf.rename(columns={"futures": "contract", "total_volume": "volume"})
    allf["trade_date"] = pd.to_datetime(allf["trade_date"])
    for col in ["open", "high", "low", "close", "settle", "volume", "open_interest"]:
        if col in allf:
            allf[col] = pd.to_numeric(allf[col], errors="coerce")

    allf = allf[allf["settle"] > 0]                       # drop empty rows
    allf["days_to_expiry"] = (allf["expiry"] - allf["trade_date"]).dt.days
    keep = ["trade_date", "expiry", "contract", "open", "high", "low", "close",
            "settle", "volume", "open_interest", "days_to_expiry"]
    allf = allf[[c for c in keep if c in allf.columns]]
    return allf.sort_values(["trade_date", "expiry"]).reset_index(drop=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    try:
        listed = expiries_from_cboe_list(session)
        print(f"Got Cboe's contract list ({len(listed)} monthly contracts listed).")
    except Exception as err:  # noqa: BLE001
        listed = {}
        print(f"Couldn't read Cboe's contract list ({type(err).__name__}); "
              "using the expiry-date rule instead.")

    months = list(months_between(FIRST_MONTH, LAST_MONTH))
    print(f"Downloading {len(months)} monthly VIX futures contracts...\n")

    missing = []
    for i, (y, m) in enumerate(months, 1):
        exp, status = download_contract(session, y, m, listed)
        label = date(y, m, 1).strftime("%b %Y")
        print(f"  [{i:3d}/{len(months)}] {label}: {status}" + (f" (expires {exp})" if exp else ""))
        if exp is None:
            missing.append(label)

    have = list(RAW_DIR.glob("VX_*.csv"))
    if not have:
        print("\nNo files downloaded. Cboe may have changed its download links. "
              "Right-click one contract link on the Cboe page, copy the link address, "
              "and send it to Claude so the script can be updated.")
        return

    allf = combine_raw_files()
    allf.to_csv(OUT_DIR / "vix_futures_all.csv", index=False)

    print(f"\n=== vix_futures_all.csv ===")
    print(f"Contracts: {allf['expiry'].nunique()}   Rows: {len(allf):,}   "
          f"From {allf['trade_date'].min():%Y-%m-%d} to {allf['trade_date'].max():%Y-%m-%d}")
    print(allf.tail(3).to_string(index=False))
    if missing:
        print(f"\nCould not find {len(missing)} contract(s): {', '.join(missing)}. "
              "Run the script again; if they're still missing, download those from the Cboe page by hand.")
    else:
        print("\nAll contracts downloaded.")


if __name__ == "__main__":
    main()
