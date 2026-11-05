"""
Download daily history for the VIX project:
  3. VIX index, S&P 500 index and SPY (dividend-adjusted) closes  -> data/market_history.csv
  4. 1-month and 3-month US T-bill yields (as decimals)          -> data/risk_free.csv
     Tries FRED first; if FRED is slow or down, uses Treasury.gov, then Yahoo ^IRX.
 
Run:   python3 -m pip install yfinance pandas
       python3 fetch_history.py
"""
 
import io
from pathlib import Path
 
import pandas as pd
import requests  # installed with yfinance; uses its own certificates, which avoids SSL errors on macOS
import yfinance as yf
 
START = "2010-01-01"
OUT_DIR = Path("data")
 
TICKERS = {"^VIX": "VIX", "^GSPC": "SPX", "SPY": "SPY"}
FRED_SERIES = ["DGS1MO", "DGS3MO"]
 
 
def download_market_history() -> pd.DataFrame:
    """Daily closes for VIX, SPX and SPY. SPY is adjusted for dividends (total return)."""
    raw = yf.download(list(TICKERS), start=START, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data - check your internet connection and try again.")
 
    closes = raw["Close"].rename(columns=TICKERS)[list(TICKERS.values())]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes.index.name = "date"
    # Keep only days when the stock market traded (drops a few stray VIX-only dates)
    return closes.dropna(subset=["SPX", "SPY"])
 
 
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}
 
 
def _get_text(url: str, attempts: int = 2, timeout: int = 60) -> str:
    """GET a URL with a browser-like header, retrying once if it is slow."""
    last_err = None
    for _ in range(attempts):
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as err:
            last_err = err
    raise last_err
 
 
def _rates_from_fred() -> pd.DataFrame:
    """Source 1: FRED series DGS1MO and DGS3MO."""
    rates = {}
    for series in FRED_SERIES:
        text = _get_text(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
        df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True)
        rates[series] = pd.to_numeric(df.iloc[:, 0], errors="coerce")  # holidays are blank or "."
    return pd.DataFrame(rates)
 
 
def _rates_from_treasury() -> pd.DataFrame:
    """Source 2: US Treasury daily yield curve (the same numbers FRED republishes), one CSV per year."""
    base = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
    frames = []
    for year in range(int(START[:4]), pd.Timestamp.today().year + 1):
        url = (f"{base}/{year}/all?type=daily_treasury_yield_curve"
               f"&field_tdr_date_value={year}&page&_format=csv")
        df = pd.read_csv(io.StringIO(_get_text(url)))
        df.columns = [c.strip().strip('"') for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])
        frames.append(df.set_index("Date")[["1 Mo", "3 Mo"]])
    out = pd.concat(frames).sort_index()
    out.columns = FRED_SERIES
    return out.apply(pd.to_numeric, errors="coerce")
 
 
def _rates_from_yahoo() -> pd.DataFrame:
    """Source 3 (last resort): Yahoo's 13-week T-bill rate ^IRX, used for both 1M and 3M."""
    irx = yf.download("^IRX", start=START, auto_adjust=False, progress=False)["Close"].squeeze()
    irx.index = pd.to_datetime(irx.index).tz_localize(None)
    return pd.DataFrame({"DGS1MO": irx, "DGS3MO": irx})
 
 
def download_risk_free() -> pd.DataFrame:
    """1M and 3M T-bill yields as decimals. Tries FRED, then Treasury.gov, then Yahoo."""
    sources = [
        ("FRED", _rates_from_fred),
        ("US Treasury (home.treasury.gov)", _rates_from_treasury),
        ("Yahoo ^IRX (13-week rate, used for both columns)", _rates_from_yahoo),
    ]
    for name, fetch in sources:
        try:
            rates = fetch()
            if rates.dropna(how="all").empty:
                raise ValueError("no data returned")
            print(f"  Got rates from {name}")
            break
        except Exception as err:  # noqa: BLE001 - any failure means "try the next source"
            print(f"  {name} failed ({type(err).__name__}: {err}); trying next source...")
    else:
        raise RuntimeError("Could not download T-bill rates from any source.")
 
    rf = rates.loc[rates.index >= START].sort_index().ffill() / 100
    rf = rf.round(6)
    rf.index.name = "date"
    return rf
 
 
def summarize(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"Rows: {len(df):,}   From {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    print("Last 3 rows:")
    print(df.tail(3).to_string())
 
 
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
 
    print("Downloading VIX / S&P 500 / SPY from Yahoo Finance...")
    market = download_market_history()
    market.to_csv(OUT_DIR / "market_history.csv")
    summarize("market_history.csv", market)
 
    print("\nDownloading T-bill yields from FRED...")
    rf = download_risk_free()
    rf.to_csv(OUT_DIR / "risk_free.csv")
    summarize("risk_free.csv (decimals, e.g. 0.042 = 4.2%)", rf)
 
    print(f"\nDone. Files saved in ./{OUT_DIR}/")
 
 
if __name__ == "__main__":
    main()
 