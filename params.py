"""
Shared project settings - everyone imports from here instead of typing numbers by hand.
Matches the "VIX Project - Team Decisions" doc. Change a value only after agreeing in the group chat.

    from params import *
"""

from pathlib import Path

# ----------------------------------------------------------------- paths
DATA_DIR = Path("data")
MARKET_FILE = DATA_DIR / "market_history.csv"        # VIX, SPX, SPY daily closes
RISK_FREE_FILE = DATA_DIR / "risk_free.csv"          # DGS1MO, DGS3MO as decimals
FUTURES_FILE = DATA_DIR / "vix_futures_all.csv"      # every monthly VX contract, daily
FRONT_MONTH_FILE = DATA_DIR / "vix_front_month.csv"  # rolled front-month series
OPTIONS_DIR = DATA_DIR / "options"                   # SPX option snapshots
FIGURES_DIR = Path("report") / "figures"             # all charts go here (PNG, 300 dpi)

# ----------------------------------------------------------------- general conventions
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_YEAR = 525_600                           # Cboe VIX methodology time unit
FIG_DPI = 300
RANDOM_SEED = 42

# ----------------------------------------------------------------- VIX calculation
VIX_TARGET_DAYS = 30
VIX_MIN_DAYS, VIX_MAX_DAYS = 23, 37                  # expiries used must lie in this range
SNAPSHOT_TIME_ET = "11:42"                           # Thursday 5 Nov 2026

# ----------------------------------------------------------------- backtest
BACKTEST_START = "2015-01-02"
BACKTEST_END = "2026-11-04"
PORTFOLIO_VALUE = 1_000_000                          # USD, invested in SPY (dividend-adjusted)
VX_MULTIPLIER = 1_000                                # USD per VIX point per contract
ROLL_DAYS_BEFORE_EXPIRY = 5                          # trading days
HEDGE_SIZES = [0, 2, 5, 10]                          # contracts per $1M of current portfolio value
HEDGE_BASE_CASE = 5

# ----------------------------------------------------------------- Heston
VIX_TODAY = 16.61                                    # official VIX close, 4 Nov 2026
V0 = (VIX_TODAY / 100) ** 2                          # starting variance, ~0.0276

# Cross-check set: Gatheral (2006), The Volatility Surface, Ch. 3 (SPX fit).
# Verify against the book before citing. v0 replaced by today's VIX^2 so curves are comparable.
HESTON_GATHERAL = {
    "kappa": 1.3253,   # mean reversion speed
    "theta": 0.0354,   # long-run variance
    "sigma": 0.3877,   # vol of vol
    "rho": -0.7165,    # correlation between stock and variance
    "v0": V0,
}

# Main set: estimated from our own data (2015-2026) by Person B on Thursday.
# Fill in once estimated; leave as None until then so nobody uses placeholder numbers by accident.
HESTON_ESTIMATED = {
    "kappa": None,
    "theta": None,
    "sigma": None,
    "rho": None,
    "v0": V0,
}

HESTON_N_PATHS = 100_000
HESTON_DT = 1 / TRADING_DAYS_PER_YEAR                # daily steps, full-truncation Euler
VIX_FUTURES_MONTHS = [1, 2, 3, 4, 5, 6]
VIX_CALL_STRIKES = [15, 20, 25, 30]
VIX_CALL_MONTHS = [1, 3]