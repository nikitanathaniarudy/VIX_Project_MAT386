"""
Checks for heston_vix.py. Run from the project root:   python3.11 -m pytest heston
"""
import numpy as np
import pytest

from heston_vix import (DAY, call_price, exact_mean_variance, exact_mean_vix_squared, simulate_variance,
                        variance_from_vix, vix_from_variance)

KAPPA, THETA, SIGMA = 1.3253, 0.0354, 0.3877
SPOT_VIX = 20.83
N_PATHS = 100_000
RATE = 0.04


@pytest.fixture(scope="module")
def v0():
    return variance_from_vix(SPOT_VIX, KAPPA, THETA)


@pytest.fixture(scope="module")
def paths(v0):
    return simulate_variance(v0, KAPPA, THETA, SIGMA, N_PATHS, 180, seed=42)


def standard_error(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def test_model_vix_today_equals_spot_vix(paths):
    vix_today = vix_from_variance(paths[0], KAPPA, THETA)
    assert np.allclose(vix_today, SPOT_VIX)


@pytest.mark.parametrize("days", [30, 91, 180])
def test_mean_variance_matches_exact_formula(paths, v0, days):
    exact = exact_mean_variance(v0, KAPPA, THETA, days * DAY)
    assert abs(paths[days].mean() - exact) < 3 * standard_error(paths[days])


@pytest.mark.parametrize("days", [30, 91, 180])
def test_mean_vix_squared_matches_exact_formula(paths, v0, days):
    vix_squared = vix_from_variance(paths[days], KAPPA, THETA) ** 2
    exact = exact_mean_vix_squared(v0, KAPPA, THETA, days * DAY)
    assert abs(vix_squared.mean() - exact) < 3 * standard_error(vix_squared)


def test_convexity_adjustment_is_never_negative(paths, v0):
    days = np.arange(1, 181)
    mean_vix = vix_from_variance(paths[days], KAPPA, THETA).mean(axis=1)
    exact_root = np.sqrt(exact_mean_vix_squared(v0, KAPPA, THETA, days * DAY))
    assert np.all(exact_root - mean_vix >= 0)


@pytest.mark.parametrize("days", [30, 91])
def test_call_prices_fall_as_strike_rises(paths, days):
    vix = vix_from_variance(paths[days], KAPPA, THETA)
    prices = [call_price(vix, strike, RATE, days * DAY)[0] for strike in [15, 20, 25, 30]]
    assert np.all(np.diff(prices) < 0)


@pytest.mark.parametrize("days", [30, 91])
@pytest.mark.parametrize("strike", [15, 20, 25, 30])
def test_call_is_worth_at_least_discounted_futures_minus_strike(paths, days, strike):
    vix = vix_from_variance(paths[days], KAPPA, THETA)
    lower_bound = np.exp(-RATE * days * DAY) * max(vix.mean() - strike, 0)
    assert call_price(vix, strike, RATE, days * DAY)[0] >= lower_bound


def test_out_of_the_money_calls_rise_with_maturity(paths):
    vix_1m = vix_from_variance(paths[30], KAPPA, THETA)
    vix_3m = vix_from_variance(paths[91], KAPPA, THETA)
    for strike in [20, 25, 30]:
        assert call_price(vix_3m, strike, RATE, 91 * DAY)[0] > call_price(vix_1m, strike, RATE, 30 * DAY)[0]


def test_doubling_the_paths_moves_prices_by_less_than_005(paths, v0):
    more_paths = simulate_variance(v0, KAPPA, THETA, SIGMA, 2 * N_PATHS, 106, seed=7)
    for days in [15, 43, 78, 106]:
        futures = vix_from_variance(paths[days], KAPPA, THETA).mean()
        futures_more = vix_from_variance(more_paths[days], KAPPA, THETA).mean()
        assert abs(futures - futures_more) < 0.05
    for days in [30, 91]:
        for strike in [15, 20, 25, 30]:
            call = call_price(vix_from_variance(paths[days], KAPPA, THETA), strike, RATE, days * DAY)[0]
            call_more = call_price(vix_from_variance(more_paths[days], KAPPA, THETA), strike, RATE, days * DAY)[0]
            assert abs(call - call_more) < 0.05
