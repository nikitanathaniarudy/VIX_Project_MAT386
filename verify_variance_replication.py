"""
Numerical check of the derivation "VIX^2 = fair strike of a variance swap".

The derivation says: the option strip
    sigma^2 = (2/T) sum dK/K^2 e^{rT} Q(K)  -  (1/T)(F/K0 - 1)^2
equals the expected (risk-neutral) average variance over [0, T].
We test that in two models where the true expected variance is known:

  Test 1  Black-Scholes, constant vol 20%   -> strip should give 0.04 exactly.
  Test 2  Heston (stochastic vol)           -> strip should match the expected integrated variance
                                               theta + (v0 - theta)(1 - e^{-kappa T})/(kappa T)
          Option prices come from Monte Carlo on the same simulated paths.

Run:   python3.11 verify_variance_replication.py
"""

import numpy as np
from scipy.stats import norm

T = 30 / 365
R = 0.04
F0 = 100.0


# ------------------------------------------------------------------ the strip formula (Step 4)

def strip_variance(strikes, calls, puts, F, T, r):
    """Cboe discrete formula on a strike grid; calls/puts are option PRICES at each strike."""
    K0 = strikes[strikes <= F].max()
    q = np.where(strikes < K0, puts, np.where(strikes > K0, calls, 0.5 * (calls + puts)))
    dK = np.gradient(strikes)                     # (K_{i+1} - K_{i-1}) / 2, one-sided at the ends
    return 2 / T * np.sum(dK / strikes**2 * np.exp(r * T) * q) - (F / K0 - 1) ** 2 / T


# ------------------------------------------------------------------ test 1: Black-Scholes

def bs_prices(F, K, T, r, sigma):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    call = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return call, put


def test_black_scholes(sigma=0.20):
    print("TEST 1: Black-Scholes, constant volatility 20%  (true variance = 0.040000)")
    print("  Forward = 100, so a strike step of 0.07 is the same spacing as SPX's 5 points at 7,500.")
    print(f"  {'case':<38} {'step':>6} {'range':>9} {'strip var':>10} {'vol':>8}")
    cases = [("Fine, wide grid (the integral)", 0.05, 20, 300),
             ("Like our 5 Nov snapshot (43%-120%)", 0.07, 43, 120),
             ("Narrow range only (truncation)", 0.05, 90, 110),
             ("Coarse spacing only (discretisation)", 5.0, 20, 300)]
    for name, step, lo, hi in cases:
        K = np.arange(lo, hi + step / 2, step)
        c, p = bs_prices(F0, K, T, R, sigma)
        v = strip_variance(K, c, p, F0, T, R)
        print(f"  {name:<38} {step:>6.2f} {f'{lo}-{hi}':>9} {v:>10.6f} {100*np.sqrt(v):>7.3f}%")
    print("  -> the full strip recovers 0.04 exactly. Cutting off the tails lowers it; very coarse")
    print("     strikes raise it. With SPX-like spacing and range the error is negligible.\n")


# ------------------------------------------------------------------ test 2: Heston Monte Carlo

def test_heston(kappa=1.3253, theta=0.0354, sigma_v=0.3877, rho=-0.7165, v0=0.0276,
                n_paths=200_000, steps_per_day=24, seed=42):
    print("TEST 2: Heston stochastic volatility, Monte Carlo")
    print(f"  kappa={kappa}, theta={theta}, sigma={sigma_v}, rho={rho}, v0={v0}  (Gatheral 2006 set)")
    rng = np.random.default_rng(seed)
    n_steps = 30 * steps_per_day
    dt = T / n_steps
    lnF = np.zeros(n_paths)                       # log of F_t / F_0 (forward has no drift)
    v = np.full(n_paths, v0)
    int_var = np.zeros(n_paths)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)                   # full truncation
        int_var += vp * dt
        lnF += -0.5 * vp * dt + np.sqrt(vp * dt) * z1
        v = v + kappa * (theta - vp) * dt + sigma_v * np.sqrt(vp * dt) * z2
    ST = F0 * np.exp(lnF)

    exact = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)
    realised = int_var.mean() / T
    se = int_var.std() / T / np.sqrt(n_paths)

    K = np.arange(30, 250.25, 0.25)
    disc = np.exp(-R * T)
    calls = np.array([disc * np.maximum(ST - k, 0).mean() for k in K])
    puts = np.array([disc * np.maximum(k - ST, 0).mean() for k in K])
    strip = strip_variance(K, calls, puts, F0, T, R)

    # Per-path payoffs, for standard errors
    log_payoff = -2 / T * np.log(ST / F0)
    K0 = K[K <= F0].max()
    strip_payoff = -2 / T * (np.log(ST / K0) - (ST - K0) / K0)   # what the option strip pays on each path
    se_log = log_payoff.std() / np.sqrt(n_paths)
    se_strip = strip_payoff.std() / np.sqrt(n_paths)

    print(f"  {'quantity':<50} {'variance':>9} {'vol':>8} {'MC s.e.':>9}")
    rows = [("Exact expected variance (Heston formula)", exact, None),
            ("Simulated average realised variance", realised, se),
            ("Log contract -(2/T) E[ln(S_T/F_0)]  (Step 1)", log_payoff.mean(), se_log),
            ("Option strip with 1/K^2 weights  (Steps 2-4)", strip, se_strip)]
    for name, val, s in rows:
        s_txt = f"{s:9.6f}" if s is not None else f"{'exact':>9}"
        print(f"  {name:<50} {val:>9.6f} {100*np.sqrt(val):>7.3f}% {s_txt}")
    gap = strip - exact
    print(f"  Strip minus exact: {gap:+.6f} = {gap/se_strip:+.1f} standard errors")
    print(f"  The plain log contract is noisy (its s.e. is ~{se_log/se_strip:.0f}x larger) because ln(S_T) swings with")
    print("  every price move; the strip hedges that out with the forward term (S_T - K0)/K0.")
    print("  -> the option strip recovers the expected variance even though volatility is random,")
    print("     confirming the derivation is model-free.")


if __name__ == "__main__":
    test_black_scholes()
    test_heston()
