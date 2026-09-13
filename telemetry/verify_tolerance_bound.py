"""
Verifies: nonparametric_upper_tolerance_bound() in stats_primitives.py.

Claim being tested: for a sample of n draws from ANY continuous
distribution, using the maximum (m=1) as the upper bound, the reported
`achieved_confidence` gamma is the true probability that the bound covers
at least `content_p` of the population.

Method: repeat the experiment thousands of times on THREE very different
distributions (normal, exponential, uniform) - the distribution-free
property means gamma should hold regardless of shape. For each trial we
know the population exactly (we generated it), so we can directly check
whether the computed bound truly covers content_p of it, and compare the
empirical hit-rate to the claimed gamma.
"""
import numpy as np
from math import comb

from stats_primitives import nonparametric_upper_tolerance_bound


def true_coverage(bound, dist_name, params):
    """Exact CDF(bound) for the population, i.e. true proportion covered."""
    if dist_name == 'normal':
        from scipy.stats import norm
        return norm.cdf(bound, *params)
    if dist_name == 'exponential':
        from scipy.stats import expon
        return expon.cdf(bound, *params)
    if dist_name == 'uniform':
        from scipy.stats import uniform
        return uniform.cdf(bound, *params)
    raise ValueError(dist_name)


def run_experiment(dist_name, sampler, params, n, content_p, m, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    gammas = []
    for _ in range(trials):
        sample = sampler(rng, n)
        bound, gamma, n_used = nonparametric_upper_tolerance_bound(sample, content_p, m)
        gammas.append(gamma)
        if true_coverage(bound, dist_name, params) >= content_p:
            hits += 1
    empirical_rate = hits / trials
    claimed_gamma = gammas[0]  # gamma depends only on n, p, m - identical every trial
    return empirical_rate, claimed_gamma


def main():
    n, content_p, m = 8, 0.99, 1
    print(f"Config: n={n} samples, content_p={content_p}, m={m} (bound = max of n)\n")
    print(f"{'distribution':<14}{'claimed gamma':>16}{'empirical rate':>18}{'trials':>10}")

    experiments = [
        ('normal', lambda rng, n: rng.normal(0, 1, n), (0, 1)),
        ('exponential', lambda rng, n: rng.exponential(2.0, n), (0, 2.0)),
        ('uniform', lambda rng, n: rng.uniform(0, 5, n), (0, 5)),
    ]

    all_ok = True
    for dist_name, sampler, params in experiments:
        empirical, claimed = run_experiment(dist_name, sampler, params, n, content_p, m)
        # Distribution-free tolerance bounds guarantee gamma is EXACT (not just
        # a lower bound), so empirical rate should match claimed gamma closely
        # for a large number of trials, regardless of distribution shape.
        ok = abs(empirical - claimed) < 0.02
        all_ok &= ok
        flag = "OK" if ok else "MISMATCH"
        print(f"{dist_name:<14}{claimed:>16.4f}{empirical:>18.4f}{'20000':>10}   [{flag}]")

    print()
    if all_ok:
        print("PASS: empirical coverage matches the claimed exact confidence across all "
              "three distributions, confirming the bound's distribution-free property.")
    else:
        print("FAIL: empirical coverage diverged from the claimed confidence - "
              "investigate nonparametric_upper_tolerance_bound().")

    # Sanity check on the closed-form formula itself for m=1: gamma should equal 1 - p^n exactly
    p, nn = 0.99, 8
    formula_direct = 1 - p ** nn
    from stats_primitives import nonparametric_upper_tolerance_bound as bound_fn
    _, gamma_fn, _ = bound_fn(np.arange(nn), p, 1)
    print(f"\nClosed-form cross-check (m=1): 1 - p^n = {formula_direct:.6f}, "
          f"function returned {gamma_fn:.6f}  ->  {'MATCH' if abs(formula_direct-gamma_fn)<1e-12 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
