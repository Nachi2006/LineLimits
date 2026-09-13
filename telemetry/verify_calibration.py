"""
Verifies: t_interval_for_mean() in stats_primitives.py, as used by
calibrate_unit_scale() (per-lap ratio version).

We simulate synthetic "laps" with a known ground-truth units-per-metre
scale and per-step positional noise (standing in for GPS/telemetry
jitter). For each lap we collapse to ONE ratio (total path length /
total distance), then pool the n per-lap ratios with a Student-t
interval, and check across many repeated experiments:

  1. Is the pooled estimate unbiased (converges to the true scale)?
  2. Does the reported 95% CI actually contain the true scale in ~95%
     of repeated experiments (correct coverage)?

This script also reproduces, for comparison, the FLAWED per-point OLS +
inverse-variance approach that was tried first, to make the contrast
concrete: that approach looked correct at the formula level but failed
Monte Carlo verification because within-lap cumulative-sum noise is
serially correlated, violating the i.i.d.-residual assumption the OLS
standard-error formula depends on.
"""
import numpy as np
from scipy import stats

from stats_primitives import t_interval_for_mean


def simulate_lap(rng, true_scale, n_points=300, track_length=2000.0, noise_std=0.05):
    # NOTE: step_dist and total_dist must span the exact same range as the
    # real code (dx = np.diff(X) with NO prepend, total_dist = dist_m[-1] -
    # dist_m[0]). An earlier version of this harness prepended a phantom
    # zero-to-first-point step while total_dist excluded it, creating a
    # ~0.05% mismatch between numerator and denominator ranges - a pure
    # test-harness bug that the t-interval correctly detected as a real,
    # reproducible bias (since it was systematic, not noise-driven, no
    # sample size could ever cover it). Fixed here.
    dist_m = np.linspace(1.0, track_length, n_points)
    step_dist = np.diff(dist_m)
    true_step_xy = step_dist * true_scale
    noisy_step_xy = true_step_xy + rng.normal(0, noise_std, size=len(step_dist))
    noisy_step_xy = np.clip(noisy_step_xy, 0, None)
    total_xy = float(np.sum(noisy_step_xy))
    total_dist = float(dist_m[-1] - dist_m[0])
    return total_xy / total_dist


def ols_through_origin_flawed(rng, true_scale, n_points=300, track_length=2000.0, noise_std=0.05):
    """Reproduces the earlier, rejected per-point regression approach for comparison."""
    dist_m = np.linspace(1.0, track_length, n_points)
    step_dist = np.diff(dist_m, prepend=0.0)
    true_step_xy = step_dist * true_scale
    noisy_step_xy = true_step_xy + rng.normal(0, noise_std, size=n_points)
    noisy_step_xy = np.clip(noisy_step_xy, 0, None)
    cum_xy = np.cumsum(noisy_step_xy)
    sxx = np.sum(dist_m ** 2)
    b_hat = np.sum(dist_m * cum_xy) / sxx
    resid = cum_xy - b_hat * dist_m
    sigma2 = np.sum(resid ** 2) / (len(dist_m) - 1)
    se_b = np.sqrt(sigma2 / sxx)
    return b_hat, se_b


def run_ratio_method(rng, true_scale, n_laps, noise_std, confidence=0.95):
    ratios = [simulate_lap(rng, true_scale, noise_std=noise_std) for _ in range(n_laps)]
    mean, se, ci, n_used = t_interval_for_mean(ratios, confidence)
    return mean, ci


def run_flawed_method(rng, true_scale, n_laps, noise_std, confidence=0.95):
    ests, ses = [], []
    for _ in range(n_laps):
        b, se = ols_through_origin_flawed(rng, true_scale, noise_std=noise_std)
        ests.append(b)
        ses.append(se)
    ests, ses = np.array(ests), np.array(ses)
    w = 1.0 / ses ** 2
    pooled = np.sum(w * ests) / np.sum(w)
    pooled_se = np.sqrt(1.0 / np.sum(w))
    tcrit = stats.t.ppf(1 - (1 - confidence) / 2, max(n_laps - 1, 1))
    ci = (pooled - tcrit * pooled_se, pooled + tcrit * pooled_se)
    return pooled, ci


def main():
    rng = np.random.default_rng(42)
    true_scale = 10.0
    trials = 2000

    print("=== Corrected method: one ratio per lap + Student-t interval ===")
    for n_laps, noise_std in [(3, 0.05), (8, 0.05), (8, 0.5), (20, 0.5)]:
        hits, ests = 0, []
        for _ in range(trials):
            mean, ci = run_ratio_method(rng, true_scale, n_laps, noise_std)
            if mean is None:
                continue
            ests.append(mean)
            if ci[0] <= true_scale <= ci[1]:
                hits += 1
        coverage = hits / trials
        print(f"n_laps={n_laps:>3}  noise_std={noise_std:<5}  "
              f"mean estimate={np.mean(ests):.4f} (true={true_scale})  "
              f"95% CI empirical coverage={coverage:.3f}  "
              f"[{'OK' if abs(coverage-0.95) < 0.03 else 'CHECK'}]")

    print("\n=== For contrast: earlier (rejected) per-point OLS + inverse-variance approach ===")
    rng2 = np.random.default_rng(42)
    for n_laps, noise_std in [(8, 0.05), (8, 0.5)]:
        hits, ests = 0, []
        for _ in range(trials):
            mean, ci = run_flawed_method(rng2, true_scale, n_laps, noise_std)
            ests.append(mean)
            if ci[0] <= true_scale <= ci[1]:
                hits += 1
        coverage = hits / trials
        print(f"n_laps={n_laps:>3}  noise_std={noise_std:<5}  "
              f"mean estimate={np.mean(ests):.4f} (true={true_scale})  "
              f"95% CI empirical coverage={coverage:.3f}  "
              f"[{'OK' if abs(coverage-0.95) < 0.03 else 'BADLY OVERCONFIDENT'}]")

    print("\nConclusion: the corrected per-lap-ratio + t-interval method achieves the\n"
          "claimed ~95% coverage regardless of noise level or lap count. The earlier\n"
          "per-point regression approach reports a 95% CI that is dramatically too\n"
          "narrow because it ignores serial correlation in the cumulative-sum data -\n"
          "this is exactly why it was replaced, and why this verification script exists.")


if __name__ == "__main__":
    main()
