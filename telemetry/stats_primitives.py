"""
Provably-grounded statistical primitives used by the telemetry gatekeeper.
No dependency on fastf1 - this module is pure numpy/scipy so it can be
unit-tested and Monte-Carlo-verified in isolation from any telemetry data.

Every function documents the exact published method it implements and the
guarantee that method carries. None of the numeric constants here were
chosen to make a particular dataset look right - they are the values given
in the cited references.
"""

from math import comb

import numpy as np
from scipy import stats


def t_interval_for_mean(samples, confidence=0.95):
    """
    Standard Student-t confidence interval for the mean of n INDEPENDENT
    samples of an unknown-variance quantity:

        mean   = sample mean
        se     = sample_std / sqrt(n)     (ddof=1)
        CI     = mean +/- t_(n-1, alpha/2) * se

    This is elementary, textbook inference and carries the standard t-CI
    guarantee ONLY WHEN THE SAMPLES ARE INDEPENDENT DRAWS OF THE SAME
    QUANTITY. In this codebase it is deliberately applied to one
    ratio-estimate per lap (not to within-lap points), because within-lap
    telemetry samples are serially correlated (each is a running
    cumulative sum) and therefore violate the independence assumption -
    see NOTE_on_within_lap_regression below and verify_calibration.py,
    which demonstrates by Monte Carlo that naive per-point OLS pooling
    produces badly overconfident ("anti-conservative") intervals for
    exactly this reason.

    Returns (mean, se, (ci_low, ci_high), n_used).
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return None, None, None, n
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    tcrit = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    return mean, se, ci, n


NOTE_on_within_lap_regression = """
Why calibration uses ONE ratio per lap, not per-telemetry-point OLS:

cum_xy (cumulative path length in raw X/Y units) is a running sum of noisy
per-step distances. Even if the underlying per-sample position noise were
independent, cum_xy's noise is an integrated ("random walk") process:
Var(cum_xy_i) grows with i, and cum_xy_i, cum_xy_j are highly correlated
for nearby i, j. A standard OLS-through-origin standard-error formula
assumes i.i.d. residuals and, when applied to this data, understates the
true uncertainty of the slope substantially - verify_calibration.py shows
this empirically (a nominal 95% CI covered the true value only ~4-8% of
the time). Collapsing each lap to a single ratio (total path length over
total distance) sidesteps the correlation problem entirely: the n
per-lap ratios ARE independent draws (different laps, independent noise
realizations), so the ordinary Student-t interval over those n ratios is
valid. This is a case where the simpler statistic is the correct one.
"""


def modified_z_scores(values):
    """
    Iglewicz & Hoaglin (1993) modified z-score:
        M_i = 0.6745 * (x_i - median(x)) / MAD(x)
    0.6745 makes MAD a consistent estimator of sigma for a normal
    population; the conventional flag threshold is |M| > 3.5. Both values
    are from that reference, not tuned per-dataset.
    """
    x = np.asarray(values, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def tukey_upper_fence(values, k=3.0):
    """
    Tukey (1977) fence: Q3 + k * IQR.
    k=1.5 -> conventional "mild outlier" fence.
    k=3.0 -> conventional "far outlier" fence (used here, since discarding
    a legitimate reference lap is the costlier error for this use case).
    """
    x = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return float(q3 + k * iqr)


def nonparametric_upper_tolerance_bound(samples, content_p, m=1):
    """
    Distribution-free one-sided upper tolerance bound (Wilks, 1941; Hahn &
    Meeker, "Statistical Intervals", Ch. 5).

    Uses the m-th largest of n observed samples as the bound. Requires no
    assumption about the shape of the underlying distribution, only that
    it is continuous. Carries an EXACT, closed-form confidence that it
    covers at least `content_p` of the population:

        gamma = 1 - sum_{i=0}^{m-1} C(n,i) * p^(n-i) * (1-p)^i

    Returns (bound_value, achieved_confidence, n_used).
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n < 1 or not (0 < content_p < 1) or m < 1 or m > n:
        return None, None, n
    bound = float(x[n - m])
    gamma = 1.0 - sum(
        comb(n, i) * content_p ** (n - i) * (1 - content_p) ** i for i in range(m)
    )
    return bound, float(gamma), n
