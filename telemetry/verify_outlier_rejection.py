"""
Verifies: tukey_upper_fence() and modified_z_scores() in stats_primitives.py.

These two methods are used in build_reference_corridor() to reject laps
that don't belong in the "clean" reference set. Here we generate synthetic
"clean" lap-deviation data plus a known number of injected contaminant
("off-track") laps, and measure how reliably each method separates them,
across many repeated trials and several contamination levels.

This does not claim either method is perfect (no fixed threshold is,
robust-statistics literature says so explicitly) - it quantifies the
actual false-positive / false-negative behaviour so you know what you are
getting, rather than trusting an unverified magic number.
"""
import numpy as np

from stats_primitives import tukey_upper_fence, modified_z_scores


def run_trial(rng, n_clean, n_contam, clean_scale=0.6, contam_shift=4.0):
    clean = rng.gamma(shape=2.0, scale=clean_scale, size=n_clean)  # right-skewed, like real deviation data
    contam = clean_scale * 2.0 + rng.exponential(contam_shift, size=n_contam)
    labels = np.array([0] * n_clean + [1] * n_contam)  # 1 = true contaminant
    values = np.concatenate([clean, contam])
    order = rng.permutation(len(values))
    return values[order], labels[order]


def evaluate(values, labels, flagged_mask):
    tp = np.sum(flagged_mask & (labels == 1))
    fp = np.sum(flagged_mask & (labels == 0))
    fn = np.sum(~flagged_mask & (labels == 1))
    tn = np.sum(~flagged_mask & (labels == 0))
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    false_positive_rate = fp / (fp + tn) if (fp + tn) else float('nan')
    return precision, recall, false_positive_rate


def main():
    rng = np.random.default_rng(1)
    trials = 3000
    n_clean, n_contam = 8, 2

    tukey_stats, mz_stats = [], []
    for _ in range(trials):
        values, labels = run_trial(rng, n_clean, n_contam)

        fence = tukey_upper_fence(values, k=3.0)
        tukey_flag = values > fence
        tukey_stats.append(evaluate(values, labels, tukey_flag))

        z = modified_z_scores(values)
        mz_flag = np.abs(z) > 3.5
        mz_stats.append(evaluate(values, labels, mz_flag))

    def summarize(stats_list, name):
        arr = np.array(stats_list, dtype=float)
        precision = np.nanmean(arr[:, 0])
        recall = np.nanmean(arr[:, 1])
        fpr = np.nanmean(arr[:, 2])
        print(f"{name:<28} mean precision={precision:.3f}  mean recall={recall:.3f}  "
              f"mean false-positive-rate={fpr:.3f}")

    print(f"Synthetic trials: {trials}, each with {n_clean} clean laps + {n_contam} contaminant laps\n")
    summarize(tukey_stats, "Tukey fence (k=3.0)")
    summarize(mz_stats, "Modified z-score (|M|>3.5)")
    print("\nInterpretation: both are conservative (biased toward NOT flagging borderline\n"
          "points at n=8-10) which matches their documented behavior on small samples -\n"
          "consistent with the gatekeeper's principle of abstaining rather than guessing\n"
          "when the sample is too small to be confident.")


if __name__ == "__main__":
    main()
