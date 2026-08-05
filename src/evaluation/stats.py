"""
Statistical significance testing (README §7.5):
  - Bootstrap resampling (Koehn, 2004) for automatic-metric significance.
  - Wilcoxon signed-rank / Mann-Whitney for ordinal human ratings.

Report non-significant differences plainly, per the README -- don't
overstate small-sample results.
"""
import random

import numpy as np


def bootstrap_significance(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap resampling (Koehn 2004): resample the same indices
    from both systems' per-item scores, and report how often system A beats
    system B under resampling as an empirical p-value proxy."""
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must be the same length (paired, same test items)")
    n = len(scores_a)
    if n == 0:
        return {"n": 0, "note": "no items to compare"}

    rng = random.Random(seed)
    a_wins = 0
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        mean_a = sum(scores_a[i] for i in idx) / n
        mean_b = sum(scores_b[i] for i in idx) / n
        diffs.append(mean_a - mean_b)
        if mean_a > mean_b:
            a_wins += 1

    observed_diff = sum(scores_a) / n - sum(scores_b) / n
    p_a_better = a_wins / n_resamples
    return {
        "n": n,
        "n_resamples": n_resamples,
        "observed_mean_diff_a_minus_b": observed_diff,
        "bootstrap_ci_95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_system_a_better": p_a_better,
        "significant_at_0.05": p_a_better >= 0.975 or p_a_better <= 0.025,
    }


def wilcoxon_signed_rank(ratings_a: list[float], ratings_b: list[float]) -> dict:
    """Paired ordinal human ratings, e.g. same items rated for system A vs B."""
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(ratings_a, ratings_b)
    except ValueError as e:
        return {"n": len(ratings_a), "note": f"wilcoxon not computable: {e}"}
    return {"n": len(ratings_a), "statistic": float(stat), "p_value": float(p), "significant_at_0.05": bool(p < 0.05)}


def mann_whitney(ratings_a: list[float], ratings_b: list[float]) -> dict:
    """Unpaired ordinal human ratings, e.g. different raters/items per system."""
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(ratings_a, ratings_b, alternative="two-sided")
    return {
        "n_a": len(ratings_a), "n_b": len(ratings_b),
        "statistic": float(stat), "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


if __name__ == "__main__":
    import json
    demo_a = [0.72, 0.65, 0.80, 0.55, 0.90, 0.60]
    demo_b = [0.60, 0.58, 0.70, 0.50, 0.75, 0.55]
    print(json.dumps(bootstrap_significance(demo_a, demo_b, n_resamples=500), indent=2))
    print(json.dumps(wilcoxon_signed_rank([4, 5, 3, 4, 5], [3, 4, 3, 3, 4]), indent=2))
