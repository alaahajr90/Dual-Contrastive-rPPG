"""Subject-cluster bootstrap, multi-seed summary, and plot-ready tabular statistics."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .hr import metrics


def subject_bootstrap(session_rows, repetitions=10000, seed=42):
    """Resample subjects with replacement and retain all sessions of each draw.

    Duplicate draws are independently tagged so clustered observations do not
    disappear. Degenerate Pearson replicates are excluded with counts reported.
    """
    by_subject = defaultdict(list)
    for r in session_rows:
        by_subject[r["subject"]].append(r)
    subjects = sorted(by_subject)
    if len(subjects) < 2:
        raise ValueError("Bootstrap requires >=2 independent subjects")
    base = metrics(session_rows)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("mae_bpm", "rmse_bpm", "pearson_r")}
    skipped = 0
    for i in range(repetitions):
        drawn = rng.choice(subjects, len(subjects), replace=True)
        subset = []
        for j, subject in enumerate(drawn):
            subset.extend({**r, "subject": f"bootstrap_{j}"} for r in by_subject[subject])
        estimate = metrics(subset)
        for name in samples:
            if np.isfinite(estimate[name]):
                samples[name].append(estimate[name])
            elif name == "pearson_r":
                skipped += 1
    return {"base": base, "n_bootstrap": repetitions, "seed": seed,
            "invalid_correlation_resamples": skipped,
            "confidence_intervals_95": {name: ([float(np.percentile(values, 2.5)),
                                                float(np.percentile(values, 97.5))]
                                               if values else [None, None])
                                        for name, values in samples.items()}}


def summarize_seeds(per_seed):
    if not per_seed:
        raise ValueError("Missing per-seed outcomes")
    return {k: {"mean": float(np.mean([v[k] for v in per_seed])),
                "sample_sd": float(np.std([v[k] for v in per_seed], ddof=1)) if len(per_seed) > 1 else None}
            for k in ("mae_bpm", "rmse_bpm", "pearson_r")}
