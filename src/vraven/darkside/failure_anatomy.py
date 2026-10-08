"""Internal anatomy of false positives, false negatives and correct predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import AnalysisResult
from .dataset import TraceDataset
from .math_utils import energy_distance, hedges_g, mmd2_rbf


def analyse_failure_anatomy(
    dataset: TraceDataset,
    *,
    failure: str = "fp",
    reference: str = "tp",
    positive_label: Any = 1,
    seed: int = 42,
) -> AnalysisResult:
    """Compare a failure cohort with a matched outcome cohort across depth."""

    if failure.lower() == reference.lower():
        raise ValueError("failure and reference cohorts must be different")
    failures = dataset.confusion_cohort(failure, positive_label=positive_label)
    controls = dataset.confusion_cohort(reference, positive_label=positive_label)
    rows = []
    for layer in dataset.layers:
        left = failures.representations[layer]
        right = controls.representations[layer]
        effects = hedges_g(left, right)
        finite_effects = np.abs(effects[np.isfinite(effects)])
        rows.append(
            {
                "layer": layer,
                "energy_distance": energy_distance(left, right),
                "mmd2_rbf": mmd2_rbf(left, right),
                "mean_absolute_hedges_g": (
                    float(finite_effects.mean()) if finite_effects.size else None
                ),
                "maximum_absolute_hedges_g": (
                    float(finite_effects.max()) if finite_effects.size else None
                ),
            }
        )

    probe_rows = []
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        labels = np.concatenate([np.ones(len(failures.sample_ids)), np.zeros(len(controls.sample_ids))])
        smallest = min(len(failures.sample_ids), len(controls.sample_ids))
        folds = min(5, smallest)
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            for layer in dataset.layers:
                values = np.vstack(
                    [failures.representations[layer], controls.representations[layer]]
                )
                estimator = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                )
                scores = cross_val_score(estimator, values, labels, cv=cv, scoring="roc_auc")
                probe_rows.append(
                    {
                        "layer": layer,
                        "failure_probe_auc": float(scores.mean()),
                        "failure_probe_auc_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                    }
                )
    except ImportError:  # pragma: no cover
        probe_rows = []

    probe_lookup = {row["layer"]: row for row in probe_rows}
    for row in rows:
        row.update(probe_lookup.get(row["layer"], {}))
    transition = max(rows, key=lambda row: row.get("failure_probe_auc", row["energy_distance"]))
    return AnalysisResult(
        analysis="failure_anatomy",
        metrics={
            "failure_cohort": failure,
            "reference_cohort": reference,
            "failure_samples": len(failures.sample_ids),
            "reference_samples": len(controls.sample_ids),
            "strongest_separation_layer": transition["layer"],
            "strongest_probe_auc": transition.get("failure_probe_auc"),
        },
        tables={"layers": rows},
        metadata={
            "claim_type": "cohort-level observational failure localisation",
            "positive_label": positive_label,
            "seed": seed,
        },
        limitations=[
            "Failure cohorts can differ because of data composition rather than a single model mechanism.",
            "Probe accuracy measures decodability, not causal responsibility.",
        ],
    )
