"""Treat disagreement among explanation methods as a scientific result."""

from __future__ import annotations

import numpy as np

from vraven.darkside import analyse_explanation_uncertainty, analyse_method_disagreement

input_gradient = np.array([0.9, -0.6, 0.3, 0.2, 0.1])
lrp = np.array([0.7, -0.2, 0.5, 0.1, 0.0])
causal_ablation = np.array([0.2, -0.8, 0.1, 0.6, 0.0])

result = analyse_method_disagreement(
    {
        "input × gradient": input_gradient,
        "LRP": lrp,
        "causal ablation": causal_ablation,
    },
    top_k=3,
)
result.figure(mode="publication_light").write_html("method_disagreement.html")

repeated = np.stack([input_gradient + np.random.default_rng(seed).normal(0, 0.05, 5) for seed in range(30)])
uncertainty = analyse_explanation_uncertainty(repeated, top_k=3)
uncertainty.figure(mode="interactive_dark").write_html("explanation_uncertainty.html")
