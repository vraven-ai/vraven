"""Signature visualisations for the VRAVEN causal decompiler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vraven.visualisation.palette import RenderMode

from .causal_proof import causal_proof
from .concept_genome import concept_genome
from .contrastive_proof import contrastive_proof
from .counterfactual_twin import counterfactual_twin
from .dark_matter import dark_matter
from .model_xray import model_xray
from .programme_proof import programme_proof

VISUALS: dict[str, Callable[..., Any]] = {
    "model_xray": model_xray,
    "raven_eye": model_xray,
    "concept_genome": concept_genome,
    "raven_flight": concept_genome,
    "causal_proof": causal_proof,
    "raven_brain": causal_proof,
    "programme_proof": programme_proof,
    "raven_code": programme_proof,
    "contrastive_proof": contrastive_proof,
    "raven_contrast": contrastive_proof,
    "counterfactual_twin": counterfactual_twin,
    "raven_shadow": counterfactual_twin,
    "dark_matter": dark_matter,
    "raven_dark_matter": dark_matter,
}


def render_visual(result, visual: str, *, mode: RenderMode = "interactive_dark", **kwargs: Any):
    key = visual.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in VISUALS:
        supported = ", ".join(sorted(set(VISUALS)))
        raise ValueError(f"Unknown decompiler visual {visual!r}. Choose from: {supported}")
    return VISUALS[key](result, mode=mode, **kwargs)


__all__ = [
    "VISUALS",
    "causal_proof",
    "concept_genome",
    "contrastive_proof",
    "counterfactual_twin",
    "dark_matter",
    "model_xray",
    "programme_proof",
    "render_visual",
]
