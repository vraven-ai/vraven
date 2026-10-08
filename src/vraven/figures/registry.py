from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .architecture import (
    device_execution,
    package_architecture,
    runtime_dataflow,
    system_architecture,
)
from .integration import user_workflow, visual_contract
from .methodology import (
    architecture_adapters,
    evidence_reconstruction,
    explanation_modes,
    latent_community_discovery,
    reasoning_dna_pipeline,
    training_monitoring,
)
from .primitives import Canvas
from .theme import Theme
from .validation import (
    benchmark_protocol,
    causal_validation_loop,
    reasoning_drift,
    reliability_calibration,
    reproducibility_provenance,
)


@dataclass(frozen=True)
class FigureSpec:
    key: str
    category: str
    caption_short: str
    renderer: Callable[[Theme], Canvas]
    current_status: str = "conceptual methodology diagram"


FIGURES: dict[str, FigureSpec] = {
    "01_system_architecture": FigureSpec(
        "01_system_architecture",
        "architecture",
        "End-to-end VRAVEN system architecture from model entry to scientific outputs.",
        system_architecture,
    ),
    "02_runtime_dataflow": FigureSpec(
        "02_runtime_dataflow",
        "architecture",
        "Runtime path showing transient tensors, on-device reduction and compact trace persistence.",
        runtime_dataflow,
    ),
    "03_package_architecture": FigureSpec(
        "03_package_architecture",
        "architecture",
        "Modular package architecture and optional acceleration sidecars.",
        package_architecture,
    ),
    "04_device_execution": FigureSpec(
        "04_device_execution",
        "architecture",
        "Device-following execution and bounded host transfer across supported framework paths.",
        device_execution,
    ),
    "05_explanation_modes": FigureSpec(
        "05_explanation_modes",
        "methodology",
        "Fast, gradient and causal explanation modes with distinct evidence semantics.",
        explanation_modes,
    ),
    "06_evidence_reconstruction": FigureSpec(
        "06_evidence_reconstruction",
        "methodology",
        "Layerwise evidence reconstruction using separate activation, gradient and uncertainty channels.",
        evidence_reconstruction,
    ),
    "07_reasoning_dna_pipeline": FigureSpec(
        "07_reasoning_dna_pipeline",
        "methodology",
        "Deterministic construction of a fixed-width reasoning fingerprint.",
        reasoning_dna_pipeline,
    ),
    "08_latent_community_discovery": FigureSpec(
        "08_latent_community_discovery",
        "methodology",
        "Activation community discovery and the distinction between latent groups and validated concepts.",
        latent_community_discovery,
    ),
    "09_training_monitoring": FigureSpec(
        "09_training_monitoring",
        "methodology",
        "Training-time instrumentation, checkpoint traces and neural diagnostics.",
        training_monitoring,
    ),
    "10_architecture_adapters": FigureSpec(
        "10_architecture_adapters",
        "methodology",
        "Architecture-specific signals mapped to the common NeuralTrace schema.",
        architecture_adapters,
    ),
    "11_benchmark_protocol": FigureSpec(
        "11_benchmark_protocol",
        "validation",
        "Controlled evaluation against attribution and explanation baselines.",
        benchmark_protocol,
    ),
    "12_causal_validation_loop": FigureSpec(
        "12_causal_validation_loop",
        "validation",
        "Bounded causal validation loop with matched controls and signed effect estimation.",
        causal_validation_loop,
    ),
    "13_reliability_calibration": FigureSpec(
        "13_reliability_calibration",
        "validation",
        "Multidimensional explanation reliability profile without arbitrary confidence claims.",
        reliability_calibration,
    ),
    "14_reasoning_drift": FigureSpec(
        "14_reasoning_drift",
        "validation",
        "Reasoning drift analysis in a shared fingerprint space.",
        reasoning_drift,
    ),
    "15_reproducibility_provenance": FigureSpec(
        "15_reproducibility_provenance",
        "validation",
        "Reproducibility and provenance bundle for traces, figures and releases.",
        reproducibility_provenance,
    ),
    "16_visual_contract": FigureSpec(
        "16_visual_contract",
        "visual_system",
        "Scientific visual contract from validated measurements to dual-mode standalone figures.",
        visual_contract,
    ),
    "17_user_workflow": FigureSpec(
        "17_user_workflow",
        "integration",
        "Local installation and model analysis workflow for live models and saved weights.",
        user_workflow,
    ),
}


def list_figures() -> list[FigureSpec]:
    return list(FIGURES.values())


def get_figure(key: str) -> FigureSpec:
    try:
        return FIGURES[key]
    except KeyError as exc:
        available = ", ".join(FIGURES)
        raise KeyError(f"Unknown figure {key!r}. Available: {available}") from exc
