"""Scientific contracts and finalisation helpers for VRAVEN figures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import plotly.graph_objects as go

from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import apply_vraven_theme

EvidenceTier = Literal[
    "predictive",
    "observational",
    "comparative",
    "behavioural",
    "interventional",
    "observational_interventional",
    "formal",
]


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """One measured result suitable for a compact publication evidence strip."""

    label: str
    value: str
    interpretation: str
    unit: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "label": self.label,
            "value": self.value,
            "interpretation": self.interpretation,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class VisualContract:
    visual_id: str
    research_question: str
    required_metrics: tuple[str, ...]
    data_domain: str
    visual_encodings: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    evidence_tier: EvidenceTier = "observational"
    interpretation: str = "This figure reports a measured association or descriptive pattern."
    metric_definitions: tuple[str, ...] = ()
    supported_modes: tuple[RenderMode, ...] = ("publication_light", "interactive_dark")


@dataclass(frozen=True, slots=True)
class VisualisationMetadata:
    visual_id: str
    scientific_question: str
    metric_names: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    evidence_tier: EvidenceTier = "observational"
    interpretation: str = ""
    metric_definitions: tuple[str, ...] = ()
    key_results: tuple[PublicationResult, ...] = ()
    normalisation: str | None = None
    random_seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_contract(
        cls,
        contract: VisualContract,
        *,
        normalisation: str | None = None,
        random_seed: int | None = None,
        key_results: tuple[PublicationResult, ...] = (),
        extra: dict[str, Any] | None = None,
    ) -> VisualisationMetadata:
        return cls(
            visual_id=contract.visual_id,
            scientific_question=contract.research_question,
            metric_names=contract.required_metrics,
            limitations=contract.limitations,
            evidence_tier=contract.evidence_tier,
            interpretation=contract.interpretation,
            metric_definitions=contract.metric_definitions,
            key_results=key_results,
            normalisation=normalisation,
            random_seed=random_seed,
            extra={
                "data_domain": contract.data_domain,
                "visual_encodings": list(contract.visual_encodings),
                **(extra or {}),
            },
        )

    def as_plotly_meta(self) -> dict[str, Any]:
        return {
            "visual_id": self.visual_id,
            "scientific_question": self.scientific_question,
            "metric_names": list(self.metric_names),
            "limitations": list(self.limitations),
            "evidence_tier": self.evidence_tier,
            "interpretation": self.interpretation,
            "metric_definitions": list(self.metric_definitions),
            "key_results": [result.as_dict() for result in self.key_results],
            "normalisation": self.normalisation,
            "random_seed": self.random_seed,
            **self.extra,
        }


def finalise_figure(
    figure: go.Figure,
    *,
    metadata: VisualisationMetadata,
    mode: RenderMode,
    title: str,
    subtitle: str | None = None,
    width: int | None = None,
    height: int | None = None,
    show_legend: bool | None = None,
    show_title: bool | None = None,
) -> go.Figure:
    existing_meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    merged_meta = dict(existing_meta)
    merged_meta["scientific"] = metadata.as_plotly_meta()
    figure.update_layout(meta=merged_meta)
    return apply_vraven_theme(
        figure,
        mode=mode,
        title=title,
        subtitle=subtitle,
        width=width,
        height=height,
        show_legend=show_legend,
        show_title=show_title,
    )
