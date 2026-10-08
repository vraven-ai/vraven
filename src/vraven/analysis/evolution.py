"""Checkpoint evolution analysis for one fixed input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vraven.core.config import VRavenConfig
from vraven.core.explainer import Explainer
from vraven.report.report import VRavenReport
from vraven.visualisation.model_evolution import model_evolution_figure
from vraven.visualisation.palette import RenderMode


@dataclass(slots=True)
class VRavenEvolutionReport:
    reports: tuple[VRavenReport, ...]
    checkpoint_names: tuple[str, ...]
    fingerprint_dimensions: int = 128

    def __post_init__(self) -> None:
        if len(self.reports) < 2:
            raise ValueError("Evolution analysis requires at least two checkpoints")
        if len(self.reports) != len(self.checkpoint_names):
            raise ValueError("reports and checkpoint_names must have equal lengths")
        if len(set(self.checkpoint_names)) != len(self.checkpoint_names):
            raise ValueError("checkpoint_names must be unique")

    @property
    def fingerprints(self) -> np.ndarray:
        return np.vstack(
            [report.reasoning_fingerprint(self.fingerprint_dimensions) for report in self.reports]
        )

    @property
    def similarity_to_final(self) -> np.ndarray:
        matrix = self.fingerprints.astype(np.float64)
        final = matrix[-1]
        final_norm = np.linalg.norm(final)
        values: list[float] = []
        for row in matrix:
            denominator = np.linalg.norm(row) * final_norm
            if denominator <= 1e-12:
                values.append(1.0 if np.allclose(row, final) else 0.0)
            else:
                values.append(float(np.dot(row, final) / denominator))
        return np.asarray(values, dtype=np.float64)

    @property
    def predictions(self) -> tuple[int | None, ...]:
        values: list[int | None] = []
        for report in self.reports:
            prediction = report.trace.prediction
            classes = prediction.predicted_classes if prediction else None
            values.append(int(classes[0]) if classes else None)
        return tuple(values)

    @property
    def confidences(self) -> np.ndarray:
        values: list[float] = []
        for report in self.reports:
            prediction = report.trace.prediction
            confidence = prediction.confidence if prediction else None
            values.append(float(confidence[0]) if confidence else float("nan"))
        return np.asarray(values, dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        predictions = self.predictions
        changes = sum(first != second for first, second in zip(predictions, predictions[1:], strict=False))
        return {
            "checkpoint_count": len(self.reports),
            "first_checkpoint": self.checkpoint_names[0],
            "final_checkpoint": self.checkpoint_names[-1],
            "prediction_changes": changes,
            "initial_to_final_fingerprint_similarity": float(self.similarity_to_final[0]),
            "fingerprint_dimensions": self.fingerprint_dimensions,
        }

    def figure(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ):
        return model_evolution_figure(
            checkpoint_names=self.checkpoint_names,
            similarity_to_final=self.similarity_to_final,
            confidences=self.confidences,
            predictions=self.predictions,
            mode=mode,
            show_title=show_title,
        )


def compare_checkpoints(
    models: Sequence[Any],
    inputs: Any,
    target: int | None = None,
    *,
    checkpoint_names: Sequence[str] | None = None,
    mode: str = "fast",
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    fingerprint_dimensions: int = 128,
    **config_overrides: Any,
) -> VRavenEvolutionReport:
    if len(models) < 2:
        raise ValueError("At least two models are required")
    names = tuple(checkpoint_names or [f"checkpoint_{index}" for index in range(len(models))])
    if len(names) != len(models):
        raise ValueError("checkpoint_names must match the number of models")
    reports: list[VRavenReport] = []
    for index, model in enumerate(models):
        config = VRavenConfig(mode=mode, **config_overrides)  # type: ignore[arg-type]
        reports.append(
            Explainer(model, config, class_names=class_names).explain(
                inputs,
                target=target,
                metadata={"checkpoint_index": index, "checkpoint_name": names[index]},
            )
        )
    return VRavenEvolutionReport(
        reports=tuple(reports),
        checkpoint_names=names,
        fingerprint_dimensions=fingerprint_dimensions,
    )
