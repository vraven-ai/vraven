"""Model-to-model comparison using the same materialised samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vraven.analysis.dataset import (
    VRavenDatasetReport,
    explain_materialised_samples,
    materialise_samples,
)
from vraven.core.config import VRavenConfig
from vraven.visualisation.model_comparison import model_comparison_figure
from vraven.visualisation.palette import RenderMode


@dataclass(slots=True)
class VRavenModelComparison:
    left: VRavenDatasetReport
    right: VRavenDatasetReport
    left_name: str = "model_a"
    right_name: str = "model_b"

    def __post_init__(self) -> None:
        if self.left.sample_ids != self.right.sample_ids:
            raise ValueError("Model comparison requires identical ordered sample IDs")
        if self.left.fingerprint_dimensions != self.right.fingerprint_dimensions:
            raise ValueError("Model comparison requires equal fingerprint dimensions")

    @property
    def fingerprint_similarity(self) -> np.ndarray:
        left = self.left.fingerprints.astype(np.float64)
        right = self.right.fingerprints.astype(np.float64)
        numerator = np.sum(left * right, axis=1)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return np.divide(
            numerator,
            denominator,
            out=np.where(np.all(np.isclose(left, right), axis=1), 1.0, 0.0),
            where=denominator > 1e-12,
        )

    @property
    def prediction_agreement(self) -> np.ndarray:
        return np.asarray(
            [left == right for left, right in zip(
                self.left.predicted_classes,
                self.right.predicted_classes,
                strict=True,
            )],
            dtype=bool,
        )

    @property
    def confidence_delta(self) -> np.ndarray:
        values: list[float] = []
        for left, right in zip(self.left.confidences, self.right.confidences, strict=True):
            values.append(float("nan") if left is None or right is None else float(right - left))
        return np.asarray(values, dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        similarities = self.fingerprint_similarity
        agreement = self.prediction_agreement
        finite_delta = self.confidence_delta[np.isfinite(self.confidence_delta)]
        return {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "sample_count": len(self.left.reports),
            "mean_fingerprint_similarity": float(np.mean(similarities)),
            "minimum_fingerprint_similarity": float(np.min(similarities)),
            "prediction_agreement_rate": float(np.mean(agreement)),
            "mean_confidence_delta_right_minus_left": (
                float(np.mean(finite_delta)) if finite_delta.size else None
            ),
            "fingerprint_dimensions": self.left.fingerprint_dimensions,
        }

    def alignment(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ):
        return model_comparison_figure(
            sample_ids=self.left.sample_ids,
            fingerprint_similarity=self.fingerprint_similarity,
            prediction_agreement=self.prediction_agreement,
            confidence_delta=self.confidence_delta,
            left_name=self.left_name,
            right_name=self.right_name,
            mode=mode,
            show_title=show_title,
        )


def compare_models(
    model_a: Any,
    model_b: Any,
    data: Any,
    *,
    targets: int | Sequence[int] | torch.Tensor | np.ndarray | None = None,
    mode: str = "fast",
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    max_samples: int | None = 128,
    fingerprint_dimensions: int = 128,
    left_name: str = "model_a",
    right_name: str = "model_b",
    **config_overrides: Any,
) -> VRavenModelComparison:
    """Explain two models on exactly the same bounded materialised sample set."""

    samples = materialise_samples(data, targets=targets, max_samples=max_samples)
    config_a = VRavenConfig(mode=mode, **config_overrides)  # type: ignore[arg-type]
    config_b = VRavenConfig(mode=mode, **config_overrides)  # type: ignore[arg-type]
    left = explain_materialised_samples(
        model_a,
        samples,
        config=config_a,
        class_names=class_names,
        fingerprint_dimensions=fingerprint_dimensions,
        metadata={"comparison_role": "left", "comparison_name": left_name},
    )
    right = explain_materialised_samples(
        model_b,
        samples,
        config=config_b,
        class_names=class_names,
        fingerprint_dimensions=fingerprint_dimensions,
        metadata={"comparison_role": "right", "comparison_name": right_name},
    )
    return VRavenModelComparison(left, right, left_name=left_name, right_name=right_name)
