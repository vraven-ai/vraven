"""Dataset-level VRAVEN analysis built from independently reproducible traces."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vraven.core.config import VRavenConfig
from vraven.core.explainer import Explainer
from vraven.core.trace import NeuralTrace
from vraven.report.report import VRavenReport
from vraven.visualisation.palette import RenderMode
from vraven.visualisation.reasoning_dataset import (
    reasoning_landscape_figure,
    reasoning_outlier_figure,
    reasoning_similarity_matrix_figure,
)


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """One explicit dataset item for unambiguous analysis."""

    inputs: Any
    target: int | None = None
    label: Any | None = None
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class MaterialisedSample:
    inputs: Any
    target: int | None
    label: Any | None
    sample_id: str


def _is_scalar_label(value: Any) -> bool:
    if isinstance(value, (bool, int, np.integer, str)):
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 1
    if isinstance(value, np.ndarray):
        return value.size == 1
    return False


def _scalar_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalise_target(value: Any) -> int | None:
    if value is None:
        return None
    resolved = _scalar_value(value)
    if isinstance(resolved, bool):
        return int(resolved)
    if isinstance(resolved, (int, np.integer)):
        return int(resolved)
    raise TypeError(f"Dataset target must be an integer class index, received {type(value).__name__}")


def _nested_batch_size(value: Any) -> int | None:
    sizes: list[int] = []

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor) and item.ndim > 0:
            sizes.append(int(item.shape[0]))
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, tuple):
            for nested in item:
                visit(nested)

    visit(value)
    if not sizes:
        return None
    if len(set(sizes)) != 1:
        return None
    return sizes[0]


def _slice_batch(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[index : index + 1]
    if isinstance(value, Mapping):
        return type(value)((key, _slice_batch(item, index)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_slice_batch(item, index) for item in value)
    return value


def _target_at(
    targets: int | Sequence[int] | torch.Tensor | np.ndarray | None,
    index: int,
) -> int | None:
    if targets is None:
        return None
    if isinstance(targets, int):
        return targets
    if isinstance(targets, torch.Tensor):
        flat = targets.detach().cpu().reshape(-1)
        if index >= flat.numel():
            raise IndexError("targets contains fewer entries than the analysed dataset")
        return int(flat[index].item())
    if isinstance(targets, np.ndarray):
        flat = targets.reshape(-1)
        if index >= flat.size:
            raise IndexError("targets contains fewer entries than the analysed dataset")
        return int(flat[index])
    if index >= len(targets):
        raise IndexError("targets contains fewer entries than the analysed dataset")
    return int(targets[index])


def _normalise_item(
    item: Any,
    *,
    index: int,
    explicit_target: int | None,
    input_getter: Callable[[Any], Any] | None,
    target_getter: Callable[[Any], Any] | None,
    label_getter: Callable[[Any], Any] | None,
    id_getter: Callable[[Any], str] | None,
) -> MaterialisedSample:
    if isinstance(item, DatasetItem):
        return MaterialisedSample(
            inputs=item.inputs,
            target=explicit_target if explicit_target is not None else item.target,
            label=item.label,
            sample_id=item.sample_id or f"sample_{index:05d}",
        )

    if input_getter is not None:
        inputs = input_getter(item)
        inferred_target = target_getter(item) if target_getter is not None else None
        label = label_getter(item) if label_getter is not None else inferred_target
        sample_id = id_getter(item) if id_getter is not None else f"sample_{index:05d}"
        return MaterialisedSample(
            inputs=inputs,
            target=explicit_target if explicit_target is not None else _normalise_target(inferred_target),
            label=_scalar_value(label),
            sample_id=str(sample_id),
        )

    if isinstance(item, tuple) and len(item) == 2 and _is_scalar_label(item[1]):
        inferred_target = _normalise_target(item[1])
        return MaterialisedSample(
            inputs=item[0],
            target=explicit_target if explicit_target is not None else inferred_target,
            label=_scalar_value(item[1]),
            sample_id=f"sample_{index:05d}",
        )

    return MaterialisedSample(
        inputs=item,
        target=explicit_target,
        label=None,
        sample_id=f"sample_{index:05d}",
    )


def materialise_samples(
    data: Any,
    *,
    targets: int | Sequence[int] | torch.Tensor | np.ndarray | None = None,
    max_samples: int | None = 256,
    input_getter: Callable[[Any], Any] | None = None,
    target_getter: Callable[[Any], Any] | None = None,
    label_getter: Callable[[Any], Any] | None = None,
    id_getter: Callable[[Any], str] | None = None,
) -> tuple[MaterialisedSample, ...]:
    """Normalise tensors, batched structures, datasets or iterables into samples.

    A tuple ``(input, scalar_label)`` is interpreted as a labelled dataset item.
    Use :class:`DatasetItem` or explicit getter functions when that convention is
    ambiguous for a model with multiple positional inputs.
    """

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be greater than zero or None")

    batch_size = _nested_batch_size(data)
    samples: list[MaterialisedSample] = []
    if batch_size is not None and isinstance(data, (torch.Tensor, Mapping, tuple)):
        limit = batch_size if max_samples is None else min(batch_size, max_samples)
        for index in range(limit):
            samples.append(
                MaterialisedSample(
                    inputs=_slice_batch(data, index),
                    target=_target_at(targets, index),
                    label=None,
                    sample_id=f"sample_{index:05d}",
                )
            )
        return tuple(samples)

    if not isinstance(data, Iterable):
        raise TypeError("data must be a batched tensor structure or an iterable of samples")

    for index, item in enumerate(data):
        if max_samples is not None and index >= max_samples:
            break
        samples.append(
            _normalise_item(
                item,
                index=index,
                explicit_target=_target_at(targets, index),
                input_getter=input_getter,
                target_getter=target_getter,
                label_getter=label_getter,
                id_getter=id_getter,
            )
        )
    if not samples:
        raise ValueError("The supplied dataset did not produce any samples")
    return tuple(samples)


def _json_value(value: Any) -> Any:
    value = _scalar_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(slots=True)
class VRavenDatasetReport:
    """A collection of per-sample traces and their fixed-width fingerprints."""

    reports: tuple[VRavenReport, ...]
    sample_ids: tuple[str, ...]
    labels: tuple[Any | None, ...]
    fingerprint_dimensions: int = 128
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reports:
            raise ValueError("A dataset report requires at least one sample report")
        if len(self.reports) != len(self.sample_ids) or len(self.reports) != len(self.labels):
            raise ValueError("reports, sample_ids and labels must have equal lengths")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")
        if self.fingerprint_dimensions <= 0:
            raise ValueError("fingerprint_dimensions must be greater than zero")

    @property
    def fingerprints(self) -> np.ndarray:
        return np.vstack(
            [report.reasoning_fingerprint(self.fingerprint_dimensions) for report in self.reports]
        )

    @property
    def predicted_classes(self) -> tuple[int | None, ...]:
        values: list[int | None] = []
        for report in self.reports:
            prediction = report.trace.prediction
            classes = prediction.predicted_classes if prediction is not None else None
            values.append(int(classes[0]) if classes else None)
        return tuple(values)

    @property
    def confidences(self) -> tuple[float | None, ...]:
        values: list[float | None] = []
        for report in self.reports:
            prediction = report.trace.prediction
            confidence = prediction.confidence if prediction is not None else None
            values.append(float(confidence[0]) if confidence else None)
        return tuple(values)

    def similarity_matrix(self) -> np.ndarray:
        matrix = self.fingerprints.astype(np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe = np.where(norms > 1e-12, norms, 1.0)
        normalised = matrix / safe
        similarity = normalised @ normalised.T
        zero = norms.reshape(-1) <= 1e-12
        if zero.any():
            for index in np.flatnonzero(zero):
                similarity[index, index] = 1.0
        return np.clip(similarity, -1.0, 1.0)

    def outlier_scores(self, neighbours: int = 5) -> np.ndarray:
        """Return ``1 - mean nearest cosine similarity`` for each sample."""

        if neighbours <= 0:
            raise ValueError("neighbours must be greater than zero")
        similarity = self.similarity_matrix().copy()
        np.fill_diagonal(similarity, -np.inf)
        k = min(neighbours, max(1, len(self.reports) - 1))
        if len(self.reports) == 1:
            return np.zeros(1, dtype=np.float64)
        nearest = np.partition(similarity, -k, axis=1)[:, -k:]
        return 1.0 - np.mean(nearest, axis=1)

    def summary(self) -> dict[str, Any]:
        confidences = np.asarray(
            [value for value in self.confidences if value is not None], dtype=np.float64
        )
        similarities = self.similarity_matrix()
        upper = similarities[np.triu_indices_from(similarities, k=1)]
        outliers = self.outlier_scores()
        prediction_counts = Counter(
            str(value) if value is not None else "none" for value in self.predicted_classes
        )
        return {
            "sample_count": len(self.reports),
            "fingerprint_dimensions": self.fingerprint_dimensions,
            "prediction_counts": dict(prediction_counts),
            "mean_confidence": float(confidences.mean()) if confidences.size else None,
            "mean_pairwise_fingerprint_similarity": float(upper.mean()) if upper.size else 1.0,
            "most_unusual_sample": self.sample_ids[int(np.argmax(outliers))],
            "maximum_outlier_score": float(outliers.max()),
            "mode": self.reports[0].trace.mode,
            "framework": self.reports[0].trace.metadata.get("framework"),
            **self.metadata,
        }

    def reasoning_similarity(
        self,
        *,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ):
        return reasoning_similarity_matrix_figure(
            self.fingerprints,
            sample_ids=self.sample_ids,
            mode=mode,
            show_title=show_title,
        )

    def reasoning_outliers(
        self,
        *,
        neighbours: int = 5,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ):
        return reasoning_outlier_figure(
            self.fingerprints,
            sample_ids=self.sample_ids,
            neighbours=neighbours,
            mode=mode,
            show_title=show_title,
        )

    def reasoning_landscape(
        self,
        *,
        colour_by: str = "prediction",
        method: str = "pca",
        highlight_index: int | None = None,
        random_seed: int = 17,
        perplexity: float = 30.0,
        mode: RenderMode = "publication_light",
        show_title: bool | None = None,
    ):
        if colour_by == "prediction":
            groups: Sequence[Any] = self.predicted_classes
        elif colour_by == "label":
            groups = self.labels
        else:
            raise ValueError("colour_by must be 'prediction' or 'label'")
        return reasoning_landscape_figure(
            self.fingerprints,
            sample_ids=self.sample_ids,
            groups=groups,
            group_name=colour_by,
            method=method,
            highlight_index=highlight_index,
            random_seed=random_seed,
            perplexity=perplexity,
            mode=mode,
            show_title=show_title,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "fingerprint_dimensions": self.fingerprint_dimensions,
            "sample_ids": list(self.sample_ids),
            "labels": [_json_value(value) for value in self.labels],
            "metadata": self.metadata,
            "traces": [report.trace.to_dict() for report in self.reports],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VRavenDatasetReport:
        traces = payload.get("traces")
        if not isinstance(traces, list) or not traces:
            raise ValueError("Dataset report payload requires a non-empty traces list")
        return cls(
            reports=tuple(VRavenReport(NeuralTrace.from_dict(item)) for item in traces),
            sample_ids=tuple(str(item) for item in payload.get("sample_ids", [])),
            labels=tuple(payload.get("labels", [])),
            fingerprint_dimensions=int(payload.get("fingerprint_dimensions", 128)),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> VRavenDatasetReport:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("A dataset report file must contain a JSON object")
        return cls.from_dict(payload)


def explain_materialised_samples(
    model: Any,
    samples: Sequence[MaterialisedSample],
    *,
    config: VRavenConfig,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    fingerprint_dimensions: int = 128,
    metadata: dict[str, Any] | None = None,
) -> VRavenDatasetReport:
    explainer = Explainer(model, config, class_names=class_names)
    reports: list[VRavenReport] = []
    for index, sample in enumerate(samples):
        reports.append(
            explainer.explain(
                sample.inputs,
                target=sample.target,
                metadata={
                    "dataset_sample_index": index,
                    "dataset_sample_id": sample.sample_id,
                },
            )
        )
    return VRavenDatasetReport(
        reports=tuple(reports),
        sample_ids=tuple(sample.sample_id for sample in samples),
        labels=tuple(sample.label for sample in samples),
        fingerprint_dimensions=fingerprint_dimensions,
        metadata=dict(metadata or {}),
    )


def explain_dataset(
    model: Any,
    data: Any,
    *,
    targets: int | Sequence[int] | torch.Tensor | np.ndarray | None = None,
    mode: str = "fast",
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    max_samples: int | None = 256,
    fingerprint_dimensions: int = 128,
    input_getter: Callable[[Any], Any] | None = None,
    target_getter: Callable[[Any], Any] | None = None,
    label_getter: Callable[[Any], Any] | None = None,
    id_getter: Callable[[Any], str] | None = None,
    metadata: dict[str, Any] | None = None,
    **config_overrides: Any,
) -> VRavenDatasetReport:
    """Explain a bounded set of samples and compare their reduced internal traces.

    Samples are explained independently so the returned fingerprint represents one
    sample rather than an aggregate over a mixed batch. This is intentionally more
    expensive than a single batch trace and is bounded by ``max_samples``.
    """

    samples = materialise_samples(
        data,
        targets=targets,
        max_samples=max_samples,
        input_getter=input_getter,
        target_getter=target_getter,
        label_getter=label_getter,
        id_getter=id_getter,
    )
    config = VRavenConfig(mode=mode, **config_overrides)  # type: ignore[arg-type]
    return explain_materialised_samples(
        model,
        samples,
        config=config,
        class_names=class_names,
        fingerprint_dimensions=fingerprint_dimensions,
        metadata={
            "analysis_semantics": "independent_per_sample_trace_collection",
            "requested_max_samples": max_samples,
            **(metadata or {}),
        },
    )
