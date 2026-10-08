"""Aligned cohort container for internal traces and representation studies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vraven.core.trace import NeuralTrace
from vraven.reasoning.fingerprint import shared_reasoning_fingerprints


def _mean_row(values: list[list[float]], width: int) -> np.ndarray:
    """Reduce one stored representation to a fixed-width sample row."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros(width, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    row = array.mean(axis=0)
    if row.size != width:
        raise ValueError("Representation width changed across aligned traces")
    return row


def _consistent_width(rows: Iterable[np.ndarray], *, family: str, layer: str) -> int:
    """Return the shared non-empty width, or one for an entirely empty family."""

    widths = {int(np.asarray(row).size) for row in rows if np.asarray(row).size}
    if len(widths) > 1:
        raise ValueError(f"{family} width changed for {layer}")
    return widths.pop() if widths else 1


def _representation_width(values: Any) -> int:
    """Return the feature width of one stored representation."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0
    if array.ndim == 1:
        return int(array.size)
    if array.ndim == 2:
        return int(array.shape[1])
    raise ValueError("Stored representations must be one- or two-dimensional")


@dataclass(slots=True)
class TraceDataset:
    """Aligned sample-level internal measurements.

    Rows in every matrix refer to the same ``sample_ids`` in the same order.
    This alignment is mandatory for CKA, neighbourhood and paired stability
    analyses.
    """

    sample_ids: tuple[str, ...]
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    target_scores: np.ndarray
    representations: dict[str, np.ndarray]
    unit_influences: dict[str, np.ndarray] = field(default_factory=dict)
    unit_energies: dict[str, np.ndarray] = field(default_factory=dict)
    activation_rms: dict[str, np.ndarray] = field(default_factory=dict)
    timing_ms: dict[str, np.ndarray] = field(default_factory=dict)
    fingerprints: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        if n < 1:
            raise ValueError("TraceDataset requires at least one sample")
        if len(set(self.sample_ids)) != n:
            raise ValueError("sample_ids must be unique for paired analyses")
        if not self.representations:
            raise ValueError("TraceDataset requires at least one representation layer")
        for name, vector in (
            ("true_labels", self.true_labels),
            ("predicted_labels", self.predicted_labels),
            ("target_scores", self.target_scores),
        ):
            array = np.asarray(vector).reshape(-1)
            if array.size != n:
                raise ValueError(f"{name} length does not match sample_ids")
            if name == "target_scores" and not np.isfinite(array.astype(np.float64)).all():
                raise ValueError("target_scores contains non-finite values")
        for family_name, family in (
            ("representations", self.representations),
            ("unit_influences", self.unit_influences),
            ("unit_energies", self.unit_energies),
            ("activation_rms", self.activation_rms),
            ("timing_ms", self.timing_ms),
        ):
            for layer, matrix in family.items():
                array = np.asarray(matrix)
                if array.ndim < 1 or array.shape[0] != n:
                    raise ValueError(f"{family_name}[{layer!r}] is not sample aligned")
                if not np.isfinite(array.astype(np.float64)).all():
                    raise ValueError(f"{family_name}[{layer!r}] contains non-finite values")
        if self.fingerprints is not None:
            if self.fingerprints.shape[0] != n:
                raise ValueError("fingerprints are not sample aligned")
            if not np.isfinite(np.asarray(self.fingerprints, dtype=np.float64)).all():
                raise ValueError("fingerprints contain non-finite values")

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[NeuralTrace],
        *,
        true_labels: Sequence[Any] | None = None,
        sample_ids: Sequence[str] | None = None,
        strict_layers: bool = True,
    ) -> TraceDataset:
        if not traces:
            raise ValueError("At least one trace is required")
        identifiers = [[layer.identifier for layer in trace.layers] for trace in traces]
        if strict_layers and any(value != identifiers[0] for value in identifiers[1:]):
            raise ValueError("All traces must contain identical ordered layer identifiers")
        common = [identifier for identifier in identifiers[0] if all(identifier in row for row in identifiers)]
        if not common:
            raise ValueError("No common layer identifiers were found")

        representation_rows: dict[str, list[np.ndarray]] = {name: [] for name in common}
        influence_rows: dict[str, list[np.ndarray]] = {name: [] for name in common}
        energy_rows: dict[str, list[np.ndarray]] = {name: [] for name in common}
        rms_rows: dict[str, list[float]] = {name: [] for name in common}
        timing_rows: dict[str, list[float]] = {name: [] for name in common}
        lookups = [{layer.identifier: layer for layer in trace.layers} for trace in traces]

        representation_widths: dict[str, int] = {}
        influence_widths: dict[str, int] = {}
        energy_widths: dict[str, int] = {}
        for name in common:
            representation_width_values = {
                width
                for lookup in lookups
                if (width := _representation_width(lookup[name].representation)) > 0
            }
            if len(representation_width_values) > 1:
                raise ValueError(f"Representation width changed for {name}")
            representation_widths[name] = (
                representation_width_values.pop() if representation_width_values else 1
            )
            influence_widths[name] = _consistent_width(
                (
                    np.asarray(lookup[name].gradient.unit_influence, dtype=np.float64)
                    if lookup[name].gradient is not None and lookup[name].gradient.unit_influence
                    else np.asarray(lookup[name].unit_signed_mean, dtype=np.float64)
                    for lookup in lookups
                ),
                family="Unit influence",
                layer=name,
            )
            energy_widths[name] = _consistent_width(
                (np.asarray(lookup[name].unit_energy, dtype=np.float64) for lookup in lookups),
                family="Unit energy",
                layer=name,
            )

        for lookup in lookups:
            for name in common:
                layer = lookup[name]
                representation_rows[name].append(
                    _mean_row(layer.representation, representation_widths[name])
                )

                influence = (
                    np.asarray(layer.gradient.unit_influence, dtype=np.float64)
                    if layer.gradient is not None and layer.gradient.unit_influence
                    else np.asarray(layer.unit_signed_mean, dtype=np.float64)
                )
                influence_rows[name].append(
                    influence if influence.size else np.zeros(influence_widths[name], dtype=np.float64)
                )

                energy = np.asarray(layer.unit_energy, dtype=np.float64)
                energy_rows[name].append(
                    energy if energy.size else np.zeros(energy_widths[name], dtype=np.float64)
                )
                rms_rows[name].append(float(layer.activation.rms))
                timing_rows[name].append(float(layer.timing_ms or 0.0))

        predicted = np.asarray(
            [
                -1 if trace.prediction.predicted_index is None else trace.prediction.predicted_index
                for trace in traces
            ],
            dtype=object,
        )
        true = np.asarray(true_labels if true_labels is not None else predicted.copy(), dtype=object)
        ids = tuple(sample_ids) if sample_ids is not None else tuple(trace.trace_id for trace in traces)
        if len(ids) != len(traces):
            raise ValueError("sample_ids length does not match traces")
        fingerprints = shared_reasoning_fingerprints(list(traces), normalise=True)

        return cls(
            sample_ids=ids,
            true_labels=true,
            predicted_labels=predicted,
            target_scores=np.asarray([trace.prediction.target_score for trace in traces]),
            representations={name: np.vstack(rows) for name, rows in representation_rows.items()},
            unit_influences={name: np.vstack(rows) for name, rows in influence_rows.items()},
            unit_energies={name: np.vstack(rows) for name, rows in energy_rows.items()},
            activation_rms={name: np.asarray(rows) for name, rows in rms_rows.items()},
            timing_ms={name: np.asarray(rows) for name, rows in timing_rows.items()},
            fingerprints=fingerprints,
            metadata={
                "framework": traces[0].framework,
                "model_name": traces[0].model_name,
                "trace_schema": traces[0].schema_version,
            },
        )

    @classmethod
    def from_reports(cls, reports: Sequence[Any], **kwargs: Any) -> TraceDataset:
        return cls.from_traces([report.trace for report in reports], **kwargs)

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.representations)

    def subset(self, mask: Any) -> TraceDataset:
        selected = np.asarray(mask)
        if selected.dtype != bool:
            indices = selected.astype(int)
        else:
            if selected.size != len(self.sample_ids):
                raise ValueError("Boolean mask length does not match dataset")
            indices = np.flatnonzero(selected)
        return TraceDataset(
            sample_ids=tuple(self.sample_ids[index] for index in indices),
            true_labels=self.true_labels[indices],
            predicted_labels=self.predicted_labels[indices],
            target_scores=self.target_scores[indices],
            representations={key: value[indices] for key, value in self.representations.items()},
            unit_influences={key: value[indices] for key, value in self.unit_influences.items()},
            unit_energies={key: value[indices] for key, value in self.unit_energies.items()},
            activation_rms={key: value[indices] for key, value in self.activation_rms.items()},
            timing_ms={key: value[indices] for key, value in self.timing_ms.items()},
            fingerprints=None if self.fingerprints is None else self.fingerprints[indices],
            metadata=dict(self.metadata),
        )

    def cohort(self, *, true_label: Any | None = None, predicted_label: Any | None = None) -> TraceDataset:
        mask = np.ones(len(self.sample_ids), dtype=bool)
        if true_label is not None:
            mask &= self.true_labels == true_label
        if predicted_label is not None:
            mask &= self.predicted_labels == predicted_label
        if not np.any(mask):
            raise ValueError("Requested cohort is empty")
        return self.subset(mask)

    def confusion_cohort(self, name: str, *, positive_label: Any = 1) -> TraceDataset:
        truth = self.true_labels == positive_label
        predicted = self.predicted_labels == positive_label
        masks = {
            "tp": truth & predicted,
            "fp": ~truth & predicted,
            "fn": truth & ~predicted,
            "tn": ~truth & ~predicted,
        }
        if name.lower() not in masks:
            raise ValueError("name must be one of tp, fp, fn or tn")
        return self.subset(masks[name.lower()])
