"""Serializable trace schema for VRAVEN analyses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from vraven.core.errors import TraceValidationError


@dataclass(slots=True)
class _ActivationView:
    rms: float


@dataclass(slots=True)
class _GradientView:
    unit_influence: list[float] | None


@dataclass(slots=True)
class InputAttributionView:
    """DarkSide-compatible view over compressed input attribution."""

    tensor_index: int
    absolute_influence: list[float]


@dataclass(slots=True)
class PredictionTrace:
    output_kind: str
    logits_shape: tuple[int, ...]
    predicted_classes: list[int] | None = None
    target_indices: list[int] | None = None
    probabilities: list[list[float]] | None = None
    target_scores: list[float] = field(default_factory=list)
    confidence: list[float] | None = None
    predictive_entropy: list[float] | None = None
    margin: list[float] | None = None
    raw_output_sample: list[float] = field(default_factory=list)

    @property
    def predicted_index(self) -> int | None:
        """DarkSide-compatible single-sample prediction index."""

        if not self.predicted_classes:
            return None
        return int(self.predicted_classes[0])

    @property
    def target_score(self) -> float:
        """DarkSide-compatible scalar target score (batch mean)."""

        if not self.target_scores:
            return 0.0
        return float(sum(self.target_scores) / len(self.target_scores))


@dataclass(slots=True)
class InputTrace:
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    batch_size: int | None = None
    attribution_available: bool = False
    attribution_shape: tuple[int, ...] | None = None
    attribution_values: list[float] | None = None
    attribution_mean_abs: float | None = None
    attribution_max_abs: float | None = None
    feature_values: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayerTrace:
    name: str
    layer_type: str
    execution_index: int
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    device: str = "unknown"
    dtype: str | None = None
    parameter_count: int = 0
    activation_mean: float | None = None
    activation_variance: float | None = None
    activation_std: float | None = None
    activation_energy: float | None = None
    mean_energy: float | None = None
    l1_magnitude: float | None = None
    absolute_mean: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    sparsity: float | None = None
    positive_fraction: float | None = None
    negative_fraction: float | None = None
    activation_dispersion_entropy: float | None = None
    gradient_norm: float | None = None
    gradient_mean_abs: float | None = None
    activation_gradient_influence: float | None = None
    activation_gradient_signed: float | None = None
    execution_time_ms: float | None = None
    representation: list[list[float]] | None = None
    unit_matrix: list[list[float]] | None = None
    unit_activations: list[float] | None = None
    unit_gradients: list[float] | None = None
    unit_contributions: list[float] | None = None
    attention: list[list[list[float]]] | None = None
    attention_shape: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gradient_signal(self) -> bool:
        return self.activation_gradient_influence is not None

    @property
    def identifier(self) -> str:
        """DarkSide-compatible layer identifier (same as ``name`` in 0.2.x)."""

        return self.name

    @property
    def timing_ms(self) -> float | None:
        return self.execution_time_ms

    @property
    def unit_signed_mean(self) -> list[float]:
        return list(self.unit_contributions or [])

    @property
    def unit_energy(self) -> list[float]:
        if self.unit_activations is not None:
            return [abs(float(value)) for value in self.unit_activations]
        if self.unit_matrix:
            return [
                float(sum(abs(float(value)) for value in row) / max(len(row), 1))
                for row in self.unit_matrix
            ]
        return []

    @property
    def activation(self) -> _ActivationView:
        if self.mean_energy is not None and self.mean_energy >= 0:
            rms = float(self.mean_energy) ** 0.5
        elif self.activation_energy is not None and self.activation_energy >= 0:
            rms = float(self.activation_energy) ** 0.5
        elif self.absolute_mean is not None:
            rms = abs(float(self.absolute_mean))
        else:
            rms = 0.0
        return _ActivationView(rms=rms)

    @property
    def gradient(self) -> _GradientView | None:
        if self.unit_contributions is None and self.unit_gradients is None:
            return None
        influence = self.unit_contributions if self.unit_contributions is not None else self.unit_gradients
        return _GradientView(unit_influence=list(influence or []))


@dataclass(slots=True)
class CausalInterventionTrace:
    layer_name: str
    strategy: str
    baseline_target_score: float
    intervened_target_score: float
    score_delta: float
    relative_delta: float | None
    prediction_before: int | None
    prediction_after: int | None
    target_index: int | None
    successful: bool
    notes: str = ""


@dataclass(slots=True)
class NeuralTrace:
    layers: list[LayerTrace] = field(default_factory=list)
    prediction: PredictionTrace | None = None
    inputs: InputTrace | None = None
    target: int | list[int] | None = None
    mode: str = "fast"
    metadata: dict[str, Any] = field(default_factory=dict)
    causal_interventions: list[CausalInterventionTrace] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: str = "1.0"

    @property
    def framework(self) -> Any:
        return self.metadata.get("framework")

    @property
    def model_name(self) -> Any:
        return self.metadata.get("model_class") or self.metadata.get("model_name")

    @property
    def input_attributions(self) -> list[InputAttributionView]:
        """Synthesise DarkSide-style attribution records from InputTrace."""

        if self.inputs is None or not self.inputs.attribution_values:
            return []
        values = [abs(float(value)) for value in self.inputs.attribution_values]
        return [InputAttributionView(tensor_index=0, absolute_influence=values)]

    def validate(self) -> None:
        indices = [layer.execution_index for layer in self.layers]
        if indices != sorted(indices):
            raise TraceValidationError("Layer traces are not ordered by execution_index")
        if len(set(indices)) != len(indices):
            raise TraceValidationError("Layer execution indices must be unique")
        if self.prediction is None:
            raise TraceValidationError("A neural trace requires a prediction summary")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NeuralTrace:
        try:
            layers = [LayerTrace(**item) for item in payload.get("layers", [])]
            prediction_payload = payload.get("prediction")
            prediction = PredictionTrace(**prediction_payload) if prediction_payload else None
            input_payload = payload.get("inputs")
            inputs = InputTrace(**input_payload) if input_payload else None
            causal = [
                CausalInterventionTrace(**item)
                for item in payload.get("causal_interventions", [])
            ]
            trace = cls(
                layers=layers,
                prediction=prediction,
                inputs=inputs,
                target=payload.get("target"),
                mode=payload.get("mode", "fast"),
                metadata=dict(payload.get("metadata", {})),
                causal_interventions=causal,
                trace_id=payload.get("trace_id", uuid4().hex),
                created_at_utc=payload.get(
                    "created_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
                ),
                schema_version=payload.get("schema_version", "1.0"),
            )
            trace.validate()
            return trace
        except (TypeError, ValueError, KeyError) as error:
            raise TraceValidationError(f"Invalid VRAVEN trace payload: {error}") from error

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> NeuralTrace:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TraceValidationError("A VRAVEN trace file must contain a JSON object")
        return cls.from_dict(payload)
