"""Common immutable contracts for VRAVEN DarkSide analyses.

The extension separates measured values, interpretation metadata and visual
presentation.  Result objects are JSON serialisable and can be rendered in the
canonical VRAVEN dark or publication themes without changing the data mapping.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values to JSON-safe built-ins."""

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(slots=True)
class AnalysisResult:
    """Base result shared by every DarkSide analysis.

    ``analysis`` is a stable machine-readable identifier. ``metrics`` contains
    scalar summary values. ``tables`` stores arrays or row records used by
    figures.  Causal and observational claims are kept explicit in ``metadata``.
    """

    analysis: str
    metrics: dict[str, Any]
    tables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    schema_version: str = "vraven.darkside.v1"

    def __post_init__(self) -> None:
        if not self.analysis.strip():
            raise ValueError("analysis identifier cannot be empty")

    def summary(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            **_jsonable(self.metrics),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    def figure(self, *, mode: str = "publication_light"):
        """Render one standalone figure through the shared VRAVEN theme."""

        from .visualisation import figure_for_result

        return figure_for_result(self, mode=mode)


@dataclass(frozen=True, slots=True)
class UnitGroup:
    """A declared intervention group inside one layer output invocation."""

    layer_name: str
    indices: tuple[int, ...]
    axis: int = -1
    call_index: int = 0
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.layer_name.strip():
            raise ValueError("layer_name cannot be empty")
        if not self.indices:
            raise ValueError("UnitGroup requires at least one unit index")
        if any(index < 0 for index in self.indices):
            raise ValueError("UnitGroup indices cannot be negative")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("UnitGroup indices must be unique")
        if self.call_index < 0:
            raise ValueError("call_index cannot be negative")

    @property
    def layer_identifier(self) -> str:
        return f"{self.layer_name}#{self.call_index}"

    @property
    def identifier(self) -> str:
        label = self.label or ",".join(str(value) for value in self.indices)
        return f"{self.layer_identifier}[{label}]"


@dataclass(frozen=True, slots=True)
class InputFeatureGroup:
    """A group of flattened input coordinates used for shortcut interventions."""

    indices: tuple[int, ...]
    tensor_index: int = 0
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("InputFeatureGroup requires at least one feature index")
        if any(index < 0 for index in self.indices):
            raise ValueError("Input feature indices cannot be negative")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("Input feature indices must be unique")
        if self.tensor_index < 0:
            raise ValueError("tensor_index cannot be negative")

    @property
    def identifier(self) -> str:
        return self.label or f"input[{','.join(str(value) for value in self.indices)}]"


@dataclass(frozen=True, slots=True)
class FeatureRecommendation:
    """One feature recommendation for a user community or scientific question."""

    name: str
    priority: str
    purpose: str
    minimum_data: tuple[str, ...]
    cost: str
    claim_type: str
