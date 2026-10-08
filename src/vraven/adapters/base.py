"""Adapter protocol shared by model-framework integrations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vraven.core.trace import NeuralTrace


@runtime_checkable
class ModelAdapter(Protocol):
    framework_name: str

    def trace(self, inputs: Any, target: Any = None, metadata: dict[str, Any] | None = None) -> NeuralTrace:
        """Execute one instrumented inference and return a serialisable trace."""

    def take_full_tensors(self) -> dict[str, Any]:
        """Return explicitly retained ephemeral tensors, then clear adapter ownership."""
