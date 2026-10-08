"""Public orchestration API for VRAVEN explanations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

import torch

from vraven.adapters.registry import resolve_adapter
from vraven.causal.validation import validate_layer_interventions
from vraven.core.config import VRavenConfig
from vraven.report.report import VRavenReport


class Explainer:
    """Reusable model-bound VRAVEN explainer."""

    def __init__(
        self,
        model: Any,
        config: VRavenConfig | None = None,
        *,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.config = config or VRavenConfig()
        self.class_names = class_names
        self.adapter = resolve_adapter(model, self.config, class_names=class_names)

    def explain(
        self,
        inputs: Any,
        target: int | Sequence[int] | torch.Tensor | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VRavenReport:
        trace = self.adapter.trace(inputs=inputs, target=target, metadata=metadata)
        if self.config.mode == "causal":
            trace.causal_interventions = validate_layer_interventions(
                self.model,
                inputs,
                trace,
                target=target,
                top_k=self.config.causal_top_k,
                strategy=self.config.causal_strategy,
                scale=self.config.causal_scale,
                auto_move_inputs=self.config.auto_move_inputs,
            )
        take_full_tensors = getattr(self.adapter, "take_full_tensors", None)
        full_tensors = take_full_tensors() if callable(take_full_tensors) else {}
        return VRavenReport(trace=trace, full_tensors=full_tensors)

    __call__ = explain


def explain(
    model: Any,
    inputs: Any,
    target: int | Sequence[int] | torch.Tensor | None = None,
    *,
    mode: str = "fast",
    config: VRavenConfig | None = None,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **config_overrides: Any,
) -> VRavenReport:
    """Explain one model invocation using the simple functional API."""

    allowed = {item.name for item in fields(VRavenConfig)}
    if config is not None:
        payload = config.to_dict()
        payload["mode"] = mode
        payload.update(config_overrides)
        resolved = VRavenConfig(**{key: value for key, value in payload.items() if key in allowed})
    else:
        resolved = VRavenConfig(mode=mode, **config_overrides)  # type: ignore[arg-type]
    return Explainer(model, resolved, class_names=class_names).explain(
        inputs,
        target,
        metadata=metadata,
    )
