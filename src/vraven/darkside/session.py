"""Convenience session for live-model DarkSide analyses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vraven.core.config import VRavenConfig

from .anatomy import inspect_internal_anatomy
from .contracts import AnalysisResult, InputFeatureGroup, UnitGroup
from .decision_circuitry import discover_decision_circuitry
from .edge_cards import enrich_circuit_wiring_edges
from .mediation import analyse_activation_mediation
from .shortcut import detect_shortcut_reliance
from .synergy import analyse_causal_synergy
from .wiring import build_circuit_wiring


class DarkSideSession:
    """Bind a live PyTorch model, sample and target for intervention analyses."""

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: Any,
        *,
        target: int | None,
        config: VRavenConfig | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("DarkSideSession currently requires a live torch.nn.Module")
        self.model = model
        self.inputs = inputs
        self.target = target
        self.config = config or VRavenConfig(mode="gradient")

    def decision_circuitry(self, **kwargs: Any):
        return discover_decision_circuitry(
            self.model,
            self.inputs,
            target=self.target,
            config=self.config,
            **kwargs,
        )

    def shortcut_detection(
        self,
        feature_groups: Sequence[InputFeatureGroup],
        **kwargs: Any,
    ):
        return detect_shortcut_reliance(
            self.model,
            self.inputs,
            target=self.target,
            feature_groups=feature_groups,
            config=self.config,
            **kwargs,
        )

    def causal_synergy(self, groups: Sequence[UnitGroup], **kwargs: Any):
        return analyse_causal_synergy(
            self.model,
            self.inputs,
            target=self.target,
            groups=groups,
            config=self.config,
            **kwargs,
        )

    def activation_mediation(
        self,
        corrupted_inputs: Any,
        *,
        layer_names: Sequence[str],
        **kwargs: Any,
    ):
        return analyse_activation_mediation(
            self.model,
            self.inputs,
            corrupted_inputs,
            target=self.target,
            layer_names=layer_names,
            config=self.config,
            **kwargs,
        )

    def internal_anatomy(self, **kwargs: Any):
        return inspect_internal_anatomy(
            self.model,
            self.inputs,
            target=self.target,
            config=self.config,
            **kwargs,
        )

    def circuit_wiring(
        self,
        circuitry: AnalysisResult,
        *,
        enrich: bool = True,
        reference_batch: Any | None = None,
        class_names: Any | None = None,
        top_k_intervene: int = 24,
        bootstrap_samples: int = 100,
        concept_associations: Any | None = None,
        **kwargs: Any,
    ):
        wiring = build_circuit_wiring(
            self.model,
            self.inputs,
            circuitry,
            **kwargs,
        )
        if not enrich:
            return wiring
        return enrich_circuit_wiring_edges(
            self.model,
            self.inputs,
            wiring,
            target=self.target,
            class_names=class_names,
            reference_batch=reference_batch,
            top_k_intervene=top_k_intervene,
            bootstrap_samples=bootstrap_samples,
            concept_associations=concept_associations,
            config=self.config,
        )
