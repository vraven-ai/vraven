"""End-to-end compiler-assisted causal decompilation pipeline for PyTorch models."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils import _pytree as pytree

from .capture import CaptureConfig, CapturedModel, ModelCaptureRouter
from .causal_graph import CausalGraphConfig, CausalMechanismBuilder, CausalMechanismGraph
from .certificate import CertificateBuilder, MechanismCertificate
from .concepts import (
    ConceptAtlas,
    ConceptDiscoveryConfig,
    ConceptDiscoveryEngine,
    ground_concepts_from_input_groups,
)
from .contrastive import ContrastiveExplainer, ContrastiveExplanation
from .counterfactual import (
    CounterfactualConfig,
    CounterfactualProof,
    CounterfactualProofEngine,
    DomainValidator,
)
from .evolution import ConceptEvolution, ConceptEvolutionBuilder, EvolutionConfig
from .profiler import MechanismProfiler, ProfilerConfig, ProfilerReport
from .programme import MechanismProgram, ProgrammeCompiler, ProgrammeConfig
from .runtime import FXValueRecorder, extract_logits, predicted_class, target_margin


@dataclass(slots=True)
class DecompileConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    concepts: ConceptDiscoveryConfig = field(default_factory=ConceptDiscoveryConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    causal: CausalGraphConfig = field(default_factory=CausalGraphConfig)
    programme: ProgrammeConfig = field(default_factory=ProgrammeConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    run_counterfactual: bool = True
    run_profiler: bool = True
    promote_association_labels: bool = False
    grounding_permutations: int = 200
    capture_example_batch: int = 2
    seed: int = 17

    @classmethod
    def quick(cls) -> DecompileConfig:
        config = cls()
        config.concepts.method = "pca"
        config.concepts.max_nodes = 4
        config.concepts.concepts_per_node = 4
        config.causal.max_source_concepts = 6
        config.causal.max_edges = 10
        config.causal.max_samples = 48
        config.causal.bootstrap_repeats = 60
        config.programme.maximum_concepts = 10
        config.programme.maximum_rules = 8
        config.counterfactual.maximum_concepts = 6
        config.counterfactual.maximum_changes = 2
        config.counterfactual.maximum_verified_candidates = 16
        config.profiler.measured_runs = 4
        return config

    @classmethod
    def publication(cls) -> DecompileConfig:
        config = cls()
        config.concepts.max_nodes = 8
        config.concepts.concepts_per_node = 8
        config.causal.max_source_concepts = 16
        config.causal.max_edges = 48
        config.causal.max_samples = 128
        config.causal.bootstrap_repeats = 500
        config.programme.maximum_concepts = 20
        config.programme.maximum_rules = 16
        config.counterfactual.maximum_concepts = 12
        config.counterfactual.maximum_changes = 3
        config.counterfactual.maximum_verified_candidates = 80
        config.profiler.measured_runs = 12
        return config


@dataclass(slots=True)
class DecompilationResult:
    captured: CapturedModel
    atlas: ConceptAtlas
    evolution: ConceptEvolution
    causal_graph: CausalMechanismGraph
    programme: MechanismProgram
    contrastive: ContrastiveExplanation
    counterfactual: CounterfactualProof | None
    certificate: MechanismCertificate
    profiler: ProfilerReport | None
    focus_concept_values: np.ndarray
    focus_logits: np.ndarray
    target: int
    alternative: int | None
    class_names: dict[int, str]
    timings_seconds: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.captured.ir.model_class,
            "capture_route": self.captured.ir.capture_route.value,
            "target": self.target,
            "target_label": self.class_names.get(self.target, f"class {self.target}"),
            "alternative": self.alternative,
            "concepts": len(self.atlas.concepts),
            "concept_nodes": len(self.atlas.concept_sets),
            "evolution_edges": len(self.evolution.edges),
            "causal_nodes": len(self.causal_graph.nodes),
            "causal_edges": len(self.causal_graph.edges),
            "programme_rules": len(self.programme.rules),
            "programme_validation_r2": self.programme.fidelity.validation_r2,
            "programme_interventional_sign_agreement": (
                self.programme.fidelity.interventional_sign_agreement
            ),
            "counterfactual_found": bool(self.counterfactual and self.counterfactual.found),
            "counterfactual_model_flip": (
                self.counterfactual.behaviourally_verified if self.counterfactual else False
            ),
            "operator_coverage": self.certificate.coverage.operator_coverage,
            "causal_semantic_coverage": self.certificate.coverage.causal_semantic_coverage,
            "warnings": self.certificate.warnings,
            "timings_seconds": self.timings_seconds,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "target": self.target,
            "alternative": self.alternative,
            "class_names": self.class_names,
            "focus_concept_values": self.focus_concept_values.tolist(),
            "focus_logits": self.focus_logits.tolist(),
            "timings_seconds": self.timings_seconds,
            "metadata": self.metadata,
        }

    def save(self, directory: str | Path, *, include_ir_graph_code: bool = True) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        ir_payload = self.captured.ir.to_dict()
        if not include_ir_graph_code:
            ir_payload.get("metadata", {}).pop("graph_code", None)
        _write_json(destination / "mechanism_ir.json", ir_payload)
        self.atlas.save(destination / "concept_atlas")
        _write_json(destination / "concept_evolution.json", self.evolution.to_dict())
        _write_json(destination / "causal_mechanism_graph.json", self.causal_graph.to_dict())
        _write_json(destination / "executable_programme.json", self.programme.to_dict())
        (destination / "executable_programme.txt").write_text(
            self.programme.to_pseudocode(), encoding="utf-8"
        )
        (destination / "decompiled_program.py").write_text(
            self.programme.to_python_code(), encoding="utf-8"
        )
        (destination / "decompiled_program.c").write_text(
            self.programme.to_c_code(), encoding="utf-8"
        )
        _write_json(destination / "contrastive_explanation.json", self.contrastive.to_dict())
        from .feature_audit import build_reverse_feature_audits

        _write_json(
            destination / "reverse_feature_audits.json",
            {name: audit.to_dict() for name, audit in build_reverse_feature_audits(self).items()},
        )
        if self.counterfactual is not None:
            _write_json(destination / "counterfactual_proof.json", self.counterfactual.to_dict())
        self.certificate.save(destination / "mechanism_certificate.json")
        if self.profiler is not None:
            self.profiler.save(destination / "mechanism_profile.json")
        _write_json(destination / "decompilation_manifest.json", self.to_manifest())
        return destination

    def figure(self, visual: str, *, mode: str = "interactive_dark", **kwargs: Any):
        from .visuals import render_visual

        return render_visual(self, visual, mode=mode, **kwargs)

    def write_report(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        initial_mode: str = "interactive_dark",
        include_plotlyjs: bool = True,
    ) -> Path:
        """Write exactly one Decision Proof report in HTML, PDF or JSON format."""

        from .report import DecisionProofReport

        return DecisionProofReport(self).write(
            path,
            format=format,
            initial_mode=initial_mode,
            include_plotlyjs=include_plotlyjs,
        )

    def write_feature_report(
        self,
        visual: str,
        path: str | Path,
        *,
        format: str | None = None,
        initial_mode: str = "interactive_dark",
        include_plotlyjs: bool = True,
    ) -> Path:
        """Write one standalone Reverse feature with its critical scientific audit."""

        from .report import DecisionProofReport

        return DecisionProofReport(self).write_feature(
            visual,
            path,
            format=format,
            initial_mode=initial_mode,
            include_plotlyjs=include_plotlyjs,
        )


class VRavenDecompiler:
    """Recover a tested high-level mechanism from a previously unseen PyTorch model."""

    def __init__(self, config: DecompileConfig | None = None) -> None:
        self.config = config or DecompileConfig()

    def reverse(
        self,
        model: nn.Module,
        reference_args: Sequence[Any],
        *,
        reference_kwargs: Mapping[str, Any] | None = None,
        labels: Sequence[int] | np.ndarray | Tensor | None = None,
        sample_ids: Sequence[str] | None = None,
        focus_args: Sequence[Any] | None = None,
        focus_kwargs: Mapping[str, Any] | None = None,
        target: int | None = None,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
        input_groups: Mapping[str, Sequence[int]] | None = None,
        validated_input_groups: Sequence[str] | None = None,
        domain_validator: DomainValidator | None = None,
    ) -> DecompilationResult:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        reference_kwargs = dict(reference_kwargs or {})
        focus_args = (
            tuple(focus_args)
            if focus_args is not None
            else _slice_batch(tuple(reference_args), 0, 1)
        )
        focus_kwargs = (
            dict(focus_kwargs)
            if focus_kwargs is not None
            else _slice_batch_mapping(reference_kwargs, 0, 1)
        )
        timings: dict[str, float] = {}

        capture_batch = max(
            1, min(self.config.capture_example_batch, _batch_size(reference_args, reference_kwargs))
        )
        capture_args = _slice_batch(tuple(reference_args), 0, capture_batch)
        capture_kwargs = _slice_batch_mapping(reference_kwargs, 0, capture_batch)
        captured = _timed(
            timings,
            "capture",
            lambda: ModelCaptureRouter(self.config.capture).capture(
                model,
                capture_args,
                capture_kwargs,
            ),
        )
        if captured.graph_module is None:
            raise ValueError(
                "The neural decompiler requires an export or FX graph. The fallback capture "
                f"route {captured.ir.capture_route.value!r} does not support graph rewriting."
            )
        focus_record = FXValueRecorder(
            captured.graph_module,
            detach=False,
            move_to_cpu=False,
        ).execute(focus_args, focus_kwargs)
        focus_logits_tensor = extract_logits(focus_record.output, self.config.causal.output_path)
        predicted = int(predicted_class(focus_logits_tensor).detach().cpu().reshape(-1)[0].item())
        target = predicted if target is None else int(target)
        alternative = _strongest_alternative(focus_logits_tensor, target)
        names = _normalise_class_names(class_names, int(focus_logits_tensor.shape[-1]))

        atlas = _timed(
            timings,
            "concept_discovery",
            lambda: ConceptDiscoveryEngine(self.config.concepts).discover(
                captured,
                tuple(reference_args),
                reference_kwargs=reference_kwargs,
                labels=labels,
                sample_ids=sample_ids,
            ),
        )
        if input_groups:
            first_input = _first_batched_input(reference_args, reference_kwargs)
            if isinstance(first_input, Tensor) and first_input.ndim == 2:
                atlas = _timed(
                    timings,
                    "semantic_grounding",
                    lambda: ground_concepts_from_input_groups(
                        atlas,
                        first_input,
                        input_groups,
                        permutations=self.config.grounding_permutations,
                        seed=self.config.seed,
                        promote_labels=self.config.promote_association_labels,
                        validated_labels=validated_input_groups,
                    ),
                )

        evolution = _timed(
            timings,
            "concept_evolution",
            lambda: ConceptEvolutionBuilder(self.config.evolution).build(atlas),
        )
        causal_graph = _timed(
            timings,
            "causal_mechanism",
            lambda: CausalMechanismBuilder(self.config.causal).build(
                captured,
                atlas,
                evolution,
                tuple(reference_args),
                reference_kwargs=reference_kwargs,
                target=target,
            ),
        )
        reference_record = FXValueRecorder(
            captured.graph_module,
            detach=True,
            move_to_cpu=True,
        ).execute(tuple(reference_args), reference_kwargs)
        reference_logits = extract_logits(reference_record.output, self.config.causal.output_path)
        margins = target_margin(reference_logits, target).detach().cpu().numpy()
        programme = _timed(
            timings,
            "programme_compilation",
            lambda: ProgrammeCompiler(self.config.programme).compile(
                atlas,
                causal_graph,
                margins,
                target=target,
            ),
        )
        focus_values = _focus_concept_values(atlas, programme, focus_record.values)
        contrastive = _timed(
            timings,
            "contrastive_explanation",
            lambda: ContrastiveExplainer().explain(
                programme,
                focus_values,
                causal_graph,
                alternative=alternative,
                class_names=names,
            ),
        )
        counterfactual: CounterfactualProof | None = None
        if self.config.run_counterfactual:
            counterfactual = _timed(
                timings,
                "counterfactual_proof",
                lambda: CounterfactualProofEngine(self.config.counterfactual).prove(
                    captured,
                    atlas,
                    programme,
                    causal_graph,
                    focus_args,
                    sample_kwargs=focus_kwargs,
                    target=target,
                    domain_validator=domain_validator,
                ),
            )
        profiler: ProfilerReport | None = None
        if self.config.run_profiler:
            profiler = _timed(
                timings,
                "profiling",
                lambda: MechanismProfiler(self.config.profiler).profile(
                    captured,
                    focus_args,
                    kwargs=focus_kwargs,
                    causal_graph=causal_graph,
                ),
            )
        certificate = _timed(
            timings,
            "certificate",
            lambda: CertificateBuilder().build(
                captured,
                atlas,
                causal_graph,
                programme,
                counterfactual,
                target=target,
            ),
        )
        return DecompilationResult(
            captured=captured,
            atlas=atlas,
            evolution=evolution,
            causal_graph=causal_graph,
            programme=programme,
            contrastive=contrastive,
            counterfactual=counterfactual,
            certificate=certificate,
            profiler=profiler,
            focus_concept_values=focus_values,
            focus_logits=focus_logits_tensor.detach().cpu().numpy(),
            target=target,
            alternative=alternative,
            class_names=names,
            timings_seconds=timings,
            metadata={
                "objective": (
                    "Recover a compact, executable, contrastive and intervention-tested high-level "
                    "programme within an explicit validity envelope."
                ),
                "torch_version": torch.__version__,
                "claim_boundary": (
                    "The result is the smallest tested high-level programme found by the configured "
                    "search. It is not asserted to be the unique true explanation."
                ),
            },
        )


def reverse(
    model: nn.Module,
    reference_args: Sequence[Any],
    **kwargs: Any,
) -> DecompilationResult:
    """Convenience entry point for the VRAVEN causal decompiler."""

    config = kwargs.pop("config", None)
    return VRavenDecompiler(config).reverse(model, reference_args, **kwargs)


def _focus_concept_values(
    atlas: ConceptAtlas,
    programme: MechanismProgram,
    recorded_values: Mapping[str, Any],
) -> np.ndarray:
    concept_map = atlas.concept_map()
    values: list[float] = []
    for concept_id in programme.concept_ids:
        concept = concept_map[concept_id]
        tensor = _first_tensor(recorded_values.get(concept.node_id))
        if tensor is None:
            raise ValueError(f"No focus activation was recorded for {concept.node_id!r}")
        concept_set = atlas.set_for_node(concept.node_id)
        coefficient = concept_set.score_component(tensor, concept.index)
        values.append(float(coefficient.detach().mean().cpu().item()))
    return np.asarray(values, dtype=float).reshape(1, -1)


def _strongest_alternative(logits: Tensor, target: int) -> int | None:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[-1] <= 1:
        return 1 - target if target in (0, 1) else None
    mean_logits = logits.detach().mean(dim=0).cpu()
    mean_logits[target] = float("-inf")
    return int(mean_logits.argmax().item())


def _normalise_class_names(
    class_names: Mapping[int, str] | Sequence[str] | None,
    classes: int,
) -> dict[int, str]:
    if class_names is None:
        return {index: f"class {index}" for index in range(max(2, classes))}
    if isinstance(class_names, Mapping):
        return {int(key): str(value) for key, value in class_names.items()}
    return {index: str(value) for index, value in enumerate(class_names)}


def _batch_size(args: Sequence[Any], kwargs: Mapping[str, Any]) -> int:
    leaves, _ = pytree.tree_flatten((tuple(args), dict(kwargs)))
    sizes = [
        int(item.shape[0])
        for item in leaves
        if isinstance(item, (Tensor, np.ndarray)) and item.ndim > 0
    ]
    if not sizes:
        raise ValueError("Reference inputs must contain at least one batched tensor or array")
    if len(set(sizes)) != 1:
        raise ValueError(f"Reference batch dimensions disagree: {sizes}")
    return sizes[0]


def _slice_batch(args: tuple[Any, ...], start: int, end: int) -> tuple[Any, ...]:
    def slice_item(item: Any) -> Any:
        if isinstance(item, (Tensor, np.ndarray)) and item.ndim > 0:
            return item[start:end]
        return item

    return tuple(pytree.tree_map(slice_item, args))


def _slice_batch_mapping(kwargs: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    def slice_item(item: Any) -> Any:
        if isinstance(item, (Tensor, np.ndarray)) and item.ndim > 0:
            return item[start:end]
        return item

    return dict(pytree.tree_map(slice_item, dict(kwargs)))


def _first_batched_input(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Tensor | None:
    leaves, _ = pytree.tree_flatten((tuple(args), dict(kwargs)))
    for item in leaves:
        if isinstance(item, Tensor) and item.ndim > 0:
            return item
        if isinstance(item, np.ndarray) and item.ndim > 0:
            return torch.as_tensor(item)
    return None


def _first_tensor(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    leaves, _ = pytree.tree_flatten(value)
    return next((item for item in leaves if isinstance(item, Tensor)), None)


def _timed(timings: dict[str, float], name: str, function):
    started = time.perf_counter()
    result = function()
    timings[name] = time.perf_counter() - started
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
