"""Sound interval certificates for local L-infinity classification robustness.

The verifier propagates an axis-aligned input box through a supported ``torch.fx``
graph.  A sample is certified only when the lower bound on the target logit exceeds
the upper bound on every competing logit.  Inconclusive bounds are never interpreted
as counterexamples.
"""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.fx import GraphModule, Node, symbolic_trace

from .verification_protocol import InputDomainSpec, resolve_input_domain

VerificationStatus = Literal["certified", "inconclusive", "unsupported"]
Bound = tuple[Tensor, Tensor]
BoundTransform = Callable[[Bound], Bound]
PairedBoundTransform = Callable[[Bound, Bound, Bound], tuple[Bound, Bound]]


@dataclass(slots=True, frozen=True)
class _PairedInterval:
    """Two executions and a relational enclosure of ``original - intervened``."""

    original: Bound
    intervened: Bound
    difference: Bound


class UnsupportedVerificationOperator(RuntimeError):
    """Raised when a graph operation lacks a sound interval transformer."""


@dataclass(slots=True)
class LayerIntervalRecord:
    name: str
    operator: str
    lower_min: float
    upper_max: float
    maximum_width: float
    unstable_fraction: float | None = None


@dataclass(slots=True)
class FormalVerificationPoint:
    epsilon: float
    status: VerificationStatus
    certified: bool
    clean_prediction: int
    target: int
    clean_margin: float
    certified_margin_lower_bound: float | None
    worst_competitor: int | None
    competitor_margin_lower_bounds: dict[str, float] = field(default_factory=dict)
    output_lower_bounds: list[float] = field(default_factory=list)
    output_upper_bounds: list[float] = field(default_factory=list)
    layer_intervals: list[LayerIntervalRecord] = field(default_factory=list)
    unsupported_reason: str | None = None


@dataclass(slots=True)
class CertifiedAccuracyPoint:
    epsilon: float
    n_samples: int
    n_clean_correct: int
    n_certified: int
    certified_accuracy: float
    clean_conditional_certified_accuracy: float
    inconclusive_clean_correct: int
    unsupported: int
    per_class: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class FormalRobustnessCertificate:
    """Machine-readable proof ledger for one sample and an optional cohort."""

    model_name: str
    property: str
    norm: str
    input_domain: tuple[float | None, float | None]
    target: int
    sample_points: list[FormalVerificationPoint]
    certified_radius_lower_bound: float
    first_inconclusive_epsilon: float | None
    cohort_points: list[CertifiedAccuracyPoint] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    input_spec: InputDomainSpec | None = None
    verifier: str = "vraven_ibp_fx_v1"
    proof_semantics: str = (
        "For every input in the declared L-infinity box, optionally intersected with "
        "the recorded input domain, interval propagation proves the target-logit lower "
        "bound strictly exceeds every competing-logit upper bound."
    )
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    def write_html(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_certificate_html(self), encoding="utf-8")
        return destination


class IntervalBoundVerifier:
    """Certify local L-infinity robustness for a supported PyTorch FX graph."""

    def __init__(self, model: nn.Module, *, model_name: str | None = None) -> None:
        self.model = model
        self.model_name = model_name or type(model).__name__
        self.device = _module_device(model)
        self.dtype = _module_dtype(model)
        self.graph: GraphModule | None = None
        self.trace_error: str | None = None
        try:
            self.graph = symbolic_trace(model)
            self.graph.eval()
        except Exception as error:  # noqa: BLE001
            self.trace_error = f"{type(error).__name__}: {error}"

    def verify_sample(
        self,
        x: Tensor,
        *,
        epsilon: float,
        target: int | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> FormalVerificationPoint:
        """Verify one fixed-epsilon local classification property."""

        epsilon = float(epsilon)
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be non-negative and finite")
        if clip_min is not None and not math.isfinite(clip_min):
            raise ValueError("clip_min must be finite when supplied")
        if clip_max is not None and not math.isfinite(clip_max):
            raise ValueError("clip_max must be finite when supplied")
        if clip_min is not None and clip_max is not None and clip_min >= clip_max:
            raise ValueError("clip_min must be smaller than clip_max")
        x_in = x.detach().to(device=self.device, dtype=self.dtype)
        if x_in.ndim == 0:
            x_in = x_in.reshape(1, 1)
        elif x_in.shape[0] != 1:
            x_in = x_in.unsqueeze(0)
        if x_in.shape[0] != 1:
            raise ValueError("verify_sample requires exactly one sample")
        if clip_min is not None and bool((x_in < clip_min).any()):
            raise ValueError("clean input lies outside the declared input domain")
        if clip_max is not None and bool((x_in > clip_max).any()):
            raise ValueError("clean input lies outside the declared input domain")

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                clean_logits = _normalise_logits(self.model(x_in)).double()
            clean_prediction = int(clean_logits.argmax(dim=-1)[0].item())
            decision_target = clean_prediction if target is None else int(target)
            if decision_target < 0 or decision_target >= clean_logits.shape[-1]:
                raise ValueError("target is outside the model class range")
            clean_margin, _ = _clean_margin(clean_logits[0], decision_target)
            if self.graph is None:
                return _unsupported_point(
                    epsilon,
                    clean_prediction,
                    decision_target,
                    clean_margin,
                    f"FX tracing failed: {self.trace_error}",
                )
            lower = x_in.double() - epsilon
            upper = x_in.double() + epsilon
            if clip_min is not None:
                lower = torch.clamp_min(lower, clip_min)
                upper = torch.clamp_min(upper, clip_min)
            if clip_max is not None:
                lower = torch.clamp_max(lower, clip_max)
                upper = torch.clamp_max(upper, clip_max)
            try:
                output_bound, records = self._propagate((lower, upper))
                output_lower, output_upper = _normalise_logit_bounds(output_bound)
            except UnsupportedVerificationOperator as error:
                return _unsupported_point(
                    epsilon,
                    clean_prediction,
                    decision_target,
                    clean_margin,
                    str(error),
                )

            target_lower = output_lower[0, decision_target]
            margin_bounds: dict[str, float] = {}
            worst_competitor: int | None = None
            worst_bound = float("inf")
            for competitor in range(int(output_lower.shape[-1])):
                if competitor == decision_target:
                    continue
                value = float((target_lower - output_upper[0, competitor]).item())
                margin_bounds[str(competitor)] = value
                if value < worst_bound:
                    worst_bound = value
                    worst_competitor = competitor
            certified = bool(margin_bounds) and worst_bound > 0.0
            return FormalVerificationPoint(
                epsilon=epsilon,
                status="certified" if certified else "inconclusive",
                certified=certified,
                clean_prediction=clean_prediction,
                target=decision_target,
                clean_margin=clean_margin,
                certified_margin_lower_bound=(worst_bound if margin_bounds else None),
                worst_competitor=worst_competitor,
                competitor_margin_lower_bounds=margin_bounds,
                output_lower_bounds=[float(value) for value in output_lower[0].tolist()],
                output_upper_bounds=[float(value) for value in output_upper[0].tolist()],
                layer_intervals=records,
            )
        finally:
            self.model.train(was_training)

    def certify(
        self,
        x: Tensor,
        *,
        target: int | None = None,
        epsilon_grid: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1),
        clip_min: float | None = None,
        clip_max: float | None = None,
        cohort_x: Tensor | None = None,
        cohort_y: Tensor | list[int] | None = None,
        selected_indices: list[int] | None = None,
        radius_search_steps: int = 24,
        input_spec: InputDomainSpec | None = None,
    ) -> FormalRobustnessCertificate:
        """Build a formal certificate curve for a sample and optional cohort."""

        clip_min, clip_max = resolve_input_domain(input_spec, clip_min=clip_min, clip_max=clip_max)
        epsilons = sorted({float(value) for value in epsilon_grid})
        if not epsilons:
            raise ValueError("epsilon_grid must contain at least one value")
        sample_points = [
            self.verify_sample(
                x,
                epsilon=epsilon,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            for epsilon in epsilons
        ]
        if radius_search_steps < 1:
            raise ValueError("radius_search_steps must be positive")
        radius_lower, radius_upper, search_trace = self._search_certified_radius(
            x,
            target=target,
            upper=max(epsilons),
            clip_min=clip_min,
            clip_max=clip_max,
            steps=int(radius_search_steps),
        )
        first_inconclusive = next(
            (point.epsilon for point in sample_points if not point.certified), None
        )
        first_inconclusive_point = next(
            (point for point in sample_points if not point.certified), None
        )
        cohort_points: list[CertifiedAccuracyPoint] = []
        indices: list[int] = []
        if cohort_x is not None:
            if cohort_y is None:
                raise ValueError("cohort_y is required for certified accuracy")
            labels = (
                [int(value) for value in cohort_y.detach().reshape(-1).tolist()]
                if isinstance(cohort_y, Tensor)
                else [int(value) for value in cohort_y]
            )
            if len(labels) != int(cohort_x.shape[0]):
                raise ValueError("cohort_y must contain one label per sample")
            indices = (
                list(range(int(cohort_x.shape[0])))
                if selected_indices is None
                else [int(value) for value in selected_indices]
            )
            if not indices or min(indices) < 0 or max(indices) >= int(cohort_x.shape[0]):
                raise ValueError("selected_indices is empty or out of range")
            for epsilon in epsilons:
                results = [
                    self.verify_sample(
                        cohort_x[index : index + 1],
                        epsilon=epsilon,
                        target=labels[index],
                        clip_min=clip_min,
                        clip_max=clip_max,
                    )
                    for index in indices
                ]
                cohort_points.append(
                    _cohort_point(epsilon, results, [labels[index] for index in indices])
                )
        decision_target = sample_points[0].target
        return FormalRobustnessCertificate(
            model_name=self.model_name,
            property="local classification invariance",
            norm="linf",
            input_domain=(clip_min, clip_max),
            target=decision_target,
            sample_points=sample_points,
            certified_radius_lower_bound=radius_lower,
            first_inconclusive_epsilon=first_inconclusive,
            cohort_points=cohort_points,
            selected_indices=indices,
            input_spec=input_spec,
            limitations=[
                "IBP is sound for the implemented operator transformers but may be conservative; inconclusive does not mean vulnerable.",
                "The certificate applies only to the exact checkpoint, input domain, norm and class property recorded here.",
                "The native verifier is incomplete and does not perform branch-and-bound or optimised linear relaxations.",
                "Mathematical pixel-space invariance does not establish semantic or functionality-preserving robustness.",
                "The proof concerns the real-valued network represented by the stored parameters; backend-specific floating-point inference semantics are outside this certificate.",
                "Float64 propagation and outward padding mitigate verifier round-off but are not a bit-level proof of every hardware kernel.",
            ],
            metadata={
                "method": "interval bound propagation over a torch.fx operator graph",
                "model_mode": "evaluation",
                "strict_margin_rule": "lower(z_target) - max_j upper(z_j) > 0",
                "operator_ledger": _operator_ledger(sample_points),
                "bound_bottleneck_at_first_inconclusive": _bound_bottleneck(
                    first_inconclusive_point
                ),
                "fx_trace_error": self.trace_error,
                "evidence_tier": "formal_local_property_certificate",
                "input_semantics_declared": bool(input_spec is not None and input_spec.declared),
                "certified_radius_search": {
                    "lower_bound": radius_lower,
                    "first_inconclusive_upper_bound": radius_upper,
                    "iterations": int(radius_search_steps),
                    "trace": search_trace,
                },
            },
        )

    def _search_certified_radius(
        self,
        x: Tensor,
        *,
        target: int | None,
        upper: float,
        clip_min: float | None,
        clip_max: float | None,
        steps: int,
    ) -> tuple[float, float | None, list[dict[str, Any]]]:
        """Bisect the monotone IBP verdict to return a certified lower bound."""

        lower = 0.0
        upper_bound = float(upper)
        trace: list[dict[str, Any]] = []
        endpoint = self.verify_sample(
            x,
            epsilon=upper_bound,
            target=target,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        trace.append(
            {
                "epsilon": upper_bound,
                "certified": endpoint.certified,
                "margin_lower_bound": endpoint.certified_margin_lower_bound,
            }
        )
        if endpoint.certified:
            return upper_bound, None, trace
        for _ in range(steps):
            midpoint = (lower + upper_bound) / 2.0
            result = self.verify_sample(
                x,
                epsilon=midpoint,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            trace.append(
                {
                    "epsilon": midpoint,
                    "certified": result.certified,
                    "margin_lower_bound": result.certified_margin_lower_bound,
                }
            )
            if result.certified:
                lower = midpoint
            else:
                upper_bound = midpoint
        return lower, upper_bound, trace

    def _propagate(self, input_bound: Bound) -> tuple[Bound, list[LayerIntervalRecord]]:
        output, records, _captured = self._propagate_detailed(input_bound)
        return output, records

    def bound_graph(
        self,
        x: Tensor,
        *,
        epsilon: float,
        clip_min: float | None = None,
        clip_max: float | None = None,
        capture_nodes: Sequence[str] = (),
        bound_transforms: Mapping[str, BoundTransform] | None = None,
    ) -> tuple[Bound, list[LayerIntervalRecord], dict[str, Bound]]:
        """Propagate one input box and optionally capture or transform graph nodes.

        This advanced interface is used by mechanism-property verifiers. Transform
        keys may be FX node names (for example ``head_2``) or module targets (for
        example ``head.2``). Each transform must itself be a sound interval map.
        """

        if self.graph is None:
            raise UnsupportedVerificationOperator(f"FX graph is unavailable: {self.trace_error}")
        epsilon = float(epsilon)
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be non-negative and finite")
        if clip_min is not None and not math.isfinite(clip_min):
            raise ValueError("clip_min must be finite when supplied")
        if clip_max is not None and not math.isfinite(clip_max):
            raise ValueError("clip_max must be finite when supplied")
        x_in = x.detach().to(device=self.device, dtype=self.dtype)
        if x_in.ndim == 0:
            x_in = x_in.reshape(1, 1)
        elif x_in.shape[0] != 1:
            x_in = x_in.unsqueeze(0)
        if x_in.shape[0] != 1:
            raise ValueError("bound_graph requires exactly one sample")
        if clip_min is not None and clip_max is not None and clip_min >= clip_max:
            raise ValueError("clip_min must be smaller than clip_max")
        if clip_min is not None and bool((x_in < clip_min).any()):
            raise ValueError("clean input lies outside the declared input domain")
        if clip_max is not None and bool((x_in > clip_max).any()):
            raise ValueError("clean input lies outside the declared input domain")
        lower = x_in.double() - epsilon
        upper = x_in.double() + epsilon
        if clip_min is not None:
            lower = torch.clamp_min(lower, clip_min)
            upper = torch.clamp_min(upper, clip_min)
        if clip_max is not None:
            lower = torch.clamp_max(lower, clip_max)
            upper = torch.clamp_max(upper, clip_max)
        return self._propagate_detailed(
            (lower, upper),
            capture_nodes=capture_nodes,
            bound_transforms=bound_transforms,
        )

    def bound_graph_difference(
        self,
        x: Tensor,
        *,
        epsilon: float,
        clip_min: float | None = None,
        clip_max: float | None = None,
        paired_bound_transforms: Mapping[str, PairedBoundTransform] | None = None,
        output_linear_coefficients: Tensor | None = None,
    ) -> tuple[Bound, Bound, Bound]:
        """Propagate original/intervened boxes with a shared-input difference bound.

        The third returned bound encloses ``original - intervened`` for executions
        driven by the *same* input in the declared box.  Affine and shape-only
        operators propagate that relational difference directly, so input terms and
        affine biases shared by both executions cancel.  Supported nonlinearities use
        sound secant-slope enclosures; unrecognised relational cases fall back to the
        independent difference of the two ordinary interval bounds.

        ``output_linear_coefficients`` optionally requests a scalar linear functional
        of the output difference.  A terminal ``nn.Linear`` is fused with that
        functional, avoiding an otherwise unnecessary loss of dependence between
        output coordinates.
        """

        if self.graph is None:
            raise UnsupportedVerificationOperator(f"FX graph is unavailable: {self.trace_error}")
        epsilon = float(epsilon)
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be non-negative and finite")
        if clip_min is not None and not math.isfinite(clip_min):
            raise ValueError("clip_min must be finite when supplied")
        if clip_max is not None and not math.isfinite(clip_max):
            raise ValueError("clip_max must be finite when supplied")
        if clip_min is not None and clip_max is not None and clip_min >= clip_max:
            raise ValueError("clip_min must be smaller than clip_max")
        x_in = x.detach().to(device=self.device, dtype=self.dtype)
        if x_in.ndim == 0:
            x_in = x_in.reshape(1, 1)
        elif x_in.shape[0] != 1:
            x_in = x_in.unsqueeze(0)
        if x_in.shape[0] != 1:
            raise ValueError("bound_graph_difference requires exactly one sample")
        if clip_min is not None and bool((x_in < clip_min).any()):
            raise ValueError("clean input lies outside the declared input domain")
        if clip_max is not None and bool((x_in > clip_max).any()):
            raise ValueError("clean input lies outside the declared input domain")
        lower = x_in.double() - epsilon
        upper = x_in.double() + epsilon
        if clip_min is not None:
            lower = torch.clamp_min(lower, clip_min)
            upper = torch.clamp_min(upper, clip_min)
        if clip_max is not None:
            lower = torch.clamp_max(lower, clip_max)
            upper = torch.clamp_max(upper, clip_max)
        return self._propagate_paired_difference(
            (lower, upper),
            paired_bound_transforms=paired_bound_transforms,
            output_linear_coefficients=output_linear_coefficients,
        )

    def _propagate_paired_difference(
        self,
        input_bound: Bound,
        *,
        paired_bound_transforms: Mapping[str, PairedBoundTransform] | None,
        output_linear_coefficients: Tensor | None,
    ) -> tuple[Bound, Bound, Bound]:
        """Execute the FX graph in ordinary and relational interval domains."""

        if self.graph is None:
            raise UnsupportedVerificationOperator("FX graph is unavailable")
        environment: dict[Node, Any] = {}
        transforms = dict(paired_bound_transforms or {})

        def load(argument: Any) -> Any:
            return torch.fx.node.map_arg(argument, lambda node: environment[node])

        for node in self.graph.graph.nodes:
            if node.op == "placeholder":
                environment[node] = _PairedInterval(
                    original=input_bound,
                    intervened=input_bound,
                    difference=_zero_bound_like(input_bound),
                )
                continue
            if node.op == "get_attr":
                environment[node] = _fetch_attr(self.graph, str(node.target))
                continue
            if node.op == "output":
                value = load(node.args[0])
                if not isinstance(value, _PairedInterval):
                    raise UnsupportedVerificationOperator(
                        "paired graph output is not a single bounded tensor"
                    )
                difference = value.difference
                if output_linear_coefficients is not None:
                    difference = _output_linear_difference(
                        graph=self.graph,
                        output_argument=node.args[0],
                        output_state=value,
                        environment=environment,
                        coefficients=output_linear_coefficients,
                        transformed_keys=set(transforms),
                    )
                return value.original, value.intervened, difference

            loaded_args = load(node.args)
            loaded_kwargs = load(node.kwargs)
            original_args = _paired_branch(loaded_args, "original")
            intervened_args = _paired_branch(loaded_args, "intervened")
            original_kwargs = _paired_branch(loaded_kwargs, "original")
            intervened_kwargs = _paired_branch(loaded_kwargs, "intervened")
            shared_inputs = _paired_arguments_are_identical((loaded_args, loaded_kwargs))

            if node.op == "call_module":
                module = self.graph.get_submodule(str(node.target))
                original = _module_interval(module, original_args[0], original_kwargs)
                input_state = loaded_args[0]
                if not isinstance(input_state, _PairedInterval):
                    raise UnsupportedVerificationOperator(
                        f"module {node.target!s} did not receive a paired bounded tensor"
                    )
                if shared_inputs:
                    intervened = original
                    difference = _zero_bound_like(original)
                else:
                    intervened = _module_interval(module, intervened_args[0], intervened_kwargs)
                    difference = _module_difference_interval(
                        module,
                        input_state.difference,
                        original=original,
                        intervened=intervened,
                    )
                key = node.name if node.name in transforms else str(node.target)
            elif node.op == "call_function":
                original = _function_interval(node.target, original_args, original_kwargs)
                if not _is_bound(original):
                    environment[node] = original
                    continue
                if shared_inputs:
                    intervened = original
                    difference = _zero_bound_like(original)
                else:
                    intervened = _function_interval(node.target, intervened_args, intervened_kwargs)
                    if not _is_bound(intervened):
                        raise UnsupportedVerificationOperator(
                            "paired function executions disagree on bounded output type"
                        )
                    difference = _function_difference_interval(
                        node.target,
                        loaded_args,
                        loaded_kwargs,
                        original=original,
                        intervened=intervened,
                    )
                key = node.name
            elif node.op == "call_method":
                original = _method_interval(str(node.target), original_args, original_kwargs)
                if not _is_bound(original):
                    environment[node] = original
                    continue
                if shared_inputs:
                    intervened = original
                    difference = _zero_bound_like(original)
                else:
                    intervened = _method_interval(
                        str(node.target), intervened_args, intervened_kwargs
                    )
                    if not _is_bound(intervened):
                        raise UnsupportedVerificationOperator(
                            "paired method executions disagree on bounded output type"
                        )
                    difference = _method_difference_interval(
                        str(node.target),
                        loaded_args,
                        loaded_kwargs,
                        original=original,
                        intervened=intervened,
                    )
                key = node.name
            else:
                raise UnsupportedVerificationOperator(f"unsupported FX node kind: {node.op}")

            state = _PairedInterval(
                original=original,
                intervened=intervened,
                difference=difference,
            )
            if key in transforms:
                transformed, transformed_difference = transforms[key](
                    state.original,
                    state.intervened,
                    state.difference,
                )
                state = _PairedInterval(
                    original=state.original,
                    intervened=transformed,
                    difference=transformed_difference,
                )
            environment[node] = state
        raise UnsupportedVerificationOperator("FX graph has no output node")

    def _propagate_detailed(
        self,
        input_bound: Bound,
        *,
        capture_nodes: Sequence[str] = (),
        bound_transforms: Mapping[str, BoundTransform] | None = None,
    ) -> tuple[Bound, list[LayerIntervalRecord], dict[str, Bound]]:
        if self.graph is None:
            raise UnsupportedVerificationOperator("FX graph is unavailable")
        environment: dict[Node, Any] = {}
        records: list[LayerIntervalRecord] = []
        captured: dict[str, Bound] = {}
        capture_set = {str(value) for value in capture_nodes}
        transforms = dict(bound_transforms or {})

        def load(argument: Any) -> Any:
            return torch.fx.node.map_arg(argument, lambda node: environment[node])

        for node in self.graph.graph.nodes:
            if node.op == "placeholder":
                environment[node] = input_bound
            elif node.op == "get_attr":
                environment[node] = _fetch_attr(self.graph, str(node.target))
            elif node.op == "call_module":
                module = self.graph.get_submodule(str(node.target))
                args = load(node.args)
                kwargs = load(node.kwargs)
                bound = _module_interval(module, args[0], kwargs)
                key = node.name if node.name in transforms else str(node.target)
                if key in transforms:
                    bound = transforms[key](bound)
                environment[node] = bound
                records.append(_layer_record(str(node.target), module, bound, input_bound=args[0]))
            elif node.op == "call_function":
                args = load(node.args)
                kwargs = load(node.kwargs)
                bound = _function_interval(node.target, args, kwargs)
                if node.name in transforms:
                    if not _is_bound(bound):
                        raise UnsupportedVerificationOperator(
                            f"node {node.name!r} is not a bounded tensor"
                        )
                    bound = transforms[node.name](bound)
                environment[node] = bound
            elif node.op == "call_method":
                args = load(node.args)
                kwargs = load(node.kwargs)
                bound = _method_interval(str(node.target), args, kwargs)
                if node.name in transforms:
                    if not _is_bound(bound):
                        raise UnsupportedVerificationOperator(
                            f"node {node.name!r} is not a bounded tensor"
                        )
                    bound = transforms[node.name](bound)
                environment[node] = bound
            elif node.op == "output":
                output = load(node.args[0])
                if not _is_bound(output):
                    raise UnsupportedVerificationOperator(
                        "graph output is not a single bounded tensor"
                    )
                return output, records, captured
            else:
                raise UnsupportedVerificationOperator(f"unsupported FX node kind: {node.op}")
            if node.op not in {"placeholder", "get_attr", "output"}:
                target_name = str(node.target) if node.op == "call_module" else None
                if node.name in capture_set and _is_bound(environment[node]):
                    captured[node.name] = environment[node]
                if (
                    target_name is not None
                    and target_name in capture_set
                    and _is_bound(environment[node])
                ):
                    captured[target_name] = environment[node]
        raise UnsupportedVerificationOperator("FX graph has no output node")


def _module_interval(module: nn.Module, bound: Bound, kwargs: dict[str, Any]) -> Bound:
    del kwargs
    lower, upper = bound
    if isinstance(module, nn.Linear):
        return _affine_interval(
            lower,
            upper,
            module.weight.detach().double(),
            None if module.bias is None else module.bias.detach().double(),
            F.linear,
            bias_axis=-1,
        )
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        function = {nn.Conv1d: F.conv1d, nn.Conv2d: F.conv2d, nn.Conv3d: F.conv3d}[type(module)]
        convolution = lambda value, weight, bias: function(  # noqa: E731
            value,
            weight,
            bias,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
        return _affine_interval(
            lower,
            upper,
            module.weight.detach().double(),
            None if module.bias is None else module.bias.detach().double(),
            convolution,
            bias_axis=1,
        )
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if module.training or module.running_mean is None or module.running_var is None:
            raise UnsupportedVerificationOperator("BatchNorm must be in evaluation mode")
        running_mean = module.running_mean.detach().double()
        running_var = module.running_var.detach().double()
        weight = (
            module.weight.detach().double()
            if module.weight is not None
            else torch.ones_like(running_mean)
        )
        bias = (
            module.bias.detach().double()
            if module.bias is not None
            else torch.zeros_like(running_mean)
        )
        scale = weight / torch.sqrt(running_var + module.eps)
        shift = bias - running_mean * scale
        shape = [1, -1, *([1] * (lower.ndim - 2))]
        scale = scale.reshape(shape)
        shift = shift.reshape(shape)
        out_lower = torch.where(scale >= 0, scale * lower, scale * upper) + shift
        out_upper = torch.where(scale >= 0, scale * upper, scale * lower) + shift
        return _outward(
            out_lower, out_upper, magnitude=torch.maximum(out_lower.abs(), out_upper.abs()), terms=6
        )
    if isinstance(module, nn.ReLU):
        return torch.relu(lower), torch.relu(upper)
    if isinstance(module, nn.LeakyReLU):
        if module.negative_slope < 0:
            raise UnsupportedVerificationOperator("negative LeakyReLU slope is unsupported")
        return F.leaky_relu(lower, module.negative_slope), F.leaky_relu(
            upper, module.negative_slope
        )
    if isinstance(module, nn.Sigmoid):
        return torch.sigmoid(lower), torch.sigmoid(upper)
    if isinstance(module, nn.Tanh):
        return torch.tanh(lower), torch.tanh(upper)
    if isinstance(module, nn.Flatten):
        return (
            torch.flatten(lower, module.start_dim, module.end_dim),
            torch.flatten(upper, module.start_dim, module.end_dim),
        )
    if isinstance(
        module,
        (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.Identity),
    ):
        if not isinstance(module, nn.Identity) and module.training:
            raise UnsupportedVerificationOperator("Dropout must be in evaluation mode")
        return lower, upper
    if isinstance(module, nn.MaxPool1d):
        return module(lower), module(upper)
    if isinstance(module, nn.MaxPool2d):
        return module(lower), module(upper)
    if isinstance(module, nn.MaxPool3d):
        return module(lower), module(upper)
    if isinstance(
        module,
        (
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
        ),
    ):
        return module(lower), module(upper)
    raise UnsupportedVerificationOperator(
        f"unsupported module {type(module).__module__}.{type(module).__qualname__}"
    )


def _affine_interval(
    lower: Tensor,
    upper: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    function: Any,
    *,
    bias_axis: int,
) -> Bound:
    positive = torch.clamp(weight, min=0)
    negative = torch.clamp(weight, max=0)
    out_lower = function(lower, positive, bias) + function(upper, negative, None)
    out_upper = function(upper, positive, bias) + function(lower, negative, None)
    magnitude_input = torch.maximum(lower.abs(), upper.abs())
    magnitude = function(magnitude_input, weight.abs(), None)
    if bias is not None:
        axis = int(bias_axis) % magnitude.ndim
        if int(magnitude.shape[axis]) != int(bias.numel()):
            raise UnsupportedVerificationOperator(
                "affine bias cannot be broadcast along the declared output axis"
            )
        bias_shape = [1] * magnitude.ndim
        bias_shape[axis] = int(bias.numel())
        magnitude = magnitude + bias.abs().reshape(bias_shape)
    terms = int(weight[0].numel()) + 3
    return _outward(out_lower, out_upper, magnitude=magnitude, terms=terms)


def _outward(lower: Tensor, upper: Tensor, *, magnitude: Tensor, terms: int) -> Bound:
    unit = torch.finfo(lower.dtype).eps / 2.0
    scaled = min(max(int(terms), 1) * 8.0 * unit, 0.25)
    padding = magnitude.abs() * scaled + torch.finfo(lower.dtype).tiny
    negative_inf = torch.full_like(lower, float("-inf"))
    positive_inf = torch.full_like(upper, float("inf"))
    return (
        torch.nextafter(lower - padding, negative_inf),
        torch.nextafter(upper + padding, positive_inf),
    )


def _zero_bound_like(bound: Bound) -> Bound:
    zeros = torch.zeros_like(bound[0])
    return zeros, zeros.clone()


def _paired_branch(value: Any, branch: Literal["original", "intervened"]) -> Any:
    """Recursively select one ordinary interval execution from paired arguments."""

    if isinstance(value, _PairedInterval):
        return getattr(value, branch)
    if isinstance(value, tuple):
        return tuple(_paired_branch(item, branch) for item in value)
    if isinstance(value, list):
        return [_paired_branch(item, branch) for item in value]
    if isinstance(value, dict):
        return {key: _paired_branch(item, branch) for key, item in value.items()}
    return value


def _paired_arguments_are_identical(value: Any) -> bool:
    """Return whether paired arguments prove equality of both executions."""

    if isinstance(value, _PairedInterval):
        return bool(
            torch.equal(value.original[0], value.intervened[0])
            and torch.equal(value.original[1], value.intervened[1])
            and torch.count_nonzero(value.difference[0]).item() == 0
            and torch.count_nonzero(value.difference[1]).item() == 0
        )
    if isinstance(value, (tuple, list)):
        return all(_paired_arguments_are_identical(item) for item in value)
    if isinstance(value, dict):
        return all(_paired_arguments_are_identical(item) for item in value.values())
    return True


def _independent_difference(original: Bound, intervened: Bound) -> Bound:
    """Sound dependency-agnostic fallback for ``original - intervened``."""

    return _bound_subtract(original, intervened)


def _intersect_sound_bounds(first: Bound, second: Bound) -> Bound:
    """Intersect two sound enclosures, retaining outward floating-point padding."""

    lower = torch.maximum(first[0], second[0])
    upper = torch.minimum(first[1], second[1])
    invalid = lower > upper
    if bool(invalid.any()):
        # Numerical disagreement between two theoretically overlapping enclosures must
        # never create an empty, hence unsound, interval.  Retain the first enclosure.
        lower = torch.where(invalid, first[0], lower)
        upper = torch.where(invalid, first[1], upper)
    magnitude = torch.maximum(lower.abs(), upper.abs())
    return _outward(lower, upper, magnitude=magnitude, terms=2)


def _monotone_secant_difference(
    difference: Bound,
    *,
    original: Bound,
    intervened: Bound,
    minimum_slope: float,
    maximum_slope: float,
) -> Bound:
    """Bound ``f(a)-f(b)`` using a global non-negative secant-slope range.

    For a scalar monotone function whose every secant slope lies in ``[m, M]``,
    ``f(a)-f(b) = q(a-b)`` for some ``q`` in that interval.  Multiplying the
    relational input-difference interval by ``[m, M]`` is therefore sound.  Its
    intersection with ordinary independent output subtraction cannot lose any
    feasible paired execution.
    """

    slope_lower = torch.full_like(difference[0], float(minimum_slope))
    slope_upper = torch.full_like(difference[1], float(maximum_slope))
    secant = _bound_multiply(difference, (slope_lower, slope_upper))
    return _intersect_sound_bounds(
        _independent_difference(original, intervened),
        secant,
    )


def _module_difference_interval(
    module: nn.Module,
    difference: Bound,
    *,
    original: Bound,
    intervened: Bound,
) -> Bound:
    """Relational transformer for a supported module."""

    lower, upper = difference
    if isinstance(module, nn.Linear):
        return _affine_interval(
            lower,
            upper,
            module.weight.detach().double(),
            None,
            F.linear,
            bias_axis=-1,
        )
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        function = {nn.Conv1d: F.conv1d, nn.Conv2d: F.conv2d, nn.Conv3d: F.conv3d}[type(module)]
        convolution = lambda value, weight, bias: function(  # noqa: E731
            value,
            weight,
            bias,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
        return _affine_interval(
            lower,
            upper,
            module.weight.detach().double(),
            None,
            convolution,
            bias_axis=1,
        )
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if module.training or module.running_mean is None or module.running_var is None:
            raise UnsupportedVerificationOperator("BatchNorm must be in evaluation mode")
        weight = (
            module.weight.detach().double()
            if module.weight is not None
            else torch.ones_like(module.running_mean.detach().double())
        )
        scale = weight / torch.sqrt(module.running_var.detach().double() + module.eps)
        shape = [1, -1, *([1] * (lower.ndim - 2))]
        scale = scale.reshape(shape)
        first, second = scale * lower, scale * upper
        result_lower, result_upper = torch.minimum(first, second), torch.maximum(first, second)
        return _outward(
            result_lower,
            result_upper,
            magnitude=torch.maximum(result_lower.abs(), result_upper.abs()),
            terms=4,
        )
    if isinstance(module, nn.ReLU):
        return _monotone_secant_difference(
            difference,
            original=original,
            intervened=intervened,
            minimum_slope=0.0,
            maximum_slope=1.0,
        )
    if isinstance(module, nn.LeakyReLU):
        return _monotone_secant_difference(
            difference,
            original=original,
            intervened=intervened,
            minimum_slope=min(float(module.negative_slope), 1.0),
            maximum_slope=max(float(module.negative_slope), 1.0),
        )
    if isinstance(module, nn.Sigmoid):
        return _monotone_secant_difference(
            difference,
            original=original,
            intervened=intervened,
            minimum_slope=0.0,
            maximum_slope=0.25,
        )
    if isinstance(module, nn.Tanh):
        return _monotone_secant_difference(
            difference,
            original=original,
            intervened=intervened,
            minimum_slope=0.0,
            maximum_slope=1.0,
        )
    if isinstance(module, nn.Flatten):
        return (
            torch.flatten(lower, module.start_dim, module.end_dim),
            torch.flatten(upper, module.start_dim, module.end_dim),
        )
    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.Identity)):
        return difference
    if isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        return module(lower), module(upper)
    if isinstance(
        module,
        (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d),
    ):
        return module(lower), module(upper)
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        relational = (-module(-lower), module(upper))
        return _intersect_sound_bounds(
            _independent_difference(original, intervened),
            relational,
        )
    return _independent_difference(original, intervened)


def _difference_operand(value: Any) -> Bound | None:
    return value.difference if isinstance(value, _PairedInterval) else None


def _negate_bound(bound: Bound) -> Bound:
    return -bound[1], -bound[0]


def _add_differences(left: Bound | None, right: Bound | None) -> Bound:
    if left is None and right is None:
        raise UnsupportedVerificationOperator("relational addition has no bounded operand")
    if left is None:
        assert right is not None
        return right
    if right is None:
        return left
    return _bound_add(left, right)


def _subtract_differences(left: Bound | None, right: Bound | None) -> Bound:
    if left is None and right is None:
        raise UnsupportedVerificationOperator("relational subtraction has no bounded operand")
    if left is None:
        assert right is not None
        return _negate_bound(right)
    if right is None:
        return left
    return _bound_subtract(left, right)


def _constant_bound(value: Any, *, reference: Bound) -> Bound:
    tensor = torch.as_tensor(value, device=reference[0].device, dtype=reference[0].dtype)
    return tensor, tensor


def _multiplication_difference(
    left: Any,
    right: Any,
    *,
    original: Bound,
    intervened: Bound,
) -> Bound:
    """Use ``ab-a'b'=(a-a')b+a'(b-b')`` with interval enclosures."""

    left_difference = _difference_operand(left)
    right_difference = _difference_operand(right)
    if left_difference is None and right_difference is None:
        return _independent_difference(original, intervened)
    if isinstance(right, _PairedInterval):
        right_original = right.original
    else:
        reference = left_difference if left_difference is not None else right_difference
        assert reference is not None
        right_original = _constant_bound(right, reference=reference)
    if isinstance(left, _PairedInterval):
        left_intervened = left.intervened
    else:
        reference = right_difference if right_difference is not None else left_difference
        assert reference is not None
        left_intervened = _constant_bound(left, reference=reference)
    terms: list[Bound] = []
    if left_difference is not None:
        terms.append(_bound_multiply(left_difference, right_original))
    if right_difference is not None:
        terms.append(_bound_multiply(left_intervened, right_difference))
    relational = terms[0]
    for term in terms[1:]:
        relational = _bound_add(relational, term)
    return _intersect_sound_bounds(
        _independent_difference(original, intervened),
        relational,
    )


def _function_difference_interval(
    target: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    original: Bound,
    intervened: Bound,
) -> Bound:
    if target in {operator.add, torch.add}:
        return _add_differences(_difference_operand(args[0]), _difference_operand(args[1]))
    if target in {operator.sub, torch.sub}:
        return _subtract_differences(
            _difference_operand(args[0]),
            _difference_operand(args[1]),
        )
    if target in {operator.mul, torch.mul}:
        return _multiplication_difference(
            args[0],
            args[1],
            original=original,
            intervened=intervened,
        )
    if target is torch.flatten:
        state = args[0]
        if isinstance(state, _PairedInterval):
            start_dim = kwargs.get("start_dim", args[1] if len(args) > 1 else 0)
            end_dim = kwargs.get("end_dim", args[2] if len(args) > 2 else -1)
            return (
                torch.flatten(state.difference[0], start_dim, end_dim),
                torch.flatten(state.difference[1], start_dim, end_dim),
            )
    if target in {torch.relu, F.relu}:
        state = args[0]
        if isinstance(state, _PairedInterval):
            return _monotone_secant_difference(
                state.difference,
                original=original,
                intervened=intervened,
                minimum_slope=0.0,
                maximum_slope=1.0,
            )
    if target is operator.getitem:
        state, index = args[0], args[1]
        if isinstance(state, _PairedInterval):
            return state.difference[0][index], state.difference[1][index]
    if target is F.pad:
        state = args[0]
        if isinstance(state, _PairedInterval):
            pad = kwargs.get("pad", args[1] if len(args) > 1 else None)
            mode = kwargs.get("mode", args[2] if len(args) > 2 else "constant")
            if isinstance(pad, (tuple, list)):
                if mode == "constant":
                    return (
                        F.pad(state.difference[0], tuple(pad), mode=mode, value=0.0),
                        F.pad(state.difference[1], tuple(pad), mode=mode, value=0.0),
                    )
                return (
                    F.pad(state.difference[0], tuple(pad), mode=mode),
                    F.pad(state.difference[1], tuple(pad), mode=mode),
                )
    if target in {torch.reshape, torch.transpose, torch.unsqueeze, torch.squeeze}:
        state = args[0]
        if isinstance(state, _PairedInterval):
            return (
                target(state.difference[0], *args[1:], **kwargs),
                target(state.difference[1], *args[1:], **kwargs),
            )
    if target in {torch.cat, torch.concat, torch.concatenate}:
        values = args[0]
        if isinstance(values, (tuple, list)):
            reference = next(
                (value.difference for value in values if isinstance(value, _PairedInterval)),
                None,
            )
            if reference is not None:
                lower_values: list[Tensor] = []
                upper_values: list[Tensor] = []
                for value in values:
                    if isinstance(value, _PairedInterval):
                        lower_values.append(value.difference[0])
                        upper_values.append(value.difference[1])
                    else:
                        constant = torch.as_tensor(
                            value,
                            device=reference[0].device,
                            dtype=reference[0].dtype,
                        )
                        lower_values.append(torch.zeros_like(constant))
                        upper_values.append(torch.zeros_like(constant))
                trailing_args = args[1:]
                return (
                    target(lower_values, *trailing_args, **kwargs),
                    target(upper_values, *trailing_args, **kwargs),
                )
    return _independent_difference(original, intervened)


def _method_difference_interval(
    target: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    original: Bound,
    intervened: Bound,
) -> Bound:
    state = args[0]
    if not isinstance(state, _PairedInterval):
        return _independent_difference(original, intervened)
    lower, upper = state.difference
    if target in {
        "flatten",
        "reshape",
        "view",
        "expand",
        "permute",
        "squeeze",
        "transpose",
        "unsqueeze",
    }:
        return (
            getattr(lower, target)(*args[1:], **kwargs),
            getattr(upper, target)(*args[1:], **kwargs),
        )
    if target in {"contiguous", "clone", "detach"}:
        return (
            getattr(lower, target)(*args[1:], **kwargs),
            getattr(upper, target)(*args[1:], **kwargs),
        )
    if target == "mean":
        return lower.mean(*args[1:], **kwargs), upper.mean(*args[1:], **kwargs)
    return _independent_difference(original, intervened)


def _linear_functional_interval(bound: Bound, coefficients: Tensor) -> Bound:
    lower, upper = bound
    weight = coefficients.detach().to(device=lower.device, dtype=lower.dtype).reshape(1, -1)
    if int(lower.shape[-1]) != int(weight.shape[-1]):
        raise UnsupportedVerificationOperator(
            "output linear coefficients do not match the bounded output width"
        )
    return _affine_interval(
        lower,
        upper,
        weight,
        None,
        F.linear,
        bias_axis=-1,
    )


def _output_linear_difference(
    *,
    graph: GraphModule,
    output_argument: Any,
    output_state: _PairedInterval,
    environment: Mapping[Node, Any],
    coefficients: Tensor,
    transformed_keys: set[str],
) -> Bound:
    """Apply an output functional, fusing a terminal Linear when sound."""

    if isinstance(output_argument, Node) and output_argument.op == "call_module":
        module = graph.get_submodule(str(output_argument.target))
        key_was_transformed = (
            output_argument.name in transformed_keys
            or str(output_argument.target) in transformed_keys
        )
        if isinstance(module, nn.Linear) and not key_was_transformed:
            input_argument = output_argument.args[0]
            if isinstance(input_argument, Node):
                input_state = environment.get(input_argument)
                if isinstance(input_state, _PairedInterval):
                    coefficient = (
                        coefficients.detach()
                        .to(
                            device=module.weight.device,
                            dtype=torch.float64,
                        )
                        .reshape(-1)
                    )
                    if int(coefficient.numel()) == int(module.weight.shape[0]):
                        fused_weight = coefficient @ module.weight.detach().double()
                        return _affine_interval(
                            input_state.difference[0],
                            input_state.difference[1],
                            fused_weight.reshape(1, -1),
                            None,
                            F.linear,
                            bias_axis=-1,
                        )
    return _linear_functional_interval(output_state.difference, coefficients)


def _function_interval(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if target in {operator.add, torch.add}:
        if not _is_bound(args[0]) and not _is_bound(args[1]):
            return target(*args, **kwargs)
        return _bound_add(args[0], args[1])
    if target in {operator.sub, torch.sub}:
        if not _is_bound(args[0]) and not _is_bound(args[1]):
            return target(*args, **kwargs)
        return _bound_subtract(args[0], args[1])
    if target in {operator.mul, torch.mul}:
        if not _is_bound(args[0]) and not _is_bound(args[1]):
            return target(*args, **kwargs)
        return _bound_multiply(args[0], args[1])
    if target is torch.flatten:
        if not _is_bound(args[0]):
            raise UnsupportedVerificationOperator("torch.flatten input is not a bounded tensor")
        lower, upper = args[0]
        start_dim = kwargs.get("start_dim", args[1] if len(args) > 1 else 0)
        end_dim = kwargs.get("end_dim", args[2] if len(args) > 2 else -1)
        return torch.flatten(lower, start_dim, end_dim), torch.flatten(upper, start_dim, end_dim)
    if target in {torch.relu, F.relu}:
        if not _is_bound(args[0]):
            raise UnsupportedVerificationOperator("ReLU input is not a bounded tensor")
        lower, upper = args[0]
        return torch.relu(lower), torch.relu(upper)
    if target is operator.getitem:
        value, index = args[0], args[1]
        if _is_bound(value):
            return value[0][index], value[1][index]
        if isinstance(value, (Tensor, tuple, list, torch.Size)):
            return value[index]
        raise UnsupportedVerificationOperator(
            f"getitem source has unsupported type {type(value).__qualname__}"
        )
    if target is getattr:
        return _tensor_metadata(args[0], str(args[1]))
    if target is F.pad:
        return _pad_interval(args, kwargs)
    if target in {torch.reshape, torch.transpose, torch.unsqueeze, torch.squeeze}:
        if not _is_bound(args[0]):
            raise UnsupportedVerificationOperator(
                f"{target.__name__} input is not a bounded tensor"
            )
        lower, upper = args[0]
        return target(lower, *args[1:], **kwargs), target(upper, *args[1:], **kwargs)
    if target in {torch.cat, torch.concat, torch.concatenate}:
        return _concatenate_interval(target, args, kwargs)
    raise UnsupportedVerificationOperator(f"unsupported function {target}")


def _method_interval(target: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = args[0]
    if target in {"size", "dim", "numel"}:
        return _tensor_metadata_method(value, target, args[1:], kwargs)
    if not _is_bound(value):
        if isinstance(value, Tensor) and target in {
            "clone",
            "contiguous",
            "detach",
            "expand",
            "flatten",
            "permute",
            "reshape",
            "squeeze",
            "transpose",
            "unsqueeze",
            "view",
        }:
            return getattr(value, target)(*args[1:], **kwargs)
        raise UnsupportedVerificationOperator(
            f"tensor method {target} received an unbounded {type(value).__qualname__}"
        )
    lower, upper = value
    if target in {"flatten", "reshape", "view"}:
        method_lower = getattr(lower, target)
        method_upper = getattr(upper, target)
        return method_lower(*args[1:], **kwargs), method_upper(*args[1:], **kwargs)
    if target in {
        "expand",
        "permute",
        "squeeze",
        "transpose",
        "unsqueeze",
    }:
        return (
            getattr(lower, target)(*args[1:], **kwargs),
            getattr(upper, target)(*args[1:], **kwargs),
        )
    if target in {"contiguous", "clone", "detach"}:
        return (
            getattr(lower, target)(*args[1:], **kwargs),
            getattr(upper, target)(*args[1:], **kwargs),
        )
    if target == "mean":
        return lower.mean(*args[1:], **kwargs), upper.mean(*args[1:], **kwargs)
    raise UnsupportedVerificationOperator(f"unsupported tensor method {target}")


def _tensor_metadata(value: Any, attribute: str) -> Any:
    """Return shape-only tensor metadata that is identical at both box endpoints."""

    if attribute not in {"shape", "ndim"}:
        raise UnsupportedVerificationOperator(
            f"unsupported tensor metadata attribute {attribute!r}"
        )
    if _is_bound(value):
        lower_value = getattr(value[0], attribute)
        upper_value = getattr(value[1], attribute)
        if lower_value != upper_value:
            raise UnsupportedVerificationOperator(
                f"interval endpoints disagree on tensor metadata {attribute!r}"
            )
        return lower_value
    if isinstance(value, Tensor):
        return getattr(value, attribute)
    raise UnsupportedVerificationOperator(
        f"metadata source has unsupported type {type(value).__qualname__}"
    )


def _tensor_metadata_method(
    value: Any,
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Evaluate a shape-only method after proving endpoint metadata agree."""

    if _is_bound(value):
        lower_value = getattr(value[0], method)(*args, **kwargs)
        upper_value = getattr(value[1], method)(*args, **kwargs)
        if lower_value != upper_value:
            raise UnsupportedVerificationOperator(
                f"interval endpoints disagree on tensor metadata method {method!r}"
            )
        return lower_value
    if isinstance(value, Tensor):
        return getattr(value, method)(*args, **kwargs)
    raise UnsupportedVerificationOperator(
        f"metadata source has unsupported type {type(value).__qualname__}"
    )


def _pad_interval(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Bound:
    if not args or not _is_bound(args[0]):
        raise UnsupportedVerificationOperator("pad input is not a bounded tensor")
    if len(args) > 4:
        raise UnsupportedVerificationOperator("pad received unsupported positional arguments")
    pad = kwargs.get("pad", args[1] if len(args) > 1 else None)
    mode = kwargs.get("mode", args[2] if len(args) > 2 else "constant")
    value = kwargs.get("value", args[3] if len(args) > 3 else None)
    if not isinstance(pad, (tuple, list)) or not all(isinstance(amount, int) for amount in pad):
        raise UnsupportedVerificationOperator("pad widths must be static integers")
    if mode not in {"constant", "reflect", "replicate", "circular"}:
        raise UnsupportedVerificationOperator(f"unsupported padding mode {mode!r}")
    if value is not None and not isinstance(value, (int, float)):
        raise UnsupportedVerificationOperator("constant padding value must be numeric")
    lower, upper = args[0]
    try:
        return (
            F.pad(lower, tuple(pad), mode=mode, value=value),
            F.pad(upper, tuple(pad), mode=mode, value=value),
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise UnsupportedVerificationOperator(f"invalid static padding: {error}") from error


def _concatenate_interval(
    target: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    values = args[0]
    if not isinstance(values, (tuple, list)):
        raise UnsupportedVerificationOperator("concatenation requires a static tensor sequence")
    if not any(_is_bound(value) for value in values):
        return target(*args, **kwargs)
    reference = next(value[0] for value in values if _is_bound(value))

    def endpoint(value: Any, position: int) -> Tensor:
        if _is_bound(value):
            return value[position]
        if isinstance(value, Tensor):
            return value.to(device=reference.device, dtype=reference.dtype)
        raise UnsupportedVerificationOperator(
            "concatenation operands must be bounded or constant tensors"
        )

    lower_values = [endpoint(value, 0) for value in values]
    upper_values = [endpoint(value, 1) for value in values]
    trailing_args = args[1:]
    return (
        target(lower_values, *trailing_args, **kwargs),
        target(upper_values, *trailing_args, **kwargs),
    )


def _bound_add(left: Any, right: Any) -> Bound:
    if _is_bound(left) and _is_bound(right):
        lower, upper = left[0] + right[0], left[1] + right[1]
        magnitude = torch.maximum(left[0].abs(), left[1].abs()) + torch.maximum(
            right[0].abs(), right[1].abs()
        )
        return _outward(lower, upper, magnitude=magnitude, terms=2)
    if _is_bound(left):
        lower, upper = left[0] + right, left[1] + right
        magnitude = (
            torch.maximum(left[0].abs(), left[1].abs())
            + torch.as_tensor(right, device=lower.device, dtype=lower.dtype).abs()
        )
        return _outward(lower, upper, magnitude=magnitude, terms=2)
    if _is_bound(right):
        lower, upper = left + right[0], left + right[1]
        magnitude = torch.as_tensor(
            left, device=lower.device, dtype=lower.dtype
        ).abs() + torch.maximum(right[0].abs(), right[1].abs())
        return _outward(lower, upper, magnitude=magnitude, terms=2)
    raise UnsupportedVerificationOperator("addition has no bounded operand")


def _bound_subtract(left: Any, right: Any) -> Bound:
    if _is_bound(left) and _is_bound(right):
        lower, upper = left[0] - right[1], left[1] - right[0]
        magnitude = torch.maximum(left[0].abs(), left[1].abs()) + torch.maximum(
            right[0].abs(), right[1].abs()
        )
        return _outward(lower, upper, magnitude=magnitude, terms=2)
    if _is_bound(left):
        lower, upper = left[0] - right, left[1] - right
        magnitude = (
            torch.maximum(left[0].abs(), left[1].abs())
            + torch.as_tensor(right, device=lower.device, dtype=lower.dtype).abs()
        )
        return _outward(lower, upper, magnitude=magnitude, terms=2)
    if _is_bound(right):
        lower, upper = left - right[1], left - right[0]
        magnitude = torch.as_tensor(
            left, device=lower.device, dtype=lower.dtype
        ).abs() + torch.maximum(right[0].abs(), right[1].abs())
        return _outward(lower, upper, magnitude=magnitude, terms=2)
    raise UnsupportedVerificationOperator("subtraction has no bounded operand")


def _bound_multiply(left: Any, right: Any) -> Bound:
    if _is_bound(left) and _is_bound(right):
        products = torch.stack(
            (left[0] * right[0], left[0] * right[1], left[1] * right[0], left[1] * right[1])
        )
        lower, upper = products.amin(dim=0), products.amax(dim=0)
        magnitude = torch.maximum(left[0].abs(), left[1].abs()) * torch.maximum(
            right[0].abs(), right[1].abs()
        )
        return _outward(lower, upper, magnitude=magnitude, terms=2)
    bound, scalar = (left, right) if _is_bound(left) else (right, left)
    if not _is_bound(bound) or not isinstance(scalar, (int, float, Tensor)):
        raise UnsupportedVerificationOperator(
            "multiplication requires a bounded tensor and numeric factor"
        )
    first, second = bound[0] * scalar, bound[1] * scalar
    lower, upper = torch.minimum(first, second), torch.maximum(first, second)
    magnitude = (
        torch.maximum(bound[0].abs(), bound[1].abs())
        * torch.as_tensor(scalar, device=lower.device, dtype=lower.dtype).abs()
    )
    return _outward(lower, upper, magnitude=magnitude, terms=2)


def _normalise_logits(logits: Any) -> Tensor:
    if not isinstance(logits, Tensor):
        raise ValueError("formal verification requires a tensor model output")
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError("formal verification requires [batch, classes] logits")
    if logits.shape[-1] == 1:
        value = logits[:, 0]
        logits = torch.stack((-0.5 * value, 0.5 * value), dim=-1)
    return logits


def _normalise_logit_bounds(bound: Bound) -> Bound:
    lower, upper = bound
    if lower.ndim == 1:
        lower, upper = lower.unsqueeze(0), upper.unsqueeze(0)
    if lower.ndim != 2:
        raise UnsupportedVerificationOperator("bounded output is not [batch, classes]")
    if lower.shape[-1] == 1:
        return (
            torch.stack((-0.5 * upper[:, 0], 0.5 * lower[:, 0]), dim=-1),
            torch.stack((-0.5 * lower[:, 0], 0.5 * upper[:, 0]), dim=-1),
        )
    return lower, upper


def _clean_margin(logits: Tensor, target: int) -> tuple[float, int | None]:
    competitors = [index for index in range(int(logits.numel())) if index != target]
    if not competitors:
        return float("nan"), None
    values = logits[competitors]
    position = int(values.argmax().item())
    competitor = competitors[position]
    return float((logits[target] - logits[competitor]).item()), competitor


def _unsupported_point(
    epsilon: float,
    clean_prediction: int,
    target: int,
    clean_margin: float,
    reason: str,
) -> FormalVerificationPoint:
    return FormalVerificationPoint(
        epsilon=epsilon,
        status="unsupported",
        certified=False,
        clean_prediction=clean_prediction,
        target=target,
        clean_margin=clean_margin,
        certified_margin_lower_bound=None,
        worst_competitor=None,
        unsupported_reason=reason,
    )


def _cohort_point(
    epsilon: float,
    results: list[FormalVerificationPoint],
    labels: list[int],
) -> CertifiedAccuracyPoint:
    n = len(results)
    clean_correct = sum(
        result.clean_prediction == label for result, label in zip(results, labels, strict=True)
    )
    certified = sum(
        result.certified and result.clean_prediction == label
        for result, label in zip(results, labels, strict=True)
    )
    unsupported = sum(result.status == "unsupported" for result in results)
    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted(set(labels)):
        positions = [index for index, value in enumerate(labels) if value == label]
        class_clean = sum(results[index].clean_prediction == label for index in positions)
        class_certified = sum(
            results[index].certified and results[index].clean_prediction == label
            for index in positions
        )
        per_class[str(label)] = {
            "n": len(positions),
            "clean_correct": class_clean,
            "certified": class_certified,
            "certified_accuracy": class_certified / len(positions),
            "clean_conditional_certified_accuracy": (
                class_certified / class_clean if class_clean else 0.0
            ),
        }
    return CertifiedAccuracyPoint(
        epsilon=float(epsilon),
        n_samples=n,
        n_clean_correct=clean_correct,
        n_certified=certified,
        certified_accuracy=certified / n,
        clean_conditional_certified_accuracy=(certified / clean_correct if clean_correct else 0.0),
        inconclusive_clean_correct=clean_correct - certified,
        unsupported=unsupported,
        per_class=per_class,
    )


def _layer_record(
    name: str,
    module: nn.Module,
    bound: Bound,
    *,
    input_bound: Bound,
) -> LayerIntervalRecord:
    lower, upper = bound
    width = upper - lower
    unstable: float | None = None
    if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
        input_lower, input_upper = input_bound
        unstable = float(((input_lower < 0) & (input_upper > 0)).double().mean().item())
    return LayerIntervalRecord(
        name=name,
        operator=type(module).__qualname__,
        lower_min=float(lower.min().item()),
        upper_max=float(upper.max().item()),
        maximum_width=float(width.max().item()),
        unstable_fraction=unstable,
    )


def _operator_ledger(points: list[FormalVerificationPoint]) -> list[dict[str, str]]:
    if not points:
        return []
    return [
        {"name": record.name, "operator": record.operator, "transformer": "sound_interval"}
        for record in points[0].layer_intervals
    ]


def _bound_bottleneck(point: FormalVerificationPoint | None) -> dict[str, Any] | None:
    if point is None or not point.layer_intervals:
        return None
    widest = max(point.layer_intervals, key=lambda record: record.maximum_width)
    relu_records = [
        record for record in point.layer_intervals if record.unstable_fraction is not None
    ]
    most_unstable = (
        max(relu_records, key=lambda record: float(record.unstable_fraction or 0.0))
        if relu_records
        else None
    )
    return {
        "epsilon": point.epsilon,
        "widest_layer": widest.name,
        "widest_layer_operator": widest.operator,
        "maximum_interval_width": widest.maximum_width,
        "most_unstable_nonlinearity": (None if most_unstable is None else most_unstable.name),
        "unstable_fraction": (None if most_unstable is None else most_unstable.unstable_fraction),
        "interpretation": (
            "This identifies where the IBP relaxation becomes widest; it is a "
            "verification bottleneck, not necessarily a causal model weakness."
        ),
    }


def _fetch_attr(module: nn.Module, target: str) -> Any:
    value: Any = module
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _is_bound(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], Tensor)
        and isinstance(value[1], Tensor)
    )


def _module_device(model: nn.Module) -> torch.device:
    for value in (*model.parameters(), *model.buffers()):
        return value.device
    return torch.device("cpu")


def _module_dtype(model: nn.Module) -> torch.dtype:
    for value in (*model.parameters(), *model.buffers()):
        if value.is_floating_point():
            return value.dtype
    return torch.float32


def _certificate_html(report: FormalRobustnessCertificate) -> str:
    sample_rows = "".join(
        "<tr>"
        f"<td>{point.epsilon:g}</td>"
        f"<td class='{point.status}'>{escape(point.status.upper())}</td>"
        f"<td>{_number(point.clean_margin)}</td>"
        f"<td>{_number(point.certified_margin_lower_bound)}</td>"
        f"<td>{escape(str(point.worst_competitor if point.worst_competitor is not None else '—'))}</td>"
        "</tr>"
        for point in report.sample_points
    )
    cohort_rows = "".join(
        "<tr>"
        f"<td>{point.epsilon:g}</td><td>{point.n_certified}/{point.n_samples}</td>"
        f"<td>{point.certified_accuracy:.3f}</td>"
        f"<td>{point.clean_conditional_certified_accuracy:.3f}</td>"
        f"<td>{point.inconclusive_clean_correct}</td><td>{point.unsupported}</td>"
        "</tr>"
        for point in report.cohort_points
    )
    inconclusive_point = next(
        (point for point in report.sample_points if not point.certified), None
    )
    layer_rows = "".join(
        "<tr>"
        f"<td>{escape(record.name)}</td><td>{escape(record.operator)}</td>"
        f"<td>{record.maximum_width:.5g}</td>"
        f"<td>{_number(record.unstable_fraction)}</td>"
        "</tr>"
        for record in ([] if inconclusive_point is None else inconclusive_point.layer_intervals)
    )
    limitations = "".join(f"<li>{escape(value)}</li>" for value in report.limitations)
    status = "CERTIFIED" if report.certified_radius_lower_bound > 0 else "NO POSITIVE RADIUS PROVEN"
    input_contract = (
        {"bounds": report.input_domain, "status": "legacy / semantics undeclared"}
        if report.input_spec is None
        else asdict(report.input_spec)
    )
    return f"""<!doctype html><html lang='en' data-theme='dark'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>VRAVEN formal robustness certificate</title>
<style>:root{{--bg:#030A12;--panel:#07111F;--raised:#0B1728;--line:#1C344D;--text:#F1F5F9;--muted:#94A3B8;--ok:#32D74B;--warn:#FFD60A;--bad:#FF2D55}}
html[data-theme='light']{{--bg:#fff;--panel:#fff;--raised:#f7f9fc;--line:#cbd5e1;--text:#111827;--muted:#526173;--ok:#047857;--warn:#92400e;--bad:#be123c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:28px}}header,.hero,.panel{{border:1px solid var(--line);background:var(--panel);border-radius:14px;padding:20px;margin-bottom:15px}}header{{display:flex;justify-content:space-between;gap:16px}}h1{{margin:3px 0;font-size:25px}}h2{{font-size:15px}}p,li{{color:var(--muted)}}.k{{color:var(--ok);font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}}.hero{{border-left:5px solid var(--ok)}}.big{{font-size:30px;font-weight:850}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:9px}}.metric{{padding:11px;border:1px solid var(--line);border-radius:9px;background:var(--raised)}}.metric span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}}.metric strong{{font:700 18px ui-monospace,monospace}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);text-transform:uppercase;font-size:10px}}td.certified{{color:var(--ok);font-weight:800}}td.inconclusive{{color:var(--warn);font-weight:800}}td.unsupported{{color:var(--bad);font-weight:800}}button{{padding:8px;border:1px solid var(--line);border-radius:7px;background:var(--raised);color:var(--text)}}code,pre{{font-family:ui-monospace,monospace}}</style>
<script>function theme(t){{document.documentElement.dataset.theme=t}}</script></head><body><main>
<header><div><div class='k'>Formal verification · {escape(report.verifier)}</div><h1>Local L∞ robustness certificate</h1><p>{escape(report.model_name)} · domain {escape(str(report.input_domain))}</p></div><div><button onclick="theme('dark')">Dark</button> <button onclick="theme('light')">Publication light</button></div></header>
<section class='hero'><div class='k'>Verifier-computed result</div><div class='big'>{status}</div><p>The largest tested radius proven for the focus sample is <strong>ε={report.certified_radius_lower_bound:g}</strong>. Inconclusive points are not adversarial counterexamples.</p><div class='metrics'><div class='metric'><span>Certified radius lower bound</span><strong>{report.certified_radius_lower_bound:g}</strong></div><div class='metric'><span>First inconclusive ε</span><strong>{_number(report.first_inconclusive_epsilon)}</strong></div><div class='metric'><span>Verified property</span><strong>argmax invariant</strong></div></div></section>
<section class='panel'><h2>Focus-sample proof curve</h2><table><thead><tr><th>ε</th><th>Verdict</th><th>Clean margin</th><th>Certified margin lower bound</th><th>Worst competitor</th></tr></thead><tbody>{sample_rows}</tbody></table></section>
<section class='panel'><h2>Cohort certified accuracy</h2><table><thead><tr><th>ε</th><th>Certified / all</th><th>Certified accuracy</th><th>Conditional certified</th><th>Inconclusive clean-correct</th><th>Unsupported</th></tr></thead><tbody>{cohort_rows or '<tr><td colspan="6">No cohort was supplied.</td></tr>'}</tbody></table></section>
<section class='panel'><h2>Bound-growth ledger at first inconclusive ε</h2><p>This localises verifier relaxation growth. It does not by itself identify a causal weakness.</p><table><thead><tr><th>Layer</th><th>Operator</th><th>Maximum interval width</th><th>Unstable activation fraction</th></tr></thead><tbody>{layer_rows or '<tr><td colspan="4">Every tested point was certified or the graph was unsupported.</td></tr>'}</tbody></table></section>
<section class='panel'><h2>Declared input contract</h2><pre>{escape(json.dumps(input_contract, indent=2))}</pre></section>
<section class='panel'><h2>Proof semantics and limitations</h2><p>{escape(report.proof_semantics)}</p><ul>{limitations}</ul><pre>{escape(json.dumps(report.metadata, indent=2))}</pre></section></main></body></html>"""


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.6g}"
