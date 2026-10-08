"""Multi-route PyTorch model capture for the VRAVEN causal decompiler.

The router prefers ``torch.export`` because it produces a normalised, ATen-level full
model graph.  It falls back to FX symbolic tracing, runtime operator dispatch tracing and
finally module hooks.  Every failed attempt is retained in the resulting certificate.

The fallback routes do not silently claim full capture.  Their limitations are reflected
in route-specific metadata and coverage values.
"""

from __future__ import annotations

import hashlib
import inspect
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils import _pytree as pytree

from .ir import (
    CaptureAttempt,
    CaptureCoverage,
    CaptureRoute,
    CaptureStatus,
    IREdge,
    IRNode,
    MechanismIR,
    MechanismRole,
    NodeKind,
    ShapeConstraint,
    SourceLocation,
    TensorSpec,
    stable_edge_id,
)
from .semantics import (
    OperatorSemanticsRegistry,
    canonical_target_name,
    module_parameter_count,
    semantics_metadata,
)


@dataclass(slots=True)
class CaptureConfig:
    """Controls capture route order and validation behaviour."""

    preferred_routes: tuple[CaptureRoute, ...] = (
        CaptureRoute.EXPORT,
        CaptureRoute.FX,
        CaptureRoute.DISPATCH,
        CaptureRoute.MODULE_HOOKS,
    )
    strict_export: bool = True
    functionalise_export: bool = True
    dynamic_shapes: Any | None = None
    auto_dynamic_batch: bool = True
    validate_equivalence: bool = True
    validate_gradients: bool = True
    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 1e-4
    dispatch_include_view_ops: bool = True
    max_stack_trace_chars: int = 4000


@dataclass(slots=True)
class CapturedModel:
    """Runtime capture artefact.

    ``graph_module`` is available for export and FX routes and is the basis for graph
    compiled interventions.  Runtime fallback routes still produce a useful IR, but do
    not claim graph rewriting support.
    """

    ir: MechanismIR
    original_model: nn.Module
    graph_module: GraphModule | None
    exported_program: Any | None = None
    example_args: tuple[Any, ...] = ()
    example_kwargs: dict[str, Any] = field(default_factory=dict)
    route_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_graph_interventions(self) -> bool:
        return self.graph_module is not None

    def summary(self) -> dict[str, Any]:
        payload = self.ir.summary()
        payload["graph_interventions"] = self.supports_graph_interventions
        return payload


class ModelCaptureError(RuntimeError):
    """Raised when every configured capture route fails."""


class ModelCaptureRouter:
    def __init__(
        self,
        config: CaptureConfig | None = None,
        semantics: OperatorSemanticsRegistry | None = None,
    ) -> None:
        self.config = config or CaptureConfig()
        self.semantics = semantics or OperatorSemanticsRegistry()

    def capture(
        self,
        model: nn.Module,
        example_args: Sequence[Any],
        example_kwargs: Mapping[str, Any] | None = None,
    ) -> CapturedModel:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        args = tuple(example_args)
        kwargs = dict(example_kwargs or {})
        model.eval()
        attempts: list[CaptureAttempt] = []

        for route in self.config.preferred_routes:
            started = time.perf_counter()
            try:
                if route == CaptureRoute.EXPORT:
                    captured = self._capture_export(model, args, kwargs)
                elif route == CaptureRoute.FX:
                    captured = self._capture_fx(model, args, kwargs)
                elif route == CaptureRoute.DISPATCH:
                    captured = self._capture_dispatch(model, args, kwargs)
                elif route == CaptureRoute.MODULE_HOOKS:
                    captured = self._capture_hooks(model, args, kwargs)
                else:  # pragma: no cover
                    raise ValueError(f"Unsupported capture route: {route}")
                duration = (time.perf_counter() - started) * 1000.0
                attempts.append(
                    CaptureAttempt(
                        route=route,
                        status=CaptureStatus.SUCCESS,
                        duration_ms=duration,
                        message="capture succeeded",
                    )
                )
                captured.ir.attempts = list(attempts)
                return captured
            except Exception as error:
                duration = (time.perf_counter() - started) * 1000.0
                attempts.append(
                    CaptureAttempt(
                        route=route,
                        status=CaptureStatus.FAILED,
                        duration_ms=duration,
                        message=str(error),
                        exception_type=type(error).__name__,
                        metadata={"traceback": _trim_traceback(self.config.max_stack_trace_chars)},
                    )
                )

        messages = "\n".join(
            f"{attempt.route.value}: {attempt.exception_type}: {attempt.message}"
            for attempt in attempts
        )
        raise ModelCaptureError(f"VRAVEN could not capture the model with any route:\n{messages}")

    def _capture_export(
        self,
        model: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> CapturedModel:
        if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
            raise RuntimeError("torch.export is unavailable in this PyTorch build")
        export_kwargs: dict[str, Any] = {
            "args": args,
            "kwargs": kwargs,
            "strict": self.config.strict_export,
        }
        dynamic_shapes = self.config.dynamic_shapes
        if dynamic_shapes is None and self.config.auto_dynamic_batch:
            dynamic_shapes = _auto_dynamic_batch_shapes(model, args, kwargs)
        if dynamic_shapes is not None:
            export_kwargs["dynamic_shapes"] = dynamic_shapes
        try:
            with torch.no_grad():
                exported_program = torch.export.export(model, **export_kwargs)
        except Exception:
            # Dynamic-shape inference is a convenience.  If it is incompatible with the
            # model, retry the official static export rather than abandoning export.
            if "dynamic_shapes" not in export_kwargs or self.config.dynamic_shapes is not None:
                raise
            export_kwargs.pop("dynamic_shapes", None)
            with torch.no_grad():
                exported_program = torch.export.export(model, **export_kwargs)
            dynamic_shapes = None
        functionalisation_warning: str | None = None
        if self.config.functionalise_export and hasattr(exported_program, "run_decompositions"):
            try:
                exported_program = exported_program.run_decompositions()
            except Exception as functionalisation_error:
                # Functionalisation is desirable, not required.  The original exported
                # program remains sound and the limitation is recorded after IR creation.
                functionalisation_warning = str(functionalisation_error)
            else:
                functionalisation_warning = None
        graph_module = exported_program.module()
        ir = _ir_from_fx(
            graph_module,
            model=model,
            route=CaptureRoute.EXPORT,
            semantics=self.semantics,
            exported_program=exported_program,
        )
        ir.constraints.extend(_export_constraints(exported_program))
        ir.metadata.update(
            {
                "capture_level": "normalised_aten_graph",
                "strict_export": self.config.strict_export,
                "functionalised": self.config.functionalise_export and functionalisation_warning is None,
                "functionalisation_warning": functionalisation_warning,
                "dynamic_shapes_requested": dynamic_shapes is not None,
                "graph_signature": str(getattr(exported_program, "graph_signature", "")),
                "range_constraints": str(getattr(exported_program, "range_constraints", "")),
                "private_or_unstable_apis": [],
            }
        )
        if self.config.validate_equivalence:
            _apply_equivalence_validation(
                ir,
                original=model,
                captured=graph_module,
                args=args,
                kwargs=kwargs,
                atol=self.config.absolute_tolerance,
                rtol=self.config.relative_tolerance,
            )
        if self.config.validate_gradients:
            _apply_gradient_validation(ir, graph_module, args=args, kwargs=kwargs)
        return CapturedModel(
            ir=ir,
            original_model=model,
            graph_module=graph_module,
            exported_program=exported_program,
            example_args=args,
            example_kwargs=kwargs,
        )

    def _capture_fx(
        self,
        model: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> CapturedModel:
        graph_module = symbolic_trace(model)
        try:
            ShapeProp(graph_module).propagate(*args, **kwargs)
        except TypeError:
            # Older ShapeProp versions do not accept keyword arguments.
            if kwargs:
                with torch.no_grad():
                    graph_module(*args, **kwargs)
            else:
                raise
        ir = _ir_from_fx(
            graph_module,
            model=model,
            route=CaptureRoute.FX,
            semantics=self.semantics,
            exported_program=None,
        )
        ir.metadata.update(
            {
                "capture_level": "python_fx_graph",
                "limitation": (
                    "FX symbolic tracing may not capture data-dependent Python control flow "
                    "or operations that bypass proxy tracing."
                ),
                "private_or_unstable_apis": [],
            }
        )
        if self.config.validate_equivalence:
            _apply_equivalence_validation(
                ir,
                original=model,
                captured=graph_module,
                args=args,
                kwargs=kwargs,
                atol=self.config.absolute_tolerance,
                rtol=self.config.relative_tolerance,
            )
        if self.config.validate_gradients:
            _apply_gradient_validation(ir, graph_module, args=args, kwargs=kwargs)
        return CapturedModel(
            ir=ir,
            original_model=model,
            graph_module=graph_module,
            example_args=args,
            example_kwargs=kwargs,
        )

    def _capture_dispatch(
        self,
        model: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> CapturedModel:
        tracer = RuntimeDispatchTracer(model=model, semantics=self.semantics)
        tracer.register_inputs(args, kwargs)
        with torch.no_grad(), tracer:
            output = model(*args, **kwargs)
        ir = tracer.build_ir(args=args, kwargs=kwargs, output=output)
        ir.metadata.update(
            {
                "capture_level": "executed_aten_path",
                "limitation": (
                    "Runtime dispatch captures only the execution path taken by the supplied "
                    "example and relies on a private PyTorch dispatch API isolated in this adapter."
                ),
                "private_or_unstable_apis": ["torch.utils._python_dispatch.TorchDispatchMode"],
            }
        )
        return CapturedModel(
            ir=ir,
            original_model=model,
            graph_module=None,
            example_args=args,
            example_kwargs=kwargs,
        )

    def _capture_hooks(
        self,
        model: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> CapturedModel:
        tracer = ModuleHookTracer(model=model, semantics=self.semantics)
        tracer.register_inputs(args, kwargs)
        with torch.no_grad(), tracer:
            output = model(*args, **kwargs)
        ir = tracer.build_ir(args=args, kwargs=kwargs, output=output)
        ir.metadata.update(
            {
                "capture_level": "module_boundaries_only",
                "limitation": (
                    "Module hooks do not expose functional tensor operations between modules; "
                    "operator coverage is therefore a conservative lower bound."
                ),
                "private_or_unstable_apis": [],
            }
        )
        return CapturedModel(
            ir=ir,
            original_model=model,
            graph_module=None,
            example_args=args,
            example_kwargs=kwargs,
        )


def _ir_from_fx(
    graph_module: GraphModule,
    *,
    model: nn.Module,
    route: CaptureRoute,
    semantics: OperatorSemanticsRegistry,
    exported_program: Any | None,
) -> MechanismIR:
    nodes: list[IRNode] = []
    edges: list[IREdge] = []
    node_ids: dict[Node, str] = {}
    modules = dict(graph_module.named_modules())
    model_buffer_names = set(dict(model.named_buffers()).keys())
    total_parameters = sum(int(parameter.numel()) for parameter in model.parameters())

    for _index, fx_node in enumerate(graph_module.graph.nodes):
        node_id = fx_node.name
        node_ids[fx_node] = node_id
        kind = _kind_from_fx_node(fx_node, exported_program=exported_program)
        if fx_node.op == "get_attr" and str(fx_node.target) in model_buffer_names:
            kind = NodeKind.BUFFER
        module = modules.get(str(fx_node.target)) if fx_node.op == "call_module" else None
        module_type = type(module).__qualname__ if module is not None else None
        resolved = semantics.resolve(
            fx_node.target,
            op=fx_node.op,
            module_type=module_type,
        )
        target_name = canonical_target_name(fx_node.target)
        if fx_node.op == "call_module" and module_type:
            target_name = f"{fx_node.target}:{module_type}"
        tensor = _tensor_spec_from_fx_meta(fx_node.meta)
        inputs = [node_ids[item] for item in fx_node.all_input_nodes if item in node_ids]
        source = _source_from_fx_meta(fx_node.meta)
        parameter_count = module_parameter_count(module, recurse=False) if module is not None else 0
        differentiable = bool(tensor and tensor.requires_grad)
        if kind == NodeKind.INPUT:
            role = MechanismRole.INPUT
            recognised = True
            intervention_capable = False
        elif kind in {NodeKind.PARAMETER, NodeKind.BUFFER}:
            role = MechanismRole.RETRIEVE
            recognised = True
            intervention_capable = False
            if tensor is not None and kind == NodeKind.PARAMETER:
                parameter_count = int(tensor.numel or 0)
        elif kind == NodeKind.OUTPUT:
            role = MechanismRole.OUTPUT
            recognised = True
            intervention_capable = False
        else:
            role = resolved.role
            recognised = role != MechanismRole.UNKNOWN
            intervention_capable = bool(resolved.intervention and tensor is not None)
        metadata = {
            "fx_op": fx_node.op,
            "fx_target": target_name,
            "fx_node_name": fx_node.name,
            "users": [user.name for user in fx_node.users],
            **semantics_metadata(resolved),
        }
        if fx_node.meta.get("nn_module_stack"):
            metadata["nn_module_stack"] = _json_safe(fx_node.meta["nn_module_stack"])
        if fx_node.meta.get("source_fn_stack"):
            metadata["source_fn_stack"] = _json_safe(fx_node.meta["source_fn_stack"])
        unresolved_reason = None if recognised else f"No semantics registered for {target_name}"
        nodes.append(
            IRNode(
                id=node_id,
                name=fx_node.name,
                kind=kind if recognised or kind != NodeKind.OPERATION else NodeKind.UNRESOLVED,
                op=fx_node.op,
                target=target_name,
                role=role,
                inputs=inputs,
                tensor=tensor,
                parameter_count=parameter_count,
                differentiable=differentiable,
                intervention_capable=intervention_capable,
                semantically_recognised=recognised,
                source=source,
                metadata=metadata,
                unresolved_reason=unresolved_reason,
            )
        )

    edge_ordinal: dict[tuple[str, str], int] = {}
    node_map = {node.id: node for node in nodes}
    for target_node in nodes:
        for source_id in target_node.inputs:
            key = (source_id, target_node.id)
            ordinal = edge_ordinal.get(key, 0)
            edge_ordinal[key] = ordinal + 1
            source_tensor = node_map.get(source_id).tensor if source_id in node_map else None
            edges.append(
                IREdge(
                    id=stable_edge_id(source_id, target_node.id, ordinal),
                    source=source_id,
                    target=target_node.id,
                    tensor=source_tensor,
                )
            )

    coverage = _calculate_coverage(nodes, total_parameters)
    return MechanismIR(
        model_class=type(model).__qualname__,
        framework="pytorch",
        framework_version=torch.__version__,
        capture_route=route,
        nodes=nodes,
        edges=edges,
        inputs=[node.id for node in nodes if node.kind == NodeKind.INPUT],
        outputs=[node.id for node in nodes if node.kind == NodeKind.OUTPUT],
        coverage=coverage,
        metadata={
            "model_training_state": bool(model.training),
            "graph_code": graph_module.code,
            "model_parameter_count": total_parameters,
        },
    )


def _kind_from_fx_node(node: Node, *, exported_program: Any | None) -> NodeKind:
    if node.op == "placeholder":
        if exported_program is not None:
            signature = getattr(exported_program, "graph_signature", None)
            if signature is not None:
                for spec in getattr(signature, "input_specs", ()):
                    argument = getattr(spec, "arg", None)
                    if getattr(argument, "name", None) == node.name:
                        kind_name = type(getattr(spec, "kind", None)).__name__
                        kind_value = str(getattr(spec, "kind", ""))
                        combined = f"{kind_name} {kind_value}".lower()
                        if "parameter" in combined:
                            return NodeKind.PARAMETER
                        if "buffer" in combined:
                            return NodeKind.BUFFER
        return NodeKind.INPUT
    if node.op == "get_attr":
        return NodeKind.PARAMETER
    if node.op == "output":
        return NodeKind.OUTPUT
    target = canonical_target_name(node.target).lower()
    if any(token in target for token in ("higher_order.cond", "while_loop", "scan", "map_impl")):
        return NodeKind.CONTROL_FLOW
    return NodeKind.OPERATION


def _tensor_spec_from_fx_meta(meta: Mapping[str, Any]) -> TensorSpec | None:
    tensor_meta = meta.get("tensor_meta")
    if tensor_meta is not None:
        shape = tuple(_symbolic_dimension(item) for item in getattr(tensor_meta, "shape", ()))
        stride = tuple(_symbolic_dimension(item) for item in getattr(tensor_meta, "stride", ()))
        dtype = str(getattr(tensor_meta, "dtype", None))
        requires_grad = bool(getattr(tensor_meta, "requires_grad", False))
        numel = _concrete_numel(shape)
        return TensorSpec(
            shape=shape,
            dtype=dtype,
            device=str(getattr(tensor_meta, "device", "unknown")),
            requires_grad=requires_grad,
            stride=stride,
            numel=numel,
            nbytes=_nbytes(numel, dtype),
        )
    value = meta.get("val")
    if isinstance(value, Tensor):
        return tensor_spec(value)
    if isinstance(value, (tuple, list)):
        tensors = [item for item in value if isinstance(item, Tensor)]
        if tensors:
            first = tensor_spec(tensors[0])
            return TensorSpec(
                shape=first.shape,
                dtype=first.dtype,
                device=first.device,
                requires_grad=any(item.requires_grad for item in tensors),
                stride=first.stride,
                numel=sum(int(item.numel()) for item in tensors),
                nbytes=sum(int(item.numel() * item.element_size()) for item in tensors),
            )
    return None


def tensor_spec(value: Tensor) -> TensorSpec:
    shape = tuple(_symbolic_dimension(item) for item in value.shape)
    stride = tuple(_symbolic_dimension(item) for item in value.stride())
    return TensorSpec(
        shape=shape,
        dtype=str(value.dtype),
        device=str(value.device),
        requires_grad=bool(value.requires_grad),
        stride=stride,
        numel=int(value.numel()),
        nbytes=int(value.numel() * value.element_size()),
    )


def _symbolic_dimension(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _concrete_numel(shape: Sequence[int | str]) -> int | None:
    product = 1
    for dimension in shape:
        if not isinstance(dimension, int):
            return None
        product *= dimension
    return product


def _nbytes(numel: int | None, dtype: str | None) -> int | None:
    if numel is None or dtype is None:
        return None
    sizes = {
        "torch.float64": 8,
        "torch.float32": 4,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int64": 8,
        "torch.int32": 4,
        "torch.int16": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.bool": 1,
    }
    size = sizes.get(dtype)
    return numel * size if size is not None else None


def _source_from_fx_meta(meta: Mapping[str, Any]) -> SourceLocation:
    stack_trace = meta.get("stack_trace")
    module_path = None
    module_stack = meta.get("nn_module_stack")
    if isinstance(module_stack, Mapping) and module_stack:
        module_path = str(next(reversed(module_stack.keys())))
    operator_source = None
    source_fn_stack = meta.get("source_fn_stack")
    if source_fn_stack:
        operator_source = str(source_fn_stack[-1])
    file_name = None
    line_number = None
    if stack_trace:
        for line in str(stack_trace).splitlines():
            stripped = line.strip()
            if stripped.startswith('File "'):
                try:
                    file_name = stripped.split('File "', 1)[1].split('"', 1)[0]
                    line_number = int(stripped.split("line ", 1)[1].split(",", 1)[0])
                except (IndexError, ValueError):
                    file_name, line_number = None, None
                break
    return SourceLocation(
        module_path=module_path,
        operator_source=operator_source,
        stack_trace=str(stack_trace) if stack_trace else None,
        file=file_name,
        line=line_number,
    )


def _calculate_coverage(nodes: Sequence[IRNode], total_parameters: int) -> CaptureCoverage:
    operations = [
        node
        for node in nodes
        if node.kind in {NodeKind.OPERATION, NodeKind.CONTROL_FLOW, NodeKind.UNRESOLVED}
    ]
    differentiable = [node for node in operations if node.differentiable]
    intervention_candidates = [node for node in operations if node.tensor is not None]
    parameter_nodes = [node for node in nodes if node.kind is NodeKind.PARAMETER]
    captured_parameters = sum(node.parameter_count for node in parameter_nodes)
    if not captured_parameters:
        # Export graph parameters are lifted placeholders; coverage is established from the
        # model parameter total and graph signature rather than per-node module ownership.
        captured_parameters = total_parameters
    return CaptureCoverage(
        executed_nodes=len(operations),
        recognised_nodes=sum(node.semantically_recognised for node in operations),
        differentiable_nodes=len(differentiable),
        gradient_nodes=len(differentiable),
        intervention_candidates=len(intervention_candidates),
        intervention_nodes=sum(node.intervention_capable for node in intervention_candidates),
        captured_parameter_count=captured_parameters,
        total_parameter_count=total_parameters,
        unresolved_nodes=sum(not node.semantically_recognised for node in operations),
    )


def _apply_gradient_validation(
    ir: MechanismIR,
    graph_module: GraphModule,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Probe which captured tensor nodes participate in a usable autograd path.

    Export is intentionally performed under ``no_grad`` for capture stability, so FX
    tensor metadata alone cannot establish gradient availability.  This probe executes
    the captured graph once with floating inputs requiring gradients and records whether
    each operation receives a non-null reverse-mode derivative from the model output.
    Failure is retained as metadata rather than converted into a false coverage claim.
    """

    try:
        from .runtime import FXValueRecorder

        def prepare(item: Any) -> Any:
            if isinstance(item, Tensor):
                clone = item.detach().clone()
                if clone.dtype.is_floating_point or clone.dtype.is_complex:
                    clone.requires_grad_(True)
                return clone
            return item

        grad_args = tuple(pytree.tree_map(prepare, args))
        grad_kwargs = dict(pytree.tree_map(prepare, kwargs))
        recorder = FXValueRecorder(graph_module, detach=False, move_to_cpu=False)
        record = recorder.execute(grad_args, grad_kwargs)
        output_leaves, _ = pytree.tree_flatten(record.output)
        outputs = [
            item
            for item in output_leaves
            if isinstance(item, Tensor)
            and (item.dtype.is_floating_point or item.dtype.is_complex)
            and item.requires_grad
        ]
        if not outputs:
            ir.metadata["gradient_probe"] = {
                "passed": False,
                "reason": "No differentiable floating output was observed.",
            }
            return
        scalar = sum(item.real.sum() if item.dtype.is_complex else item.sum() for item in outputs)
        candidate_nodes: list[IRNode] = []
        candidate_tensors: list[Tensor] = []
        for node in ir.mechanism_nodes():
            value = record.values.get(node.name)
            leaves, _ = pytree.tree_flatten(value)
            tensor = next(
                (
                    item
                    for item in leaves
                    if isinstance(item, Tensor)
                    and (item.dtype.is_floating_point or item.dtype.is_complex)
                    and item.requires_grad
                ),
                None,
            )
            if tensor is None:
                continue
            node.differentiable = True
            candidate_nodes.append(node)
            candidate_tensors.append(tensor)
        gradients = torch.autograd.grad(
            scalar,
            candidate_tensors,
            allow_unused=True,
            retain_graph=False,
        ) if candidate_tensors else ()
        gradient_nodes = 0
        for node, gradient in zip(candidate_nodes, gradients, strict=True):
            available = gradient is not None
            node.metadata["gradient_available"] = available
            if available:
                gradient_nodes += 1
        ir.coverage.differentiable_nodes = len(candidate_nodes)
        ir.coverage.gradient_nodes = gradient_nodes
        ir.metadata["gradient_probe"] = {
            "passed": True,
            "candidate_nodes": len(candidate_nodes),
            "gradient_nodes": gradient_nodes,
        }
    except Exception as error:
        ir.metadata["gradient_probe"] = {
            "passed": False,
            "reason": str(error),
            "exception_type": type(error).__name__,
        }


def _apply_equivalence_validation(
    ir: MechanismIR,
    *,
    original: nn.Module,
    captured: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    atol: float,
    rtol: float,
) -> None:
    training_original = original.training
    training_captured = getattr(captured, "training", None)
    original.eval()
    try:
        captured.eval()
    except (AttributeError, RuntimeError) as state_error:
        # Modules returned by ExportedProgram.module() may deliberately reject
        # train/eval changes.  The state was already captured from ``model.eval()``.
        ir.metadata["captured_eval_warning"] = str(state_error)
    with torch.no_grad():
        expected = original(*args, **kwargs)
        actual = captured(*args, **kwargs)
    max_abs, max_rel, passed = compare_outputs(expected, actual, atol=atol, rtol=rtol)
    ir.coverage.numeric_equivalence_max_abs = max_abs
    ir.coverage.numeric_equivalence_max_rel = max_rel
    ir.coverage.numeric_equivalence_passed = passed
    original.train(training_original)
    if training_captured is not None:
        try:
            captured.train(training_captured)
        except (AttributeError, RuntimeError) as state_error:
            ir.metadata["captured_state_restore_warning"] = str(state_error)


def compare_outputs(
    expected: Any,
    actual: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> tuple[float, float, bool]:
    expected_flat, expected_spec = pytree.tree_flatten(expected)
    actual_flat, actual_spec = pytree.tree_flatten(actual)
    if expected_spec != actual_spec or len(expected_flat) != len(actual_flat):
        return float("inf"), float("inf"), False
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for expected_item, actual_item in zip(expected_flat, actual_flat, strict=True):
        if isinstance(expected_item, Tensor) and isinstance(actual_item, Tensor):
            if expected_item.shape != actual_item.shape:
                return float("inf"), float("inf"), False
            expected_cpu = expected_item.detach().to("cpu")
            actual_cpu = actual_item.detach().to("cpu")
            if expected_cpu.dtype == torch.bool or not (expected_cpu.is_floating_point() or expected_cpu.is_complex()):
                item_passed = bool(torch.equal(expected_cpu, actual_cpu))
                abs_error = 0.0 if item_passed else float("inf")
                rel_error = abs_error
            else:
                difference = (expected_cpu - actual_cpu).abs()
                abs_error = float(difference.max().item()) if difference.numel() else 0.0
                denominator = expected_cpu.abs().clamp_min(1e-12)
                rel_error = float((difference / denominator).max().item()) if difference.numel() else 0.0
                item_passed = bool(torch.allclose(expected_cpu, actual_cpu, atol=atol, rtol=rtol))
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
            passed = passed and item_passed
        else:
            passed = passed and expected_item == actual_item
    return max_abs, max_rel, passed


def _auto_dynamic_batch_shapes(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Create a shared dynamic batch dimension for tensor inputs when possible."""

    try:
        signature = inspect.signature(model.forward)
        bound = signature.bind_partial(*args, **dict(kwargs))
        batch = torch.export.Dim("vraven_batch", min=1)
    except Exception:
        return None

    def specification(value: Any) -> Any:
        if isinstance(value, Tensor):
            return {0: batch} if value.ndim > 0 else None
        if isinstance(value, tuple):
            return tuple(specification(item) for item in value)
        if isinstance(value, list):
            return [specification(item) for item in value]
        if isinstance(value, Mapping):
            return {key: specification(item) for key, item in value.items()}
        return None

    result = {name: specification(value) for name, value in bound.arguments.items()}
    return result if any(value is not None for value in result.values()) else None


def _export_constraints(exported_program: Any) -> list[ShapeConstraint]:
    constraints: list[ShapeConstraint] = []
    range_constraints = getattr(exported_program, "range_constraints", None)
    if isinstance(range_constraints, Mapping):
        for symbol, value_range in range_constraints.items():
            lower = getattr(value_range, "lower", None)
            upper = getattr(value_range, "upper", None)
            constraints.append(
                ShapeConstraint(
                    input_name=str(symbol),
                    dimension=-1,
                    minimum=_finite_int(lower),
                    maximum=_finite_int(upper),
                    expression=f"{symbol} in {value_range}",
                    source="torch.export.range_constraints",
                )
            )
    return constraints


def _finite_int(value: Any) -> int | None:
    try:
        numeric = int(value)
    except Exception:
        return None
    return numeric


def _trim_traceback(limit: int) -> str:
    text = traceback.format_exc()
    return text[-limit:]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


class _ModuleStack:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.stack: list[str] = []
        self.handles: list[Any] = []

    def __enter__(self) -> _ModuleStack:
        for name, module in self.model.named_modules():
            if not name:
                continue
            self.handles.append(
                module.register_forward_pre_hook(
                    lambda _module, _args, module_name=name: self.stack.append(module_name)
                )
            )
            self.handles.append(
                module.register_forward_hook(
                    lambda _module, _args, _output, module_name=name: self._pop(module_name)
                )
            )
        return self

    def _pop(self, module_name: str) -> None:
        if self.stack and self.stack[-1] == module_name:
            self.stack.pop()
        elif module_name in self.stack:
            self.stack.remove(module_name)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stack.clear()

    @property
    def current(self) -> str | None:
        return self.stack[-1] if self.stack else None


try:  # private API isolated here and reported in capture metadata
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:  # pragma: no cover
    TorchDispatchMode = object  # type: ignore[assignment,misc]


class RuntimeDispatchTracer(TorchDispatchMode):  # type: ignore[misc]
    """Capture the actually executed ATen path through ``__torch_dispatch__``."""

    def __init__(self, *, model: nn.Module, semantics: OperatorSemanticsRegistry) -> None:
        super().__init__()
        self.model = model
        self.semantics = semantics
        self.nodes: list[IRNode] = []
        self.edges: list[IREdge] = []
        self.tensor_producers: dict[int, str] = {}
        self._module_stack = _ModuleStack(model)
        self._edge_ordinal: dict[tuple[str, str], int] = {}
        self._input_ids: list[str] = []
        self._output_ids: list[str] = []
        self._started = False

    def __enter__(self) -> RuntimeDispatchTracer:
        self._module_stack.__enter__()
        super().__enter__()
        self._started = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        super().__exit__(exc_type, exc, tb)
        self._module_stack.__exit__(exc_type, exc, tb)
        self._started = False

    def register_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        flat, _ = pytree.tree_flatten((args, kwargs))
        tensor_index = 0
        for item in flat:
            if not isinstance(item, Tensor):
                continue
            node_id = f"input_{tensor_index}"
            tensor_index += 1
            self.nodes.append(
                IRNode(
                    id=node_id,
                    name=node_id,
                    kind=NodeKind.INPUT,
                    op="placeholder",
                    target=node_id,
                    role=MechanismRole.INPUT,
                    tensor=tensor_spec(item),
                    differentiable=bool(item.requires_grad),
                    intervention_capable=False,
                    semantically_recognised=True,
                )
            )
            self.tensor_producers[id(item)] = node_id
            self._input_ids.append(node_id)
        for name, parameter in self.model.named_parameters():
            node_id = f"parameter::{name}"
            self.nodes.append(
                IRNode(
                    id=node_id,
                    name=name,
                    kind=NodeKind.PARAMETER,
                    op="parameter",
                    target=name,
                    role=MechanismRole.RETRIEVE,
                    tensor=tensor_spec(parameter),
                    parameter_count=int(parameter.numel()),
                    differentiable=bool(parameter.requires_grad),
                    intervention_capable=False,
                    semantically_recognised=True,
                )
            )
            self.tensor_producers[id(parameter)] = node_id
        for name, buffer in self.model.named_buffers():
            node_id = f"buffer::{name}"
            self.nodes.append(
                IRNode(
                    id=node_id,
                    name=name,
                    kind=NodeKind.BUFFER,
                    op="buffer",
                    target=name,
                    role=MechanismRole.RETRIEVE,
                    tensor=tensor_spec(buffer),
                    parameter_count=0,
                    differentiable=False,
                    intervention_capable=False,
                    semantically_recognised=True,
                )
            )
            self.tensor_producers[id(buffer)] = node_id

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        kwargs = dict(kwargs or {})
        output = func(*args, **kwargs)
        target_name = canonical_target_name(func)
        resolved = self.semantics.resolve(func, op="call_function")
        node_id = f"op_{len([node for node in self.nodes if node.kind in {NodeKind.OPERATION, NodeKind.UNRESOLVED}])}_{_short_hash(target_name)}"
        input_tensors = _tensor_leaves((args, kwargs))
        input_ids = []
        for tensor in input_tensors:
            producer = self.tensor_producers.get(id(tensor))
            if producer is not None and producer not in input_ids:
                input_ids.append(producer)
        output_tensors = _tensor_leaves(output)
        first_tensor = output_tensors[0] if output_tensors else None
        recognised = resolved.role != MechanismRole.UNKNOWN
        node = IRNode(
            id=node_id,
            name=target_name,
            kind=NodeKind.OPERATION if recognised else NodeKind.UNRESOLVED,
            op="call_function",
            target=target_name,
            role=resolved.role,
            inputs=input_ids,
            tensor=tensor_spec(first_tensor) if first_tensor is not None else None,
            differentiable=any(tensor.requires_grad for tensor in output_tensors),
            intervention_capable=bool(resolved.intervention and output_tensors),
            semantically_recognised=recognised,
            source=SourceLocation(module_path=self._module_stack.current),
            metadata={
                **semantics_metadata(resolved),
                "output_tensor_count": len(output_tensors),
                "module_stack_leaf": self._module_stack.current,
            },
            unresolved_reason=None if recognised else f"No semantics registered for {target_name}",
        )
        self.nodes.append(node)
        for source_id in input_ids:
            key = (source_id, node_id)
            ordinal = self._edge_ordinal.get(key, 0)
            self._edge_ordinal[key] = ordinal + 1
            self.edges.append(
                IREdge(
                    id=stable_edge_id(source_id, node_id, ordinal),
                    source=source_id,
                    target=node_id,
                    tensor=node.tensor,
                )
            )
        for tensor in output_tensors:
            self.tensor_producers[id(tensor)] = node_id
        return output

    def build_ir(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> MechanismIR:
        if not self._input_ids:
            # Inputs must be registered before execution.  The router calls build after
            # execution, so retrofit only when a custom user directly employs this tracer.
            self.register_inputs(args, kwargs)
        output_sources = []
        for tensor in _tensor_leaves(output):
            producer = self.tensor_producers.get(id(tensor))
            if producer is not None and producer not in output_sources:
                output_sources.append(producer)
        output_node = IRNode(
            id="output",
            name="output",
            kind=NodeKind.OUTPUT,
            op="output",
            target="output",
            role=MechanismRole.OUTPUT,
            inputs=output_sources,
            tensor=tensor_spec(_tensor_leaves(output)[0]) if _tensor_leaves(output) else None,
            semantically_recognised=True,
        )
        self.nodes.append(output_node)
        for source_id in output_sources:
            self.edges.append(
                IREdge(
                    id=stable_edge_id(source_id, "output"),
                    source=source_id,
                    target="output",
                    tensor=next((node.tensor for node in self.nodes if node.id == source_id), None),
                )
            )
        total_parameters = sum(int(parameter.numel()) for parameter in self.model.parameters())
        coverage = _calculate_coverage(self.nodes, total_parameters)
        return MechanismIR(
            model_class=type(self.model).__qualname__,
            framework="pytorch",
            framework_version=torch.__version__,
            capture_route=CaptureRoute.DISPATCH,
            nodes=self.nodes,
            edges=self.edges,
            inputs=self._input_ids,
            outputs=["output"],
            coverage=coverage,
            metadata={"model_parameter_count": total_parameters},
        )


class ModuleHookTracer:
    """Final conservative fallback that captures leaf-module boundaries."""

    def __init__(self, *, model: nn.Module, semantics: OperatorSemanticsRegistry) -> None:
        self.model = model
        self.semantics = semantics
        self.handles: list[Any] = []
        self.nodes: list[IRNode] = []
        self.edges: list[IREdge] = []
        self.tensor_producers: dict[int, str] = {}
        self._input_ids: list[str] = []
        self._call_index = 0

    def __enter__(self) -> ModuleHookTracer:
        for name, module in self.model.named_modules():
            if not name or any(True for _ in module.children()):
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def register_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        for index, tensor in enumerate(_tensor_leaves((args, kwargs))):
            node_id = f"input_{index}"
            self.nodes.append(
                IRNode(
                    id=node_id,
                    name=node_id,
                    kind=NodeKind.INPUT,
                    op="placeholder",
                    target=node_id,
                    role=MechanismRole.INPUT,
                    tensor=tensor_spec(tensor),
                    differentiable=bool(tensor.requires_grad),
                    semantically_recognised=True,
                )
            )
            self.tensor_producers[id(tensor)] = node_id
            self._input_ids.append(node_id)

    def _hook(self, module_name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            node_id = f"module_{self._call_index}_{_short_hash(module_name)}"
            self._call_index += 1
            input_ids = []
            for tensor in _tensor_leaves(inputs):
                producer = self.tensor_producers.get(id(tensor))
                if producer is not None and producer not in input_ids:
                    input_ids.append(producer)
            output_tensors = _tensor_leaves(output)
            semantics = self.semantics.resolve(
                type(module).__qualname__,
                op="call_module",
                module_type=type(module).__qualname__,
            )
            recognised = semantics.role != MechanismRole.UNKNOWN
            node = IRNode(
                id=node_id,
                name=module_name,
                kind=NodeKind.OPERATION if recognised else NodeKind.UNRESOLVED,
                op="call_module",
                target=f"{module_name}:{type(module).__qualname__}",
                role=semantics.role,
                inputs=input_ids,
                tensor=tensor_spec(output_tensors[0]) if output_tensors else None,
                parameter_count=module_parameter_count(module, recurse=False),
                differentiable=any(tensor.requires_grad for tensor in output_tensors),
                intervention_capable=bool(semantics.intervention and output_tensors),
                semantically_recognised=recognised,
                source=SourceLocation(module_path=module_name),
                metadata=semantics_metadata(semantics),
                unresolved_reason=None if recognised else f"No semantics registered for {type(module).__qualname__}",
            )
            self.nodes.append(node)
            for source_id in input_ids:
                self.edges.append(
                    IREdge(
                        id=stable_edge_id(source_id, node_id),
                        source=source_id,
                        target=node_id,
                        tensor=node.tensor,
                    )
                )
            for tensor in output_tensors:
                self.tensor_producers[id(tensor)] = node_id

        return hook

    def build_ir(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> MechanismIR:
        if not self._input_ids:
            self.register_inputs(args, kwargs)
        sources = []
        for tensor in _tensor_leaves(output):
            producer = self.tensor_producers.get(id(tensor))
            if producer is not None and producer not in sources:
                sources.append(producer)
        output_node = IRNode(
            id="output",
            name="output",
            kind=NodeKind.OUTPUT,
            op="output",
            target="output",
            role=MechanismRole.OUTPUT,
            inputs=sources,
            tensor=tensor_spec(_tensor_leaves(output)[0]) if _tensor_leaves(output) else None,
            semantically_recognised=True,
        )
        self.nodes.append(output_node)
        for source in sources:
            self.edges.append(
                IREdge(
                    id=stable_edge_id(source, "output"),
                    source=source,
                    target="output",
                )
            )
        total_parameters = sum(int(parameter.numel()) for parameter in self.model.parameters())
        coverage = _calculate_coverage(self.nodes, total_parameters)
        return MechanismIR(
            model_class=type(self.model).__qualname__,
            framework="pytorch",
            framework_version=torch.__version__,
            capture_route=CaptureRoute.MODULE_HOOKS,
            nodes=self.nodes,
            edges=self.edges,
            inputs=self._input_ids,
            outputs=["output"],
            coverage=coverage,
            metadata={"model_parameter_count": total_parameters},
        )


def _tensor_leaves(value: Any) -> list[Tensor]:
    flat, _ = pytree.tree_flatten(value)
    return [item for item in flat if isinstance(item, Tensor)]


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
