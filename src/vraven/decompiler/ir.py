"""Architecture-neutral mechanism intermediate representation for VRAVEN.

The IR is deliberately framework-neutral.  PyTorch capture adapters populate it, while
concept discovery, causal analysis, programme extraction and visualisation consume it.
No tensor values are stored in the IR by default; only structure, metadata and hashes are
persisted.  This keeps the artefact small and avoids leaking model inputs or activations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"


class CaptureRoute(str, Enum):
    """Supported routes used to recover model computation."""

    EXPORT = "torch_export"
    FX = "torch_fx"
    DISPATCH = "runtime_dispatch"
    MODULE_HOOKS = "module_hooks"


class CaptureStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeKind(str, Enum):
    INPUT = "input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    OPERATION = "operation"
    CONTROL_FLOW = "control_flow"
    OUTPUT = "output"
    UNRESOLVED = "unresolved"


class MechanismRole(str, Enum):
    """Typed high-level operation vocabulary used by the mechanism dialect."""

    INPUT = "input"
    TRANSFORM = "transform"
    DETECT = "detect"
    GATE = "gate"
    SUPPRESS = "suppress"
    NORMALISE = "normalise"
    POOL = "pool"
    MERGE = "merge"
    ROUTE = "route"
    ACCUMULATE = "accumulate"
    COMPARE = "compare"
    STORE = "store"
    RETRIEVE = "retrieve"
    ATTEND = "attend"
    REPEAT = "repeat"
    READOUT = "readout"
    SHAPE = "shape"
    RANDOM = "random"
    OUTPUT = "output"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Serializable tensor metadata.

    Symbolic dimensions are represented as strings.  ``numel`` and ``nbytes`` are only
    present when all dimensions are concrete.
    """

    shape: tuple[int | str, ...] = ()
    dtype: str | None = None
    device: str | None = None
    requires_grad: bool | None = None
    stride: tuple[int | str, ...] | None = None
    numel: int | None = None
    nbytes: int | None = None

    @property
    def rank(self) -> int:
        return len(self.shape)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> TensorSpec | None:
        if payload is None:
            return None
        return cls(
            shape=tuple(payload.get("shape", ())),
            dtype=payload.get("dtype"),
            device=payload.get("device"),
            requires_grad=payload.get("requires_grad"),
            stride=(tuple(payload["stride"]) if payload.get("stride") is not None else None),
            numel=payload.get("numel"),
            nbytes=payload.get("nbytes"),
        )


@dataclass(frozen=True, slots=True)
class SourceLocation:
    module_path: str | None = None
    operator_source: str | None = None
    stack_trace: str | None = None
    file: str | None = None
    line: int | None = None


@dataclass(slots=True)
class IRNode:
    id: str
    name: str
    kind: NodeKind
    op: str
    target: str
    role: MechanismRole
    inputs: list[str] = field(default_factory=list)
    tensor: TensorSpec | None = None
    parameter_count: int = 0
    differentiable: bool = False
    intervention_capable: bool = False
    semantically_recognised: bool = False
    source: SourceLocation = field(default_factory=SourceLocation)
    metadata: dict[str, Any] = field(default_factory=dict)
    unresolved_reason: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IRNode:
        source_payload = payload.get("source") or {}
        return cls(
            id=str(payload["id"]),
            name=str(payload.get("name", payload["id"])),
            kind=NodeKind(payload.get("kind", NodeKind.OPERATION.value)),
            op=str(payload.get("op", "")),
            target=str(payload.get("target", "")),
            role=MechanismRole(payload.get("role", MechanismRole.UNKNOWN.value)),
            inputs=[str(item) for item in payload.get("inputs", ())],
            tensor=TensorSpec.from_mapping(payload.get("tensor")),
            parameter_count=int(payload.get("parameter_count", 0)),
            differentiable=bool(payload.get("differentiable", False)),
            intervention_capable=bool(payload.get("intervention_capable", False)),
            semantically_recognised=bool(payload.get("semantically_recognised", False)),
            source=SourceLocation(**source_payload),
            metadata=dict(payload.get("metadata", {})),
            unresolved_reason=payload.get("unresolved_reason"),
        )


@dataclass(slots=True)
class IREdge:
    id: str
    source: str
    target: str
    tensor: TensorSpec | None = None
    edge_type: str = "tensor_flow"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def volume(self) -> int | None:
        return self.tensor.numel if self.tensor is not None else None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IREdge:
        return cls(
            id=str(payload["id"]),
            source=str(payload["source"]),
            target=str(payload["target"]),
            tensor=TensorSpec.from_mapping(payload.get("tensor")),
            edge_type=str(payload.get("edge_type", "tensor_flow")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ShapeConstraint:
    input_name: str
    dimension: int
    minimum: int | None = None
    maximum: int | None = None
    expression: str | None = None
    source: str = "observed"


@dataclass(frozen=True, slots=True)
class CaptureAttempt:
    route: CaptureRoute
    status: CaptureStatus
    duration_ms: float
    message: str = ""
    exception_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CaptureCoverage:
    """Separate coverage measurements rather than a single opaque confidence score."""

    executed_nodes: int = 0
    recognised_nodes: int = 0
    differentiable_nodes: int = 0
    gradient_nodes: int = 0
    intervention_candidates: int = 0
    intervention_nodes: int = 0
    captured_parameter_count: int = 0
    total_parameter_count: int = 0
    unresolved_nodes: int = 0
    numeric_equivalence_max_abs: float | None = None
    numeric_equivalence_max_rel: float | None = None
    numeric_equivalence_passed: bool | None = None
    runtime_weighted_recognition: float | None = None
    flop_weighted_recognition: float | None = None
    memory_weighted_recognition: float | None = None

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    @property
    def operator_coverage(self) -> float:
        return self._ratio(self.recognised_nodes, self.executed_nodes)

    @property
    def gradient_coverage(self) -> float:
        return self._ratio(self.gradient_nodes, self.differentiable_nodes)

    @property
    def intervention_coverage(self) -> float:
        return self._ratio(self.intervention_nodes, self.intervention_candidates)

    @property
    def parameter_coverage(self) -> float:
        return self._ratio(self.captured_parameter_count, self.total_parameter_count)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> CaptureCoverage:
        return cls(**dict(payload or {}))


@dataclass(slots=True)
class MechanismIR:
    """VRAVEN's architecture-neutral representation of an observed model computation."""

    model_class: str
    framework: str
    framework_version: str
    capture_route: CaptureRoute
    nodes: list[IRNode]
    edges: list[IREdge]
    inputs: list[str]
    outputs: list[str]
    attempts: list[CaptureAttempt] = field(default_factory=list)
    constraints: list[ShapeConstraint] = field(default_factory=list)
    coverage: CaptureCoverage = field(default_factory=CaptureCoverage)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def node_map(self) -> dict[str, IRNode]:
        return {node.id: node for node in self.nodes}

    def get_node(self, node_id_or_name: str) -> IRNode:
        for node in self.nodes:
            if node.id == node_id_or_name or node.name == node_id_or_name:
                return node
        raise KeyError(f"No VRAVEN IR node matches {node_id_or_name!r}")

    def topological_index(self) -> dict[str, int]:
        return {node.id: index for index, node in enumerate(self.nodes)}

    def predecessors(self, node_id: str) -> list[str]:
        return [edge.source for edge in self.edges if edge.target == node_id]

    def successors(self, node_id: str) -> list[str]:
        return [edge.target for edge in self.edges if edge.source == node_id]

    def unresolved(self) -> list[IRNode]:
        return [node for node in self.nodes if not node.semantically_recognised]

    def mechanism_nodes(self) -> list[IRNode]:
        return [
            node
            for node in self.nodes
            if node.kind in {NodeKind.OPERATION, NodeKind.CONTROL_FLOW, NodeKind.UNRESOLVED}
        ]

    def to_dict(self) -> dict[str, Any]:
        return _normalise_enum_values(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MechanismIR:
        return cls(
            model_class=str(payload["model_class"]),
            framework=str(payload.get("framework", "pytorch")),
            framework_version=str(payload.get("framework_version", "unknown")),
            capture_route=CaptureRoute(payload["capture_route"]),
            nodes=[IRNode.from_mapping(item) for item in payload.get("nodes", ())],
            edges=[IREdge.from_mapping(item) for item in payload.get("edges", ())],
            inputs=[str(item) for item in payload.get("inputs", ())],
            outputs=[str(item) for item in payload.get("outputs", ())],
            attempts=[
                CaptureAttempt(
                    route=CaptureRoute(item["route"]),
                    status=CaptureStatus(item["status"]),
                    duration_ms=float(item.get("duration_ms", 0.0)),
                    message=str(item.get("message", "")),
                    exception_type=item.get("exception_type"),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload.get("attempts", ())
            ],
            constraints=[ShapeConstraint(**item) for item in payload.get("constraints", ())],
            coverage=CaptureCoverage.from_mapping(payload.get("coverage")),
            metadata=dict(payload.get("metadata", {})),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> MechanismIR:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_class": self.model_class,
            "framework": f"{self.framework} {self.framework_version}",
            "capture_route": self.capture_route.value,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "inputs": len(self.inputs),
            "outputs": len(self.outputs),
            "operator_coverage": self.coverage.operator_coverage,
            "gradient_coverage": self.coverage.gradient_coverage,
            "intervention_coverage": self.coverage.intervention_coverage,
            "parameter_coverage": self.coverage.parameter_coverage,
            "unresolved_nodes": self.coverage.unresolved_nodes,
            "numeric_equivalence_passed": self.coverage.numeric_equivalence_passed,
            "hash": self.content_hash(),
        }


def _normalise_enum_values(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalise_enum_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_enum_values(item) for item in value]
    return value


def stable_node_id(index: int, name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"n{index:04d}_{digest}"


def stable_edge_id(source: str, target: str, ordinal: int = 0) -> str:
    digest = hashlib.sha1(
        f"{source}->{target}:{ordinal}".encode(), usedforsecurity=False
    ).hexdigest()[:10]
    return f"e_{digest}"


def ensure_unique_names(names: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for raw_name in names:
        count = counts.get(raw_name, 0)
        counts[raw_name] = count + 1
        output.append(raw_name if count == 0 else f"{raw_name}_{count}")
    return output


def extract_input_nodes(nodes: Sequence[IRNode]) -> list[str]:
    return [node.id for node in nodes if node.kind == NodeKind.INPUT]


def extract_output_nodes(nodes: Sequence[IRNode]) -> list[str]:
    return [node.id for node in nodes if node.kind == NodeKind.OUTPUT]
