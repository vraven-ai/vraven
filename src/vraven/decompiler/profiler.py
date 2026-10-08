"""Execution and provenance profiling for the VRAVEN mechanism IR.

The primary profiler executes the captured FX graph node by node and reports median and
quantile latency.  These timings describe interpreter-visible node cost and must not be
confused with compiler-kernel attribution.  An optional ``torch.profiler`` summary records
operator-level CPU or accelerator events when available.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .capture import CapturedModel
from .causal_graph import CausalMechanismGraph
from .runtime import FXValueRecorder


@dataclass(slots=True)
class ProfilerConfig:
    warmup_runs: int = 2
    measured_runs: int = 8
    use_torch_profiler: bool = False
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False


@dataclass(slots=True)
class NodeProfile:
    node_id: str
    name: str
    target: str
    role: str
    median_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    tensor_bytes: int | None
    parameter_count: int
    causal_effect: float | None = None
    effect_per_ms: float | None = None
    effect_per_megabyte: float | None = None
    source_module: str | None = None
    source_file: str | None = None
    source_line: int | None = None


@dataclass(slots=True)
class ProfilerReport:
    nodes: list[NodeProfile]
    total_median_ms: float
    torch_profiler_events: list[dict[str, Any]]
    limitations: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(item) for item in self.nodes],
            "total_median_ms": self.total_median_ms,
            "torch_profiler_events": self.torch_profiler_events,
            "limitations": list(self.limitations),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination


class MechanismProfiler:
    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self.config = config or ProfilerConfig()

    def profile(
        self,
        captured: CapturedModel,
        args: Sequence[Any],
        *,
        kwargs: Mapping[str, Any] | None = None,
        causal_graph: CausalMechanismGraph | None = None,
    ) -> ProfilerReport:
        if captured.graph_module is None:
            return ProfilerReport(
                nodes=[],
                total_median_ms=0.0,
                torch_profiler_events=[],
                limitations=(
                    "Per-node profiling requires an FX or Export graph capture.",
                ),
                metadata={"capture_route": captured.ir.capture_route.value},
            )
        call_args = tuple(args)
        call_kwargs = dict(kwargs or {})
        for _ in range(self.config.warmup_runs):
            FXValueRecorder(captured.graph_module, detach=True).execute(call_args, call_kwargs)
        timings: dict[str, list[float]] = {}
        for _ in range(self.config.measured_runs):
            record = FXValueRecorder(
                captured.graph_module,
                detach=True,
                move_to_cpu=False,
                profile=True,
            ).execute(call_args, call_kwargs)
            for node_id, duration in record.durations_ms.items():
                timings.setdefault(node_id, []).append(float(duration))
        causal_by_node: dict[str, float] = {}
        if causal_graph is not None:
            for concept in causal_graph.nodes:
                causal_by_node[concept.node_id] = causal_by_node.get(concept.node_id, 0.0) + abs(
                    float(concept.output_effect)
                )
        profiles: list[NodeProfile] = []
        for node in captured.ir.nodes:
            values = np.asarray(timings.get(node.id, ()), dtype=float)
            if values.size == 0:
                continue
            effect = causal_by_node.get(node.id)
            median = float(np.median(values))
            tensor_bytes = node.tensor.nbytes if node.tensor is not None else None
            profiles.append(
                NodeProfile(
                    node_id=node.id,
                    name=node.name,
                    target=node.target,
                    role=node.role.value,
                    median_ms=median,
                    p10_ms=float(np.quantile(values, 0.10)),
                    p90_ms=float(np.quantile(values, 0.90)),
                    mean_ms=float(np.mean(values)),
                    tensor_bytes=tensor_bytes,
                    parameter_count=node.parameter_count,
                    causal_effect=effect,
                    effect_per_ms=(effect / max(median, 1e-9) if effect is not None else None),
                    effect_per_megabyte=(
                        effect / max(tensor_bytes / (1024**2), 1e-9)
                        if effect is not None and tensor_bytes is not None
                        else None
                    ),
                    source_module=node.source.module_path,
                    source_file=node.source.file,
                    source_line=node.source.line,
                )
            )
        events = self._torch_profiler(captured, call_args, call_kwargs)
        return ProfilerReport(
            nodes=profiles,
            total_median_ms=float(sum(item.median_ms for item in profiles)),
            torch_profiler_events=events,
            limitations=(
                "FX interpreter node timings include Python dispatch overhead and are not equivalent to optimised kernel timings.",
                "Compiler provenance to generated Triton or C++ kernels is version-dependent and is not asserted unless explicitly captured.",
                "Causal effect per unit cost is a descriptive ratio, not a model-quality objective.",
            ),
            metadata={
                "warmup_runs": self.config.warmup_runs,
                "measured_runs": self.config.measured_runs,
                "capture_route": captured.ir.capture_route.value,
                "torch_version": torch.__version__,
            },
        )

    def _torch_profiler(
        self,
        captured: CapturedModel,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.config.use_torch_profiler:
            return []
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            tensors = [item for item in _tensor_leaves((args, kwargs))]
            if any(item.device.type == "cuda" for item in tensors) and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
            ) as profiler:
                captured.graph_module(*args, **kwargs)
            output = []
            for event in profiler.key_averages()[:100]:
                output.append(
                    {
                        "key": event.key,
                        "count": int(event.count),
                        "cpu_time_total_us": float(event.cpu_time_total),
                        "self_cpu_time_total_us": float(event.self_cpu_time_total),
                        "cpu_memory_usage": int(getattr(event, "cpu_memory_usage", 0)),
                        "cuda_time_total_us": float(getattr(event, "cuda_time_total", 0.0)),
                        "cuda_memory_usage": int(getattr(event, "cuda_memory_usage", 0)),
                    }
                )
            return output
        except Exception as error:
            return [{"error": f"{type(error).__name__}: {error}"}]


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    leaves, _ = torch.utils._pytree.tree_flatten(value)
    return [item for item in leaves if isinstance(item, torch.Tensor)]
