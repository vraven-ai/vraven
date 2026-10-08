"""Operator semantics registry used to lift PyTorch operations into VRAVEN's dialect."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .ir import MechanismRole


@dataclass(frozen=True, slots=True)
class OperatorSemantics:
    patterns: tuple[str, ...]
    role: MechanismRole
    family: str
    description: str
    intervention: bool
    feature_axis: str = "auto"

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        tokens = set(re.split(r"[^a-z0-9_]+", lowered))
        for raw_pattern in self.patterns:
            pattern = raw_pattern.lower()
            if not pattern:
                continue
            # Short operator names such as ``ge`` and ``lt`` must be token matched;
            # naïve substring matching would classify ``weight`` as a comparison.
            if len(pattern) <= 3 and pattern.replace("_", "").isalnum():
                if pattern in tokens:
                    return True
            elif pattern in lowered:
                return True
        return False


class OperatorSemanticsRegistry:
    """Deterministic registry mapping low-level operations to typed mechanism roles.

    The registry describes operation *semantics*, not what a trained feature means.  A
    convolution can be classified as a local detector, but its learned concept remains
    unresolved until concept grounding and intervention validation provide evidence.
    """

    def __init__(self, entries: Iterable[OperatorSemantics] | None = None) -> None:
        self._entries = tuple(entries or self._load_default_entries())

    @staticmethod
    def _load_default_entries() -> tuple[OperatorSemantics, ...]:
        resource = resources.files("vraven.decompiler.data").joinpath("operator_semantics.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
        return tuple(
            OperatorSemantics(
                patterns=tuple(item["patterns"]),
                role=MechanismRole(item["role"]),
                family=str(item["family"]),
                description=str(item["description"]),
                intervention=bool(item["intervention"]),
                feature_axis=str(item.get("feature_axis", "auto")),
            )
            for item in payload
        )

    @property
    def entries(self) -> tuple[OperatorSemantics, ...]:
        return self._entries

    def resolve(
        self,
        target: Any,
        *,
        op: str = "",
        module_type: str | None = None,
    ) -> OperatorSemantics:
        text = " ".join(
            part
            for part in (
                op,
                module_type or "",
                canonical_target_name(target),
            )
            if part
        )
        for entry in self._entries:
            if entry.matches(text):
                return entry
        return OperatorSemantics(
            patterns=(),
            role=MechanismRole.UNKNOWN,
            family="unresolved",
            description="No registered VRAVEN operator semantics",
            intervention=False,
            feature_axis="auto",
        )

    def coverage(self, targets: Iterable[Any]) -> float:
        resolved = [self.resolve(target) for target in targets]
        return (
            sum(item.role != MechanismRole.UNKNOWN for item in resolved) / len(resolved)
            if resolved
            else 0.0
        )

    def export(self) -> list[dict[str, Any]]:
        return [
            {
                "patterns": list(entry.patterns),
                "role": entry.role.value,
                "family": entry.family,
                "description": entry.description,
                "intervention": entry.intervention,
                "feature_axis": entry.feature_axis,
            }
            for entry in self._entries
        ]


def canonical_target_name(target: Any) -> str:
    """Return a stable, human-readable target name for FX, ATen and Python callables."""

    if target is None:
        return ""
    if isinstance(target, str):
        return target
    schema = getattr(target, "_schema", None)
    if schema is not None:
        name = getattr(schema, "name", None)
        overload = getattr(schema, "overload_name", None)
        if name:
            return f"{name}.{overload}" if overload else str(name)
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if name:
        module = getattr(target, "__module__", None)
        return f"{module}.{name}" if module else str(name)
    text = str(target)
    return re.sub(r"\s+", " ", text).strip()


def module_parameter_count(module: Any, *, recurse: bool = False) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(parameter.numel()) for parameter in parameters(recurse=recurse))


def semantics_metadata(semantics: OperatorSemantics) -> dict[str, Any]:
    return {
        "mechanism_family": semantics.family,
        "mechanism_description": semantics.description,
        "feature_axis_hint": semantics.feature_axis,
        "operator_semantics_patterns": list(semantics.patterns),
    }
