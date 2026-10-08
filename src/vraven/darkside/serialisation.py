"""Persistence helpers for DarkSide result files."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import AnalysisResult


def load_result(path: str | Path) -> AnalysisResult:
    """Load and validate a versioned VRAVEN DarkSide analysis result."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "vraven.darkside.v1":
        raise ValueError("Unsupported or missing VRAVEN DarkSide schema version")
    return AnalysisResult(
        analysis=payload["analysis"],
        metrics=payload.get("metrics", {}),
        tables=payload.get("tables", {}),
        metadata=payload.get("metadata", {}),
        warnings=payload.get("warnings", []),
        limitations=payload.get("limitations", []),
        schema_version=payload["schema_version"],
    )
