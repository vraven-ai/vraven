from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_PATTERNS = (
    re.compile(r"\bTODO\b"),
    re.compile(r"\bFIXME\b"),
    re.compile(r"\bNotImplementedError\b"),
    re.compile(r"^\s*pass\s*(?:#.*)?$", re.MULTILINE),
)


def test_source_files_are_non_empty_and_not_placeholders() -> None:
    root = Path(__file__).parents[1] / "src" / "vraven"
    source_files = sorted(root.rglob("*.py"))
    assert source_files, "VRAVEN source tree is missing"

    empty = [path for path in source_files if not path.read_text(encoding="utf-8").strip()]
    assert empty == []

    violations: list[str] = []
    for path in source_files:
        content = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(content):
                violations.append(f"{path.relative_to(root)} matched {pattern.pattern}")
    assert violations == []
