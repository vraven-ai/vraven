#!/usr/bin/env python3
"""Generate ``docs/CODE_INVENTORY.md`` from the executable repository.

The inventory is deliberately generated from the source tree rather than maintained by
hand. This gives contributors a reliable map of implemented modules and prevents
roadmap ideas from being mistaken for working code.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "vraven"
TEST_ROOT = ROOT / "tests"
OUTPUT = ROOT / "docs" / "CODE_INVENTORY.md"


def line_count(path: Path) -> int:
    """Return the number of physical lines in a UTF-8 text file."""

    return len(path.read_text(encoding="utf-8").splitlines())


def module_summary(path: Path) -> str:
    """Return the first sentence of a module docstring or a conservative fallback."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstring = ast.get_docstring(tree, clean=True)
    if not docstring:
        return "Executable VRAVEN module."
    first_line = " ".join(docstring.splitlines()).strip()
    if ". " in first_line:
        first_line = first_line.split(". ", 1)[0] + "."
    return first_line.replace("|", "\\|")


def test_names(path: Path) -> list[str]:
    """Return top-level test function names in source order."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def humanise_test(name: str) -> str:
    return name.removeprefix("test_").replace("_", " ")


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build_inventory() -> str:
    source_files = sorted(SOURCE_ROOT.rglob("*.py"))
    test_files = sorted(TEST_ROOT.rglob("test_*.py"))
    source_lines = sum(line_count(path) for path in source_files)
    test_count = sum(len(test_names(path)) for path in test_files)

    lines: list[str] = [
        "# VRAVEN Code Inventory",
        "",
        "This inventory is generated from the executable repository by "
        "`scripts/generate_code_inventory.py`. It helps contributors distinguish "
        "working modules from roadmap ideas. Every Python source file listed below "
        "contains executable code.",
        "",
        "## Package summary",
        "",
        f"* Python modules: `{len(source_files)}`",
        f"* Python source lines: `{source_lines}`",
        f"* Test modules: `{len(test_files)}`",
        f"* Test cases: `{test_count}`",
        "* Native framework adapter: PyTorch",
        "* Official render modes: `interactive_dark` and `publication_light`",
        "* Optional native crate: `rust_core`",
        "",
        "## Source modules",
        "",
        "| Module | Lines | Responsibility |",
        "| --- | ---: | --- |",
    ]
    for path in source_files:
        lines.append(
            f"| `{relative(path)}` | {line_count(path)} | {module_summary(path)} |"
        )

    lines.extend(
        [
            "",
            "## Tests",
            "",
            "| Test module | Test cases |",
            "| --- | --- |",
        ]
    )
    for path in test_files:
        names = test_names(path)
        description = ", ".join(humanise_test(name) for name in names) or "module import checks"
        lines.append(f"| `{relative(path)}` | {description} |")

    examples = sorted((ROOT / "examples").glob("*.py"))
    benchmarks = sorted((ROOT / "benchmarks").glob("*.py"))
    lines.extend(["", "## Examples and benchmarks", ""])
    for path in [*examples, *benchmarks]:
        lines.append(f"* `{relative(path)}`")

    support_files = [
        ROOT / "SECURITY.md",
        ROOT / "CITATION.cff",
        ROOT / "Makefile",
        ROOT / ".editorconfig",
        ROOT / "scripts" / "build_package.sh",
        ROOT / "scripts" / "generate_code_inventory.py",
        ROOT / "scripts" / "quality_check.sh",
        ROOT / "scripts" / "release_validate.sh",
        ROOT / "scripts" / "run_vraven_artifact.py",
        ROOT / "scripts" / "run_vraven_on_models.py",
        ROOT / "scripts" / "run_tests.sh",
        ROOT / "scripts" / "wheel_smoke.py",
        ROOT / "scripts" / "create_release.py",
        ROOT / ".github" / "workflows" / "release.yml",
        ROOT / ".github" / "workflows" / "tests.yml",
    ]
    lines.extend(["", "## Build and repository support", ""])
    for path in support_files:
        if path.exists():
            lines.append(f"* `{relative(path)}`")

    documentation = sorted((ROOT / "docs").glob("*.md"))
    lines.extend(["", "## Documentation", ""])
    for path in documentation:
        lines.append(f"* `{relative(path)}`")

    lines.extend(
        [
            "",
            "## Rust core",
            "",
            "`rust_core/src/lib.rs` contains parallel activation summary, cosine "
            "similarity and linear resampling primitives. It is intentionally not "
            "bound into the Python wheel until profiling and numerical parity tests "
            "justify the additional build complexity. The Python implementation "
            "remains authoritative in the current research alpha.",
            "",
            "## Explicitly absent",
            "",
            "The repository does not contain fake TensorFlow, ONNX, custom CUDA or "
            "VRAVEN Studio implementations. Those items remain in the roadmap. Contributors "
            "must add real code, tests and documentation before changing their "
            "status.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(build_inventory(), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
