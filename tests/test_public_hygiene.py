from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
PUBLIC_AREAS = (
    ROOT / ".github",
    ROOT / "assets",
    ROOT / "benchmarks",
    ROOT / "docs",
    ROOT / "examples",
    ROOT / "scripts",
    ROOT / "site",
    ROOT / "src",
    ROOT / "tests",
)


def _public_files() -> list[Path]:
    files = [
        ROOT / "CHANGELOG.md",
        ROOT / "CITATION.cff",
        ROOT / "CONTRIBUTING.md",
        ROOT / "LICENSE",
        ROOT / "Makefile",
        ROOT / "NOTICE",
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "pyproject.toml",
    ]
    for directory in PUBLIC_AREAS:
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix not in {".pyc", ".pyo"}
            and not any(
                part in {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
                for part in path.relative_to(ROOT).parts
            )
        )
    return sorted(set(files))


def test_public_tree_has_no_development_instruction_files() -> None:
    forbidden_names = {
        "agents.md",
        "copilot-instructions.md",
        "claude.md",
    }
    violations = []
    for path in _public_files():
        name = path.name.lower()
        if (
            name in forbidden_names
            or "agent_instructions" in name
            or name.startswith(("build_results_", "test_results_"))
            or name == "original_concept_migrated_to_vraven.md"
        ):
            violations.append(path.relative_to(ROOT).as_posix())
        if ".cursor" in path.parts:
            violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []


def test_public_text_has_no_private_environment_markers() -> None:
    markers = (
        b"/Users/" + b"admin",
        b"/home/" + b"oai",
        b"/mnt/" + b"data",
        b"applied-" + b"caas-gateway",
        b"github_" + b"pat_",
        b"ghp" + b"_",
        b"github-" + b"pat2",
        b"VRAVENAI_" + b"TOKEN",
    )
    validator = ROOT / "scripts" / "check_docs_site.py"
    violations = []
    for path in _public_files():
        if path == validator:
            continue
        content = path.read_bytes()
        for marker in markers:
            if marker in content:
                violations.append(f"{path.relative_to(ROOT)}: {marker!r}")
    assert violations == []


def test_notebooks_are_clean_source_documents() -> None:
    violations = []
    for path in sorted((ROOT / "examples" / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                violations.append(f"{path.name}: code cell {index} contains execution state")
    assert violations == []


def test_public_version_metadata_is_consistent() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package = (ROOT / "src" / "vraven" / "_version.py").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    version_match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert version_match is not None
    version = version_match.group(1)
    assert f'__version__ = "{version}"' in package
    assert f"version: {version}" in citation
    assert "python -m pip install vraven" in readme
    assert f"version = {{{version}}}" in readme


def test_package_does_not_claim_unverified_pep561_compliance() -> None:
    assert not (ROOT / "src" / "vraven" / "py.typed").exists()


def test_generated_reports_have_no_remote_runtime_dependencies() -> None:
    generators = (
        ROOT / "src" / "vraven" / "report" / "html.py",
        ROOT / "scripts" / "build_vraven_gallery.py",
        ROOT / "scripts" / "polish_vraven_html.py",
    )
    forbidden = (
        "@import url(",
        "cdn.jsdelivr.net",
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "unpkg.com",
    )
    violations = []
    for path in generators:
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in source:
                violations.append(f"{path.relative_to(ROOT)}: {marker}")
    assert violations == []
    assert not (ROOT / "src" / "vraven" / "visualisation" / "scichart_export.py").exists()


def test_examples_do_not_disable_restricted_checkpoint_loading() -> None:
    violations = []
    for directory in (ROOT / "examples", ROOT / "scripts"):
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ipynb"}:
                continue
            if "weights_only=False" in path.read_text(encoding="utf-8"):
                violations.append(path.relative_to(ROOT).as_posix())
    assert violations == []
