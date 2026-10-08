from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "create_release.py"
SPEC = importlib.util.spec_from_file_location("vraven_create_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_release_selection_excludes_generated_and_build_files() -> None:
    root = Path(__file__).parents[1]
    assert MODULE.should_include(root / "src" / "vraven" / "__init__.py")
    assert not MODULE.should_include(root / "dist" / "vraven.whl")
    assert not MODULE.should_include(root / "release" / "VRAVEN-source.zip")
    assert not MODULE.should_include(root / ".coverage")
    assert not MODULE.should_include(root / "htmlcov" / "index.html")
    assert not MODULE.should_include(root / "src" / "vraven" / "__pycache__" / "x.pyc")
    assert not MODULE.should_include(root / "vraven_demo" / "report.html")


def test_release_runtime_validation_and_version() -> None:
    MODULE.validate_runtime_source()
    assert MODULE.project_version() == "0.10.1"


def test_release_file_list_is_limited_to_version_controlled_files() -> None:
    root = SCRIPT.parents[1]
    files = MODULE.source_files()

    assert root / "pyproject.toml" in files
    assert all(path.is_file() and path.is_relative_to(root) for path in files)
    assert not any(part.startswith(".venv") for path in files for part in path.parts)
    MODULE.validate_public_file_names(files)


def test_release_requires_a_clean_committed_tree() -> None:
    clean = Mock(stdout="")
    with patch.object(MODULE.subprocess, "run", return_value=clean):
        MODULE.validate_clean_git_tree()

    dirty = Mock(stdout="?? src/vraven/_version.py\n")
    with (
        patch.object(MODULE.subprocess, "run", return_value=dirty),
        pytest.raises(RuntimeError, match="uncommitted tree"),
    ):
        MODULE.validate_clean_git_tree()
