from __future__ import annotations

from pathlib import Path


def test_every_runtime_subpackage_contains_real_source() -> None:
    root = Path(__file__).parents[1] / "src" / "vraven"
    expected = {
        "adapters",
        "analysis",
        "capture",
        "causal",
        "core",
        "metrics",
        "reasoning",
        "report",
        "visualisation",
    }
    assert expected <= {path.name for path in root.iterdir() if path.is_dir()}
    for name in expected:
        directory = root / name
        assert (directory / "__init__.py").stat().st_size > 0
        modules = [path for path in directory.glob("*.py") if path.name != "__init__.py"]
        assert modules, f"{name} has no implementation modules"
        assert all(path.stat().st_size > 100 for path in modules)


def test_runtime_source_has_no_scaffold_markers() -> None:
    root = Path(__file__).parents[1] / "src" / "vraven"
    forbidden = ("NotImplementedError", "TODO", "FIXME", "placeholder implementation")
    findings: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                findings.append(f"{path.relative_to(root)}: {token}")
    assert findings == []


def test_inventory_document_matches_runtime_source_tree() -> None:
    root = Path(__file__).parents[1]
    inventory = (root / "docs" / "CODE_INVENTORY.md").read_text(encoding="utf-8")
    source_files = sorted((root / "src" / "vraven").rglob("*.py"))
    assert f"* Python modules: `{len(source_files)}`" in inventory
    for path in source_files:
        relative = path.relative_to(root).as_posix()
        assert f"`{relative}`" in inventory
