#!/usr/bin/env python3
"""Merge only VRAVEN DarkSide extension files into an existing workspace.

The command performs a complete preflight before writing anything. Existing
DarkSide files are never overwritten unless ``--overwrite`` is supplied. When
overwriting, the displaced files are backed up inside the target repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _build_plan(extension: Path, target: Path) -> list[tuple[Path, Path, str]]:
    source_package = extension / "merge" / "src" / "vraven" / "darkside"
    if not source_package.is_dir():
        raise FileNotFoundError(f"Extension source was not found at {source_package}")
    if not (target / "src" / "vraven").is_dir():
        raise FileNotFoundError(f"Target does not look like a VRAVEN repository: {target}")

    plan: list[tuple[Path, Path, str]] = [
        (source_package, target / "src" / "vraven" / "darkside", "directory")
    ]
    for category, pattern in (
        ("tests", "test_darkside*.py"),
        ("examples", "darkside_*.py"),
    ):
        source_dir = extension / "merge" / category
        for source in sorted(source_dir.glob(pattern)):
            plan.append((source, target / category / source.name, "file"))
    return plan


def _preflight(plan: list[tuple[Path, Path, str]], *, overwrite: bool) -> None:
    missing = [str(source) for source, _, _ in plan if not source.exists()]
    if missing:
        raise FileNotFoundError("Missing extension files:\n" + "\n".join(missing))
    conflicts = [str(destination) for _, destination, _ in plan if destination.exists()]
    if conflicts and not overwrite:
        rendered = "\n".join(conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing DarkSide files. Review the conflicts and rerun "
            f"with --overwrite only after a backup:\n{rendered}"
        )


def _backup_conflicts(
    plan: list[tuple[Path, Path, str]],
    *,
    target: Path,
) -> Path | None:
    conflicts = [(destination, kind) for _, destination, kind in plan if destination.exists()]
    if not conflicts:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = target / ".vraven_darkside_backups" / stamp
    for destination, kind in conflicts:
        relative = destination.relative_to(target)
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if kind == "directory":
            shutil.copytree(destination, backup)
        else:
            shutil.copy2(destination, backup)
    return backup_root


def _install_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vraven-darkside-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staged, destination)


def _install_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        staged = Path(temporary.name)
    try:
        shutil.copy2(source, staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge the isolated VRAVEN DarkSide extension into an existing VRAVEN repository."
    )
    parser.add_argument("extension", type=Path, help="Root of VRAVEN_DarkSide_Extensions")
    parser.add_argument("--target", type=Path, default=Path.cwd(), help="Existing VRAVEN root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing DarkSide files after creating a timestamped backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the merge plan without changing the target.",
    )
    args = parser.parse_args()

    extension = args.extension.expanduser().resolve()
    target = args.target.expanduser().resolve()
    plan = _build_plan(extension, target)
    _preflight(plan, overwrite=args.overwrite)

    print("VRAVEN DarkSide merge plan")
    for source, destination, kind in plan:
        print(f"  {kind:9s} {source} -> {destination}")
    if args.dry_run:
        print("Dry run completed. No files were changed.")
        return 0

    backup_root = _backup_conflicts(plan, target=target) if args.overwrite else None
    for source, destination, kind in plan:
        if kind == "directory":
            _install_directory(source, destination)
        else:
            _install_file(source, destination)

    print(f"Merged VRAVEN DarkSide into {target}")
    if backup_root is not None:
        print(f"Previous DarkSide files were backed up to {backup_root}")
    print("Run: PYTHONPATH=src python -m pytest -q tests/test_darkside*.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
