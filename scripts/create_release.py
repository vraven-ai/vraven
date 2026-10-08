#!/usr/bin/env python3
"""Create a clean, reproducible VRAVEN source archive and checksum.

The archive intentionally excludes build products, interpreter caches and generated
reports. A SHA-256 manifest inside the archive records every included project file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "release",
    "vraven_attention_demo",
    "vraven_causal_demo",
    "vraven_dataset_demo",
    "vraven_demo",
    "vraven_quickstart",
    "vraven_visual_gallery",
    "vraven_visual_modes",
}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo", ".whl"}
EXCLUDED_FILE_NAMES = {".DS_Store", ".coverage", "smoke_report.html"}
FORBIDDEN_RUNTIME_MARKERS = ("NotImplementedError", "TODO", "FIXME")
FORBIDDEN_PUBLIC_FILENAMES = {
    "agents.md",
    "claude.md",
    "copilot-instructions.md",
}
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def project_version() -> str:
    """Read the PEP 621 project version without importing VRAVEN or PyTorch."""

    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def validate_clean_git_tree() -> None:
    """Require release archives to come from a committed, reproducible tree."""

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Refusing to create a source release from an uncommitted tree. "
            "Commit or intentionally remove every tracked and untracked change first."
        )


def should_include(path: Path) -> bool:
    """Return whether a project path belongs in a clean source release."""

    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRECTORY_NAMES or part.endswith(".egg-info") for part in relative.parts):
        return False
    return not (path.name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_FILE_SUFFIXES)


def source_files() -> list[Path]:
    """Return version-controlled release files in deterministic path order.

    Git is the release boundary. This prevents ignored local datasets, model
    checkpoints, virtual environments and research workspaces from entering an
    archive merely because they exist beside the source checkout.
    """

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = (
        ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    )
    return sorted(path for path in paths if path.is_file() and should_include(path))


def validate_runtime_source() -> None:
    """Fail when runtime Python modules are empty or contain scaffold markers."""

    runtime = ROOT / "src" / "vraven"
    files = sorted(runtime.rglob("*.py"))
    if not files:
        raise RuntimeError("VRAVEN runtime source is missing")
    failures: list[str] = []
    for path in files:
        content = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        if not content.strip():
            failures.append(f"{relative}: empty")
        for marker in FORBIDDEN_RUNTIME_MARKERS:
            if marker in content:
                failures.append(f"{relative}: contains {marker}")
        if any(line.strip() == "pass" for line in content.splitlines()):
            failures.append(f"{relative}: contains a pass-only statement")
    if failures:
        raise RuntimeError("Runtime source validation failed:\n" + "\n".join(failures))


def validate_public_file_names(files: list[Path]) -> None:
    """Reject development-only instruction files from public archives."""

    failures = []
    for path in files:
        relative = path.relative_to(ROOT)
        name = path.name.lower()
        if (
            name in FORBIDDEN_PUBLIC_FILENAMES
            or "agent_instructions" in name
            or name.startswith(("build_results_", "test_results_"))
            or name == "original_concept_migrated_to_vraven.md"
            or ".cursor" in relative.parts
        ):
            failures.append(relative.as_posix())
    if failures:
        raise RuntimeError(
            "Public release contains development-only instruction files:\n"
            + "\n".join(failures)
        )


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(files: list[Path]) -> dict[str, object]:
    """Build a stable per-file hash manifest for release verification."""

    return {
        "project": "VRAVEN",
        "version": project_version(),
        "algorithm": "sha256",
        "files": {
            path.relative_to(ROOT).as_posix(): file_digest(path)
            for path in files
        },
    }


def copy_release_tree(destination: Path, files: list[Path]) -> None:
    """Copy selected files while preserving executable script permissions."""

    for source in files:
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_zip(source_root: Path, archive_path: Path) -> None:
    """Write a deterministic ZIP archive with normalised timestamps."""

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            arcname = path.relative_to(source_root.parent).as_posix()
            information = zipfile.ZipInfo(arcname, date_time=ZIP_TIMESTAMP)
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = (stat.S_IFREG | (path.stat().st_mode & 0o777)) << 16
            archive.writestr(information, path.read_bytes())


def archive_digest(path: Path) -> str:
    return file_digest(path)


def create_release(output_dir: Path) -> tuple[Path, Path]:
    """Create the clean source archive and its external checksum file."""

    validate_clean_git_tree()
    validate_runtime_source()
    files = source_files()
    validate_public_file_names(files)
    version = project_version()
    root_name = f"VRAVEN-v{version}-source"
    archive = output_dir / f"{root_name}.zip"
    checksum = output_dir / f"{root_name}.zip.sha256"

    with tempfile.TemporaryDirectory(prefix="vraven-release-") as directory:
        staging = Path(directory) / root_name
        copy_release_tree(staging, files)
        manifest = build_manifest(files)
        (staging / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_zip(staging, archive)

    checksum.write_text(
        f"{archive_digest(archive)}  {archive.name}\n",
        encoding="utf-8",
    )
    return archive, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "release",
        help="Directory for the ZIP archive and SHA-256 checksum",
    )
    args = parser.parse_args()
    archive, checksum = create_release(args.output_dir.resolve())
    print(f"archive={archive}")
    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
