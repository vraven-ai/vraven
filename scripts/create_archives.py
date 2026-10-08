from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)


def archive(name: str, source: Path) -> Path:
    target = DIST / name
    if target.exists():
        target.unlink()
    base = target.with_suffix("")
    created = shutil.make_archive(str(base), "zip", root_dir=source.parent, base_dir=source.name)
    return Path(created)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    staging = ROOT / "_archive_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    exports_pkg = staging / "VRAVEN_Scientific_Diagrams_Exports"
    sources_pkg = staging / "VRAVEN_Scientific_Diagrams_Sources"
    complete_pkg = staging / "VRAVEN_Scientific_Diagrams_Complete"

    exports_pkg.mkdir()
    shutil.copytree(ROOT / "exports", exports_pkg / "exports")
    shutil.copytree(ROOT / "docs", exports_pkg / "docs")
    shutil.copy(ROOT / "README.md", exports_pkg / "README.md")
    shutil.copy(ROOT / "VALIDATION_REPORT.md", exports_pkg / "VALIDATION_REPORT.md")
    shutil.copy(
        ROOT / "docs" / "references" / "VRAVEN_REFERENCES.bib",
        exports_pkg / "REFERENCES.bib",
    )
    shutil.copy(ROOT / "FIGURE_INDEX.csv", exports_pkg / "FIGURE_INDEX.csv")
    shutil.copy(ROOT / "GALLERY.html", exports_pkg / "GALLERY.html")
    shutil.copytree(ROOT / "preview", exports_pkg / "preview")

    sources_pkg.mkdir()
    ignore_runtime = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for item in ["src", "configs", "docs", "source_material", "scripts", "tests"]:
        shutil.copytree(ROOT / item, sources_pkg / item, ignore=ignore_runtime)
    source_files = {
        "README.md": ROOT / "README.md",
        "render_all.py": ROOT / "scripts" / "figures" / "render_all.py",
        "render_one.py": ROOT / "scripts" / "figures" / "render_one.py",
        "requirements.txt": ROOT / "scripts" / "figures" / "requirements.txt",
        "pyproject.toml": ROOT / "pyproject.toml",
        "REFERENCES.bib": ROOT / "docs" / "references" / "VRAVEN_REFERENCES.bib",
        "LICENSE": ROOT / "LICENSE",
        "CITATION.cff": ROOT / "CITATION.cff",
        "VALIDATION_REPORT.md": ROOT / "VALIDATION_REPORT.md",
        "VALIDATION.json": ROOT / "VALIDATION.json",
        "FIGURE_INDEX.csv": ROOT / "FIGURE_INDEX.csv",
    }
    for destination, source in source_files.items():
        shutil.copy(source, sources_pkg / destination)
    # Include editable vector exports with the source bundle.
    for mode in ["publication_light", "interactive_dark"]:
        for category in [p.name for p in (ROOT / "exports" / mode).iterdir() if p.is_dir()]:
            src = ROOT / "exports" / mode / category / "svg"
            dst = sources_pkg / "editable_svg" / mode / category
            shutil.copytree(src, dst)

    shutil.copytree(ROOT, complete_pkg, ignore=shutil.ignore_patterns("dist", "_archive_staging", "__pycache__", "*.pyc"))

    outputs = [
        archive("VRAVEN_Scientific_Diagrams_Exports.zip", exports_pkg),
        archive("VRAVEN_Scientific_Diagrams_Sources.zip", sources_pkg),
        archive("VRAVEN_Scientific_Diagrams_Complete.zip", complete_pkg),
    ]

    checksum_lines = [f"{sha256(path)}  {path.name}" for path in outputs]
    (DIST / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    shutil.rmtree(staging)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
