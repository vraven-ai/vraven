from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vraven.figures.registry import FIGURES

MODES = ("publication_light", "interactive_dark")
FORMATS = ("svg", "png", "pdf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    files: list[dict] = []

    for key, spec in FIGURES.items():
        for mode in MODES:
            base = ROOT / "exports" / mode / spec.category
            paths = {
                "svg": base / "svg" / f"{key}_{mode}.svg",
                "png": base / "png" / f"{key}_{mode}.png",
                "pdf": base / "pdf" / f"{key}_{mode}.pdf",
            }
            for _fmt, path in paths.items():
                if not path.exists():
                    errors.append(f"missing {path.relative_to(ROOT)}")
                    continue
                if path.stat().st_size == 0:
                    errors.append(f"empty {path.relative_to(ROOT)}")
                    continue
                files.append({
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })

            if paths["svg"].exists():
                try:
                    tree = ET.parse(paths["svg"])
                    root = tree.getroot()
                    titles = root.findall("{http://www.w3.org/2000/svg}title")
                    if titles:
                        errors.append(f"internal SVG heading/title found in {paths['svg'].name}")
                    texts = root.findall(".//{http://www.w3.org/2000/svg}text")
                    if not texts:
                        errors.append(f"no labels in {paths['svg'].name}")
                except ET.ParseError as exc:
                    errors.append(f"invalid SVG {paths['svg'].name}: {exc}")

            if paths["png"].exists():
                try:
                    with Image.open(paths["png"]) as image:
                        image.verify()
                    with Image.open(paths["png"]) as image:
                        if image.width < 2000 or image.height < 1200:
                            warnings.append(f"PNG resolution below preferred threshold: {paths['png'].name} {image.size}")
                        corner = image.convert("RGB").getpixel((2, 2))
                        if mode == "publication_light" and min(corner) < 245:
                            errors.append(f"publication canvas is not white in {paths['png'].name}: {corner}")
                        if mode == "interactive_dark" and max(corner) > 30:
                            errors.append(f"dark canvas is not dark in {paths['png'].name}: {corner}")
                except Exception as exc:
                    errors.append(f"invalid PNG {paths['png'].name}: {exc}")

            if paths["pdf"].exists():
                try:
                    reader = PdfReader(str(paths["pdf"]))
                    if len(reader.pages) != 1:
                        errors.append(f"PDF is not single-page: {paths['pdf'].name}")
                    mediabox = reader.pages[0].mediabox
                    if float(mediabox.width) <= 0 or float(mediabox.height) <= 0:
                        errors.append(f"invalid PDF media box: {paths['pdf'].name}")
                except Exception as exc:
                    errors.append(f"invalid PDF {paths['pdf'].name}: {exc}")

    font_files = [path for path in ROOT.rglob("*") if path.suffix.lower() in {".ttf", ".otf", ".woff", ".woff2"}]
    if font_files:
        errors.append("font files must not be included: " + ", ".join(str(p.relative_to(ROOT)) for p in font_files))

    expected = len(FIGURES) * len(MODES) * len(FORMATS)
    if len(files) != expected:
        errors.append(f"expected {expected} output files, found {len(files)}")

    report = {
        "figure_families": len(FIGURES),
        "modes": list(MODES),
        "formats": list(FORMATS),
        "expected_outputs": expected,
        "validated_outputs": len(files),
        "errors": errors,
        "warnings": warnings,
        "files": files,
    }
    (ROOT / "VALIDATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Validation report",
        "",
        f"* Figure families: {len(FIGURES)}",
        f"* Render modes: {len(MODES)}",
        f"* Formats per mode: {len(FORMATS)}",
        f"* Expected exported files: {expected}",
        f"* Validated exported files: {len(files)}",
        f"* Errors: {len(errors)}",
        f"* Warnings: {len(warnings)}",
        "",
    ]
    if errors:
        lines += ["## Errors", ""] + [f"* {item}" for item in errors] + [""]
    if warnings:
        lines += ["## Warnings", ""] + [f"* {item}" for item in warnings] + [""]
    if not errors:
        lines += [
            "All SVG files parsed successfully, all PNG files opened successfully, all PDFs contained one valid page, both render-mode canvas checks passed and no font files were included.",
            "",
        ]
    (ROOT / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = validate()
    if result["errors"]:
        for error in result["errors"]:
            print("ERROR:", error)
        raise SystemExit(1)
    print(f"Validated {result['validated_outputs']} exported files with {len(result['warnings'])} warning(s).")
