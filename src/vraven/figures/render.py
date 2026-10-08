from __future__ import annotations

import argparse
import json
from pathlib import Path

import cairosvg

from .registry import FIGURES, get_figure, list_figures
from .theme import get_theme

MODES = ("publication_light", "interactive_dark")
FORMATS = ("svg", "png", "pdf")


def render_figure(
    key: str,
    mode: str,
    output_root: Path,
    formats: tuple[str, ...] = FORMATS,
    png_scale: float = 2.0,
) -> dict[str, str]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    unsupported = set(formats) - set(FORMATS)
    if unsupported:
        raise ValueError(f"Unsupported formats: {sorted(unsupported)}")

    spec = get_figure(key)
    theme = get_theme(mode)
    canvas = spec.renderer(theme)

    svg_dir = output_root / mode / spec.category / "svg"
    png_dir = output_root / mode / spec.category / "png"
    pdf_dir = output_root / mode / spec.category / "pdf"
    for directory in (svg_dir, png_dir, pdf_dir):
        directory.mkdir(parents=True, exist_ok=True)

    svg_path = svg_dir / f"{key}_{mode}.svg"
    canvas.save(str(svg_path))

    result: dict[str, str] = {"svg": str(svg_path)}
    svg_bytes = svg_path.read_bytes()

    if "png" in formats:
        png_path = png_dir / f"{key}_{mode}.png"
        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=str(png_path),
            output_width=int(canvas.width * png_scale),
            output_height=int(canvas.height * png_scale),
        )
        result["png"] = str(png_path)

    if "pdf" in formats:
        pdf_path = pdf_dir / f"{key}_{mode}.pdf"
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(pdf_path))
        result["pdf"] = str(pdf_path)

    return result


def render_all(output_root: Path, modes: tuple[str, ...] = MODES, formats: tuple[str, ...] = FORMATS, png_scale: float = 2.0) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for key in FIGURES:
        for mode in modes:
            outputs.append(render_figure(key, mode, output_root, formats=formats, png_scale=png_scale))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render editable VRAVEN scientific diagrams.")
    parser.add_argument("--figure", default="all", help="Figure key or 'all'.")
    parser.add_argument("--mode", default="all", choices=("all",) + MODES)
    parser.add_argument("--formats", default="svg,png,pdf", help="Comma-separated: svg,png,pdf")
    parser.add_argument("--output", default="exports", help="Output root directory")
    parser.add_argument("--png-scale", type=float, default=2.0, help="PNG scale relative to SVG viewBox")
    parser.add_argument("--list", action="store_true", help="List available figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for spec in list_figures():
            print(f"{spec.key}\t{spec.category}\t{spec.caption_short}")
        return 0

    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    modes = MODES if args.mode == "all" else (args.mode,)
    output = Path(args.output).resolve()

    if args.figure == "all":
        results = render_all(output, modes=modes, formats=formats, png_scale=args.png_scale)
    else:
        results = [render_figure(args.figure, mode, output, formats=formats, png_scale=args.png_scale) for mode in modes]

    manifest = output / "render_manifest.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Rendered {len(results)} figure-mode combinations into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
