from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vraven.figures.registry import FIGURES
from vraven.figures.theme import get_theme


@pytest.mark.parametrize("mode", ["publication_light", "interactive_dark"])
@pytest.mark.parametrize("key", list(FIGURES))
def test_every_figure_renders_as_svg(key: str, mode: str, tmp_path: Path) -> None:
    canvas = FIGURES[key].renderer(get_theme(mode))
    output = tmp_path / f"{key}_{mode}.svg"
    canvas.save(str(output))
    text = output.read_text(encoding="utf-8")
    assert output.stat().st_size > 1000
    assert "<svg" in text
    assert "<title" not in text
    assert canvas.theme.name == mode


def test_expected_figure_count() -> None:
    assert len(FIGURES) == 17
