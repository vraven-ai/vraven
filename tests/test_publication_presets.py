from __future__ import annotations

from vraven.visualisation.export import PUBLICATION_PRESETS


def test_publication_preset_pixel_conversion() -> None:
    width, height = PUBLICATION_PRESETS["single_column"].pixels(300)
    assert width > 1000
    assert height > 700
