from __future__ import annotations

import plotly.io as plotly_io

from vraven.visualisation.palette import (
    SEMANTIC_COLOURS,
    VRAVEN_COLORS,
    VRAVEN_IMPACT,
    semantic_colour,
    signed_colour_scale,
)
from vraven.visualisation.theme import VRavenTheme, get_theme


def test_publication_theme_is_white_and_has_no_glow() -> None:
    theme = get_theme("publication_light")
    assert theme.canvas == "#FFFFFF"
    assert theme.panel == "#FFFFFF"
    assert theme.glow_allowed is False


def test_interactive_theme_uses_vraven_navy() -> None:
    theme = get_theme("interactive_dark")
    assert theme.canvas == VRAVEN_COLORS["background"]
    assert theme.panel == VRAVEN_COLORS["panel"]
    assert theme.raised_panel == "#0B1728"
    assert theme.border == "#1C344D"
    assert theme.primary_text == VRAVEN_COLORS["text"]
    assert theme.glow_allowed is True


def test_semantic_roles_exist_in_both_modes() -> None:
    assert set(SEMANTIC_COLOURS["interactive_dark"]) == set(
        SEMANTIC_COLOURS["publication_light"]
    )
    assert semantic_colour("supporting", "interactive_dark") != semantic_colour(
        "supporting", "publication_light"
    )


def test_vraven_semantic_meaning() -> None:
    dark = SEMANTIC_COLOURS["interactive_dark"]
    assert dark["opposing"] == VRAVEN_COLORS["red"]
    assert dark["contradiction"] == VRAVEN_COLORS["magenta"]
    assert dark["supporting"] == VRAVEN_COLORS["blue"]
    assert dark["strong_supporting"] == VRAVEN_COLORS["cyan"]
    assert dark["stable"] == VRAVEN_COLORS["green"]
    assert dark["uncertainty"] == VRAVEN_COLORS["orange"]
    assert dark["latent"] == VRAVEN_COLORS["purple"]


def test_signed_scale_matches_vraven_impact() -> None:
    assert signed_colour_scale("interactive_dark") == VRAVEN_IMPACT
    assert signed_colour_scale("interactive_dark")[0][1] == "#0759C7"
    assert signed_colour_scale("interactive_dark")[-1][1] == "#FFD60A"


def test_vraven_theme_apply_registers_templates() -> None:
    name = VRavenTheme.apply()
    assert name == "vraven"
    assert "vraven" in plotly_io.templates
    assert "vraven_light" in plotly_io.templates
    assert plotly_io.templates.default == "vraven"
