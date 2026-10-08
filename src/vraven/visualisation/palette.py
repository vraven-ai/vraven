"""Canonical semantic colour tokens for VRAVEN figures.

Colours are never decorative. Every role has a fixed meaning across the library:

* RED / MAGENTA  – harmful signal, positive contribution, high risk
* ORANGE / YELLOW – uncertainty, warning, moderate risk
* GREEN          – benign, safe, strong reliability
* BLUE / CYAN    – negative contribution, contextual information
* PURPLE         – internal representations and attention
* DARK NAVY      – background and panels
* WHITE / GREY   – primary and secondary text
"""

from __future__ import annotations

from typing import Final, Literal

RenderMode = Literal["interactive_dark", "publication_light"]
SemanticRole = Literal[
    "supporting",
    "strong_supporting",
    "opposing",
    "contradiction",
    "uncertainty",
    "stable",
    "latent",
    "neutral",
    "unknown",
    "selected",
]

# VRAVEN brand tokens (interactive / studio surface).
VRAVEN_COLORS: Final[dict[str, str]] = {
    "background": "#030A12",
    "panel": "#07111F",
    "panel_alt": "#0B1728",
    "border": "#1C344D",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "blue": "#168BFF",
    "cyan": "#00D4FF",
    "purple": "#8B5CF6",
    "magenta": "#E83E8C",
    "red": "#FF2D55",
    "orange": "#FF8A00",
    "yellow": "#FFD60A",
    "green": "#32D74B",
    "lime": "#A3E635",
}

# Continuous signed impact scale used by heatmaps and attribution fields.
# 0 = strong negative contribution, 0.5 = neutral, 1 = strongest positive / risk.
VRAVEN_IMPACT: Final[list[list[object]]] = [
    [0.00, "#0759C7"],
    [0.20, "#00B8FF"],
    [0.40, "#6D28D9"],
    [0.50, "#111827"],
    [0.65, "#E11D73"],
    [0.82, "#FF453A"],
    [1.00, "#FFD60A"],
]

# Print-safe impact scale with the same semantic endpoints.
VRAVEN_IMPACT_LIGHT: Final[list[list[object]]] = [
    [0.00, "#0057B8"],
    [0.20, "#0284C7"],
    [0.40, "#6D28D9"],
    [0.50, "#F8FAFC"],
    [0.65, "#BE185D"],
    [0.82, "#B42318"],
    [1.00, "#A15C00"],
]

# Role mapping:
# supporting / strong_supporting = negative contribution / contextual (blue/cyan)
# opposing / contradiction       = positive contribution / harmful (red/magenta)
# stable                         = benign / reliable (green)
# uncertainty                    = warning (orange/yellow)
# latent                         = representation / attention (purple)
# selected                       = focus accent (cyan)
SEMANTIC_COLOURS: Final[dict[RenderMode, dict[SemanticRole, str]]] = {
    "interactive_dark": {
        "supporting": VRAVEN_COLORS["blue"],
        "strong_supporting": VRAVEN_COLORS["cyan"],
        "opposing": VRAVEN_COLORS["red"],
        "contradiction": VRAVEN_COLORS["magenta"],
        "uncertainty": VRAVEN_COLORS["orange"],
        "stable": VRAVEN_COLORS["green"],
        "latent": VRAVEN_COLORS["purple"],
        "neutral": VRAVEN_COLORS["muted"],
        "unknown": "#64748B",
        "selected": VRAVEN_COLORS["cyan"],
    },
    "publication_light": {
        "supporting": "#0057B8",
        "strong_supporting": "#0284C7",
        "opposing": "#B42318",
        "contradiction": "#9D174D",
        "uncertainty": "#C2410C",
        "stable": "#15803D",
        "latent": "#6D28D9",
        "neutral": "#64748B",
        "unknown": "#475569",
        "selected": "#0E7490",
    },
}

# Class identity palette: prefer VRAVEN semantic hues before fallbacks.
CLASS_COLOURS: Final[tuple[str, ...]] = (
    VRAVEN_COLORS["green"],   # often benign / class 0
    VRAVEN_COLORS["red"],     # often malware / class 1
    VRAVEN_COLORS["blue"],
    VRAVEN_COLORS["purple"],
    VRAVEN_COLORS["orange"],
    VRAVEN_COLORS["cyan"],
    VRAVEN_COLORS["magenta"],
    VRAVEN_COLORS["lime"],
)

CLASS_SYMBOLS: Final[tuple[str, ...]] = (
    "circle",
    "square",
    "diamond",
    "cross",
    "triangle-up",
    "triangle-down",
    "x",
    "star",
)

SEMANTIC_MEANING: Final[dict[str, str]] = {
    "red": "harmful, positive contribution, high risk",
    "magenta": "harmful, positive contribution, high risk",
    "orange": "uncertainty, warning, moderate risk",
    "yellow": "uncertainty, warning, moderate risk",
    "green": "benign, safe, strong reliability",
    "blue": "negative contribution, contextual information",
    "cyan": "negative contribution, contextual information",
    "purple": "internal representations and attention",
    "background": "canvas and panels",
    "text": "primary labels and values",
    "muted": "secondary information",
}


def semantic_colour(role: SemanticRole, mode: RenderMode) -> str:
    return SEMANTIC_COLOURS[mode][role]


def signed_colour_scale(mode: RenderMode) -> list[list[object]]:
    """Diverging impact scale: blue/cyan (negative) → dark neutral → red/yellow (positive)."""

    if mode == "interactive_dark":
        return [list(stop) for stop in VRAVEN_IMPACT]
    return [list(stop) for stop in VRAVEN_IMPACT_LIGHT]


def sequential_colour_scale(mode: RenderMode) -> list[list[object]]:
    colours = SEMANTIC_COLOURS[mode]
    low = VRAVEN_COLORS["background"] if mode == "interactive_dark" else "#F8FAFC"
    return [
        [0.0, low],
        [0.35, colours["latent"]],
        [0.70, colours["uncertainty"]],
        [1.0, colours["opposing"]],
    ]


def relevance_colour_scale(mode: RenderMode) -> list[list[object]]:
    """Non-negative relevance / magnitude scale (dark → purple → amber)."""

    if mode == "interactive_dark":
        return [
            [0.0, VRAVEN_COLORS["background"]],
            [0.35, VRAVEN_COLORS["purple"]],
            [0.70, VRAVEN_COLORS["orange"]],
            [1.0, VRAVEN_COLORS["yellow"]],
        ]
    return [
        [0.0, "#F8FAFC"],
        [0.35, "#6D28D9"],
        [0.70, "#C2410C"],
        [1.0, "#A15C00"],
    ]


SUPPORTING = SEMANTIC_COLOURS["publication_light"]["supporting"]
STRONG_SUPPORTING = SEMANTIC_COLOURS["publication_light"]["strong_supporting"]
SELECTED = SEMANTIC_COLOURS["publication_light"]["selected"]
OPPOSING = SEMANTIC_COLOURS["publication_light"]["opposing"]
CONTRADICTION = SEMANTIC_COLOURS["publication_light"]["contradiction"]
UNCERTAINTY = SEMANTIC_COLOURS["publication_light"]["uncertainty"]
STABLE = SEMANTIC_COLOURS["publication_light"]["stable"]
LATENT = SEMANTIC_COLOURS["publication_light"]["latent"]
NEUTRAL = SEMANTIC_COLOURS["publication_light"]["neutral"]
UNKNOWN = SEMANTIC_COLOURS["publication_light"]["unknown"]
BACKGROUND_LIGHT = "#FFFFFF"
TEXT_LIGHT = "#111827"
