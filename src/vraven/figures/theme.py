from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    canvas: str
    panel: str
    panel_alt: str
    border: str
    grid: str
    text: str
    text_secondary: str
    support: str
    support_strong: str
    oppose: str
    oppose_strong: str
    uncertainty: str
    stable: str
    latent: str
    neutral: str
    unknown: str
    selected: str
    white: str = "#FFFFFF"
    black: str = "#000000"


PUBLICATION_LIGHT = Theme(
    name="publication_light",
    canvas="#FFFFFF",
    panel="#FFFFFF",
    panel_alt="#F8FAFC",
    border="#CBD5E1",
    grid="#DCE3EC",
    text="#111827",
    text_secondary="#475569",
    support="#0057B8",
    support_strong="#0284C7",
    oppose="#B42318",
    oppose_strong="#9D174D",
    uncertainty="#C2410C",
    stable="#15803D",
    latent="#6D28D9",
    neutral="#64748B",
    unknown="#475569",
    selected="#0E7490",
)

INTERACTIVE_DARK = Theme(
    name="interactive_dark",
    canvas="#030A12",
    panel="#07111F",
    panel_alt="#0B1728",
    border="#1C344D",
    grid="#15263A",
    text="#F1F5F9",
    text_secondary="#94A3B8",
    support="#168BFF",
    support_strong="#00D4FF",
    oppose="#FF2D55",
    oppose_strong="#E83E8C",
    uncertainty="#FF8A00",
    stable="#32D74B",
    latent="#8B5CF6",
    neutral="#94A3B8",
    unknown="#94A3B8",
    selected="#00D4FF",
)


def get_theme(name: str) -> Theme:
    if name == "publication_light":
        return PUBLICATION_LIGHT
    if name == "interactive_dark":
        return INTERACTIVE_DARK
    raise ValueError(f"Unknown theme: {name}")
