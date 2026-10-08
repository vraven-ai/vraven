"""Two-mode visual theme shared by every standalone VRAVEN figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import plotly.graph_objects as go
import plotly.io as plotly_io

from vraven.visualisation.palette import (
    SEMANTIC_COLOURS,
    VRAVEN_COLORS,
    VRAVEN_IMPACT,
    VRAVEN_IMPACT_LIGHT,
    RenderMode,
)

_TEMPLATE_REGISTERED = False


@dataclass(frozen=True, slots=True)
class RenderTheme:
    """Concrete colour/typography tokens for one render mode."""

    mode: RenderMode
    canvas: str
    panel: str
    raised_panel: str
    border: str
    grid: str
    primary_text: str
    secondary_text: str
    font_family: str
    title_size: int
    label_size: int
    note_size: int
    default_width: int
    default_height: int
    hover_background: str
    hover_border: str
    glow_allowed: bool

    @property
    def semantic(self) -> dict[str, str]:
        return dict(SEMANTIC_COLOURS[self.mode])


THEMES: Final[dict[RenderMode, RenderTheme]] = {
    "interactive_dark": RenderTheme(
        mode="interactive_dark",
        canvas=VRAVEN_COLORS["background"],
        panel=VRAVEN_COLORS["panel"],
        raised_panel=VRAVEN_COLORS["panel_alt"],
        border=VRAVEN_COLORS["border"],
        grid="rgba(28, 52, 77, 0.55)",
        primary_text=VRAVEN_COLORS["text"],
        secondary_text=VRAVEN_COLORS["muted"],
        font_family="Inter, IBM Plex Sans, Source Sans 3, Arial, sans-serif",
        title_size=17,
        label_size=12,
        note_size=10,
        default_width=1100,
        default_height=650,
        hover_background=VRAVEN_COLORS["panel_alt"],
        hover_border=VRAVEN_COLORS["border"],
        glow_allowed=True,
    ),
    "publication_light": RenderTheme(
        mode="publication_light",
        canvas="#FFFFFF",
        panel="#FFFFFF",
        raised_panel="#F8FAFC",
        border="#CBD5E1",
        grid="#DCE3EC",
        primary_text="#111827",
        secondary_text="#475569",
        font_family="Arial, Helvetica, Liberation Sans, sans-serif",
        title_size=14,
        label_size=10,
        note_size=8,
        default_width=900,
        default_height=520,
        hover_background="#FFFFFF",
        hover_border="#94A3B8",
        glow_allowed=False,
    ),
}


def get_theme(mode: RenderMode = "publication_light") -> RenderTheme:
    try:
        return THEMES[mode]
    except KeyError as error:  # pragma: no cover
        supported = ", ".join(sorted(THEMES))
        raise ValueError(f"Unsupported VRAVEN render mode: {mode}. Choose from {supported}.") from error


def _build_plotly_template(mode: RenderMode) -> go.layout.Template:
    theme = get_theme(mode)
    colours = theme.semantic
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=theme.canvas,
            plot_bgcolor=theme.panel,
            font={"family": theme.font_family, "size": theme.label_size, "color": theme.primary_text},
            colorway=[
                colours["opposing"],
                colours["supporting"],
                colours["stable"],
                colours["uncertainty"],
                colours["latent"],
                colours["selected"],
                colours["neutral"],
            ],
            colorscale={
                "sequential": [
                    [0.0, theme.canvas],
                    [0.45, colours["latent"]],
                    [1.0, colours["uncertainty"]],
                ],
                "diverging": (
                    VRAVEN_IMPACT if mode == "interactive_dark" else VRAVEN_IMPACT_LIGHT
                ),
            },
            xaxis={
                "gridcolor": theme.grid,
                "zerolinecolor": theme.border,
                "linecolor": theme.border,
                "tickcolor": theme.border,
                "automargin": True,
            },
            yaxis={
                "gridcolor": theme.grid,
                "zerolinecolor": theme.border,
                "linecolor": theme.border,
                "tickcolor": theme.border,
                "automargin": True,
            },
            legend={
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "font": {"color": theme.primary_text},
            },
            hoverlabel={
                "bgcolor": theme.hover_background,
                "bordercolor": theme.hover_border,
                "font": {"color": theme.primary_text, "family": theme.font_family},
            },
        )
    )


def register_plotly_templates() -> None:
    """Register ``vraven`` Plotly templates once per process."""

    global _TEMPLATE_REGISTERED
    if _TEMPLATE_REGISTERED:
        return
    dark = _build_plotly_template("interactive_dark")
    light = _build_plotly_template("publication_light")
    plotly_io.templates["vraven"] = dark
    plotly_io.templates["vraven_dark"] = dark
    plotly_io.templates["vraven_light"] = light
    _TEMPLATE_REGISTERED = True


class VRavenTheme:
    """Library-level VRAVEN visual controller for scientific figures.

    Example::

        from vraven import VRavenTheme

        VRavenTheme.apply()
        figure.update_layout(template="vraven")
    """

    COLORS = VRAVEN_COLORS
    IMPACT = VRAVEN_IMPACT
    THEME_VERSION = "3.1"

    @classmethod
    def apply(cls, mode: RenderMode = "interactive_dark") -> str:
        """Register Plotly templates and optionally set the default template name."""

        register_plotly_templates()
        template_name = "vraven" if mode == "interactive_dark" else "vraven_light"
        plotly_io.templates.default = template_name
        return template_name

    @classmethod
    def template_name(cls, mode: RenderMode = "interactive_dark") -> str:
        register_plotly_templates()
        return "vraven" if mode == "interactive_dark" else "vraven_light"


def _paper_annotation_ceiling(figure: go.Figure) -> float:
    """Highest paper-y used by annotations that sit above the plot box."""

    ceiling = 1.0
    annotations = figure.layout.annotations or ()
    for annotation in annotations:
        yref = str(getattr(annotation, "yref", "paper") or "paper")
        if yref not in {"paper", "y domain"} and not yref.endswith("paper"):
            continue
        y_value = getattr(annotation, "y", None)
        if isinstance(y_value, (int, float)):
            ceiling = max(ceiling, float(y_value))
    return ceiling


def apply_vraven_theme(
    figure: go.Figure,
    *,
    mode: RenderMode = "publication_light",
    title: str | None = None,
    subtitle: str | None = None,
    width: int | None = None,
    height: int | None = None,
    show_legend: bool | None = None,
    show_title: bool | None = None,
) -> go.Figure:
    """Apply presentation only. Analysis values are never changed here."""

    register_plotly_templates()
    theme = get_theme(mode)
    if show_title is None:
        # Standalone figures must remain self-explanatory in both modes.  The
        # composed HTML report passes ``show_title=False`` because its section
        # heading and evidence panel own the title there.
        show_title = True
    candidate_title = title or figure.layout.title.text
    title_text = candidate_title if show_title else None
    has_subtitle = bool(title_text and subtitle)
    if has_subtitle:
        title_text = (
            f"{title_text}<br>"
            f"<span style='font-size:{theme.note_size}px;color:{theme.secondary_text}'>"
            f"{subtitle}</span>"
        )

    existing_meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    meta = dict(existing_meta)
    vraven_meta = dict(meta.get("vraven", {}))
    vraven_meta.update(
        {
            "render_mode": mode,
            "theme_version": VRavenTheme.THEME_VERSION,
            "theme_family": "vraven",
            "standalone": True,
        }
    )
    meta["vraven"] = vraven_meta

    legend_visible = True if show_legend is None else bool(show_legend)
    annotation_ceiling = _paper_annotation_ceiling(figure)
    has_overhead_labels = annotation_ceiling > 1.0 or any(
        str(getattr(annotation, "yref", "paper") or "paper").endswith("paper")
        and isinstance(getattr(annotation, "y", None), (int, float))
        and float(annotation.y) >= 0.995
        for annotation in (figure.layout.annotations or ())
    )
    # Hard pixel budget: title band, badge band, and plot never share the same
    # vertical slice. Legend lives below the plot only (never over series ink).
    #
    # Do NOT pin titles with yref="container" + yanchor="top" — that clips the
    # "VRAVEN" brand line against the figure edge. Use Plotly's default title
    # slot inside an explicit top margin (no custom y/yref).
    if title_text and has_overhead_labels:
        top_margin = 236 if has_subtitle else 208
        title_bottom_pad = 48
    elif title_text:
        top_margin = 168 if has_subtitle else 132
        title_bottom_pad = 20
    elif has_overhead_labels:
        top_margin = 96
        title_bottom_pad = 0
    else:
        top_margin = 48
        title_bottom_pad = 0
    bottom_margin = 148 if legend_visible else 80
    left_margin = 96
    right_margin = 56
    template_name = VRavenTheme.template_name(mode)
    figure.update_layout(
        template=template_name,
        paper_bgcolor=theme.canvas,
        plot_bgcolor=theme.panel,
        font={"family": theme.font_family, "size": theme.label_size, "color": theme.primary_text},
        title=(
            {
                "text": title_text,
                "x": 0.0,
                "xanchor": "left",
                "pad": {"t": 18, "b": title_bottom_pad, "l": 2, "r": 0},
                "font": {"size": theme.title_size, "color": theme.primary_text},
                "automargin": True,
            }
            if title_text
            else None
        ),
        width=width or theme.default_width,
        height=height or theme.default_height,
        margin={"l": left_margin, "r": right_margin, "t": top_margin, "b": bottom_margin},
        colorway=[
            theme.semantic["opposing"],
            theme.semantic["supporting"],
            theme.semantic["stable"],
            theme.semantic["uncertainty"],
            theme.semantic["latent"],
            theme.semantic["selected"],
            theme.semantic["neutral"],
        ],
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": theme.border,
            "borderwidth": 0,
            "font": {"color": theme.primary_text, "size": theme.label_size},
            "orientation": "h",
            "x": 0.0,
            "y": -0.22,
            "xanchor": "left",
            "yanchor": "top",
            "traceorder": "normal",
            "itemwidth": 42,
            "tracegroupgap": 10,
            "itemsizing": "constant",
            "valign": "top",
        },
        hoverlabel={
            "bgcolor": theme.hover_background,
            "bordercolor": theme.hover_border,
            "font": {"color": theme.primary_text, "family": theme.font_family},
        },
        meta=meta,
        uirevision="vraven",
    )
    if show_legend is not None:
        figure.update_layout(showlegend=show_legend)

    axis_style = {
        "showgrid": True,
        "gridcolor": theme.grid,
        "gridwidth": 0.6,
        "zeroline": True,
        "zerolinecolor": theme.border,
        "zerolinewidth": 0.8,
        "linecolor": theme.border,
        "linewidth": 0.8,
        "mirror": False,
        "ticks": "outside",
        "tickcolor": theme.border,
        "tickfont": {"color": theme.secondary_text, "size": theme.label_size},
        "title_font": {"color": theme.primary_text, "size": theme.label_size},
        "automargin": True,
        "title_standoff": 22,
        "separatethousands": False,
    }
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)
    figure.update_layout(hovermode=False if mode == "publication_light" else "closest")
    return figure
