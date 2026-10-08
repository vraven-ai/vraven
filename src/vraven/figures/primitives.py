from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import svgwrite

from .theme import Theme

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"


def normalise_weight(weight: int | str) -> str:
    if isinstance(weight, str):
        if weight.isdigit():
            weight = int(weight)
        else:
            return weight
    if weight < 500:
        return "400"
    if weight < 700:
        return "600"
    return "700"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = hex_to_rgb(c1)
    b = hex_to_rgb(c2)
    rgb = tuple(round((1 - t) * x + t * y) for x, y in zip(a, b, strict=False))
    return "#" + "".join(f"{v:02X}" for v in rgb)


def alpha(theme: Theme, colour: str, amount: float) -> str:
    return mix(theme.canvas, colour, amount)


@dataclass
class Canvas:
    width: int
    height: int
    theme: Theme
    filename: str | None = None

    def __post_init__(self) -> None:
        self.dwg = svgwrite.Drawing(
            filename=self.filename,
            size=(self.width, self.height),
            viewBox=f"0 0 {self.width} {self.height}",
            profile="full",
            debug=False,
        )
        self.dwg.attribs["shape-rendering"] = "geometricPrecision"
        self._define_markers()
        self._define_filters()
        self.background()

    def _define_markers(self) -> None:
        for name, colour in (
            ("arrow", self.theme.text_secondary),
            ("arrow_support", self.theme.support),
            ("arrow_selected", self.theme.selected),
            ("arrow_oppose", self.theme.oppose),
            ("arrow_stable", self.theme.stable),
            ("arrow_uncertainty", self.theme.uncertainty),
            ("arrow_latent", self.theme.latent),
        ):
            marker = self.dwg.marker(
                id=name,
                insert=(10, 5),
                size=(12, 12),
                orient="auto",
                markerUnits="strokeWidth",
            )
            marker.add(self.dwg.path(d="M 0 0 L 10 5 L 0 10 z", fill=colour))
            self.dwg.defs.add(marker)

    def _define_filters(self) -> None:
        if self.theme.name != "interactive_dark":
            return
        glow = self.dwg.filter(id="soft_glow", x="-30%", y="-30%", width="160%", height="160%")
        glow.feGaussianBlur(in_="SourceGraphic", stdDeviation=4, result="blur")
        glow.feMerge(["blur", "SourceGraphic"])
        self.dwg.defs.add(glow)

    def background(self) -> None:
        self.dwg.add(self.dwg.rect(insert=(0, 0), size=(self.width, self.height), fill=self.theme.canvas))
        if self.theme.name == "interactive_dark":
            # restrained grid to give depth without decorative noise
            for x in range(0, self.width + 1, 80):
                self.dwg.add(self.dwg.line((x, 0), (x, self.height), stroke=self.theme.grid, stroke_width=0.5, opacity=0.20))
            for y in range(0, self.height + 1, 80):
                self.dwg.add(self.dwg.line((0, y), (self.width, y), stroke=self.theme.grid, stroke_width=0.5, opacity=0.20))

    def group(self, **kwargs):
        g = self.dwg.g(**kwargs)
        self.dwg.add(g)
        return g

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.5,
        radius: float = 14,
        opacity: float = 1.0,
        dash: str | None = None,
        parent=None,
    ):
        item = self.dwg.rect(
            insert=(x, y),
            size=(w, h),
            rx=radius,
            ry=radius,
            fill=fill or self.theme.panel,
            stroke=stroke or self.theme.border,
            stroke_width=stroke_width,
            opacity=opacity,
        )
        if dash:
            item["stroke-dasharray"] = dash
        (parent or self.dwg).add(item)
        return item

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str | None = None,
        width: float = 1.5,
        opacity: float = 1.0,
        dash: str | None = None,
        marker_end: str | None = None,
        parent=None,
    ):
        item = self.dwg.line(
            start=(x1, y1),
            end=(x2, y2),
            stroke=stroke or self.theme.text_secondary,
            stroke_width=width,
            opacity=opacity,
            stroke_linecap="round",
        )
        if dash:
            item["stroke-dasharray"] = dash
        if marker_end:
            item["marker-end"] = f"url(#{marker_end})"
        (parent or self.dwg).add(item)
        return item

    def path(
        self,
        d: str,
        *,
        fill: str = "none",
        stroke: str | None = None,
        width: float = 1.5,
        opacity: float = 1.0,
        dash: str | None = None,
        marker_end: str | None = None,
        filter_id: str | None = None,
        parent=None,
    ):
        item = self.dwg.path(
            d=d,
            fill=fill,
            stroke=stroke or self.theme.text_secondary,
            stroke_width=width,
            opacity=opacity,
            stroke_linecap="round",
            stroke_linejoin="round",
        )
        if dash:
            item["stroke-dasharray"] = dash
        if marker_end:
            item["marker-end"] = f"url(#{marker_end})"
        if filter_id:
            item["filter"] = f"url(#{filter_id})"
        (parent or self.dwg).add(item)
        return item

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.5,
        opacity: float = 1.0,
        filter_id: str | None = None,
        parent=None,
    ):
        item = self.dwg.circle(
            center=(cx, cy),
            r=r,
            fill=fill or self.theme.panel,
            stroke=stroke or self.theme.border,
            stroke_width=stroke_width,
            opacity=opacity,
        )
        if filter_id:
            item["filter"] = f"url(#{filter_id})"
        (parent or self.dwg).add(item)
        return item

    def polygon(self, points: Sequence[tuple[float, float]], *, fill: str, stroke: str | None = None, width: float = 1.0, opacity: float = 1.0, parent=None):
        item = self.dwg.polygon(points=points, fill=fill, stroke=stroke or fill, stroke_width=width, opacity=opacity)
        (parent or self.dwg).add(item)
        return item

    def text(
        self,
        value: str,
        x: float,
        y: float,
        *,
        size: float = 20,
        colour: str | None = None,
        weight: int | str = 500,
        anchor: str = "start",
        family: str = FONT,
        opacity: float = 1.0,
        parent=None,
        letter_spacing: float | None = None,
        baseline: str | None = None,
    ):
        item = self.dwg.text(
            value,
            insert=(x, y),
            fill=colour or self.theme.text,
            font_size=size,
            font_family=family,
            font_weight=normalise_weight(weight),
            text_anchor=anchor,
            opacity=opacity,
        )
        if letter_spacing is not None:
            item["letter-spacing"] = letter_spacing
        if baseline is not None:
            item["dominant-baseline"] = baseline
        (parent or self.dwg).add(item)
        return item

    def multiline(
        self,
        lines: Sequence[str],
        x: float,
        y: float,
        *,
        size: float = 20,
        line_height: float = 1.28,
        colour: str | None = None,
        weight: int | str = 500,
        anchor: str = "start",
        family: str = FONT,
        parent=None,
    ):
        item = self.dwg.text(
            "",
            insert=(x, y),
            fill=colour or self.theme.text,
            font_size=size,
            font_family=family,
            font_weight=normalise_weight(weight),
            text_anchor=anchor,
        )
        for i, line in enumerate(lines):
            span = self.dwg.tspan(line, x=[x], dy=[0 if i == 0 else size * line_height])
            item.add(span)
        (parent or self.dwg).add(item)
        return item

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        colour: str | None = None,
        width: float = 2.2,
        opacity: float = 1.0,
        dash: str | None = None,
        bend: float = 0.0,
        marker: str | None = None,
        parent=None,
    ):
        colour = colour or self.theme.text_secondary
        if marker is None:
            if colour == self.theme.support:
                marker = "arrow_support"
            elif colour == self.theme.selected:
                marker = "arrow_selected"
            elif colour == self.theme.oppose:
                marker = "arrow_oppose"
            elif colour == self.theme.stable:
                marker = "arrow_stable"
            elif colour == self.theme.uncertainty:
                marker = "arrow_uncertainty"
            elif colour == self.theme.latent:
                marker = "arrow_latent"
            else:
                marker = "arrow"
        if abs(bend) < 1e-6:
            return self.line(x1, y1, x2, y2, stroke=colour, width=width, opacity=opacity, dash=dash, marker_end=marker, parent=parent)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + bend
        d = f"M {x1} {y1} Q {mx} {my} {x2} {y2}"
        return self.path(d, stroke=colour, width=width, opacity=opacity, dash=dash, marker_end=marker, parent=parent)

    def bezier(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        colour: str,
        width: float,
        opacity: float = 0.75,
        marker_end: str | None = None,
        parent=None,
    ):
        dx = (x2 - x1) * 0.45
        d = f"M {x1} {y1} C {x1+dx} {y1}, {x2-dx} {y2}, {x2} {y2}"
        return self.path(d, stroke=colour, width=width, opacity=opacity, marker_end=marker_end, parent=parent)

    def pill(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        colour: str,
        fill_strength: float = 0.11,
        text_size: float = 16,
        parent=None,
    ):
        self.rect(x, y, w, h, fill=alpha(self.theme, colour, fill_strength), stroke=alpha(self.theme, colour, 0.78), stroke_width=1.4, radius=h/2, parent=parent)
        self.text(label, x+w/2, y+h/2+1, size=text_size, colour=colour, weight=650, anchor="middle", baseline="middle", parent=parent)

    def box_label(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str | Sequence[str],
        *,
        accent: str | None = None,
        fill: str | None = None,
        dashed: bool = False,
        sublabel: str | None = None,
        text_size: float = 20,
        parent=None,
    ):
        accent = accent or self.theme.support
        self.rect(x, y, w, h, fill=fill or self.theme.panel, stroke=self.theme.border, stroke_width=1.5, radius=16, dash="8 6" if dashed else None, parent=parent)
        self.rect(x, y, 8, h, fill=accent, stroke=accent, stroke_width=0, radius=6, parent=parent)
        lines = [label] if isinstance(label, str) else list(label)
        top = y + h/2 - (len(lines)-1)*text_size*0.58
        self.multiline(lines, x+24, top, size=text_size, colour=self.theme.text, weight=650, line_height=1.14, parent=parent)
        if sublabel:
            self.text(sublabel, x+24, y+h-18, size=13.5, colour=self.theme.text_secondary, weight=450, parent=parent)

    def separator(self, x1: float, y1: float, x2: float, y2: float, *, parent=None):
        self.line(x1, y1, x2, y2, stroke=self.theme.grid, width=1.1, opacity=0.95, parent=parent)

    def small_metric(self, x: float, y: float, label: str, value: str, colour: str, *, parent=None):
        self.circle(x, y, 7, fill=colour, stroke=colour, stroke_width=0, parent=parent)
        self.text(label, x+18, y+1, size=15, colour=self.theme.text_secondary, weight=500, baseline="middle", parent=parent)
        self.text(value, x+210, y+1, size=15, colour=colour, weight=700, anchor="end", baseline="middle", parent=parent)

    def matrix(
        self,
        x: float,
        y: float,
        rows: int,
        cols: int,
        cell: float,
        values: Sequence[Sequence[float]],
        *,
        negative: str,
        positive: str,
        neutral: str | None = None,
        border: bool = True,
        parent=None,
    ):
        neutral = neutral or self.theme.panel_alt
        for r in range(rows):
            for c in range(cols):
                v = max(-1.0, min(1.0, values[r][c]))
                fill = mix(neutral, negative, abs(v)) if v < 0 else mix(neutral, positive, v)
                self.rect(x+c*cell, y+r*cell, cell-1.0, cell-1.0, fill=fill, stroke=fill, stroke_width=0, radius=1.5, parent=parent)
        if border:
            self.rect(x-1, y-1, cols*cell+1, rows*cell+1, fill="none", stroke=self.theme.border, stroke_width=1.2, radius=5, parent=parent)

    def layer_stack(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        count: int,
        *,
        colour: str,
        label: str | None = None,
        parent=None,
    ):
        offset = 8
        for i in range(count-1, -1, -1):
            strength = 0.06 + (count-i)*0.02
            self.rect(x+i*offset, y-i*offset, w, h, fill=alpha(self.theme, colour, strength), stroke=alpha(self.theme, colour, 0.50), stroke_width=1.1, radius=12, parent=parent)
        if label:
            self.text(label, x+w/2+(count-1)*offset/2, y+h/2-(count-1)*offset/2, size=18, colour=self.theme.text, weight=650, anchor="middle", baseline="middle", parent=parent)

    def brace(self, x: float, y1: float, y2: float, *, side: str = "right", colour: str | None = None, parent=None):
        colour = colour or self.theme.text_secondary
        sign = 1 if side == "right" else -1
        mid = (y1+y2)/2
        d = (
            f"M {x} {y1} "
            f"C {x+sign*18} {y1}, {x+sign*18} {mid-16}, {x+sign*34} {mid-8} "
            f"C {x+sign*18} {mid-4}, {x+sign*18} {y2}, {x} {y2}"
        )
        self.path(d, stroke=colour, width=1.7, parent=parent)

    def save(self, path: str) -> None:
        self.dwg.saveas(path, pretty=True)


# Geometry helpers

def regular_polygon(cx: float, cy: float, r: float, n: int, rotation: float = -math.pi/2) -> list[tuple[float, float]]:
    return [(cx + r*math.cos(rotation+2*math.pi*i/n), cy + r*math.sin(rotation+2*math.pi*i/n)) for i in range(n)]


def interpolate_points(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]], t: float) -> list[tuple[float, float]]:
    return [((1-t)*x1+t*x2, (1-t)*y1+t*y2) for (x1,y1),(x2,y2) in zip(a,b, strict=False)]
