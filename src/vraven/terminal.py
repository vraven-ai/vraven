"""Dependency-free terminal presentation for the VRAVEN command-line interface."""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TextIO

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_RED = "\x1b[38;5;196m"
_GREY = "\x1b[38;5;245m"
_WHITE = "\x1b[38;5;255m"

_RAVEN_FRAMES = {
    "still": (
        "      __\n"
        "  ___/  \\_\n"
        " /  _    _>\n"
        "/  / \\__/\n"
        "\\_/   /\n"
        "     _/\\_\n"
        "    VRAVEN",
    ),
    "blink": (
        "      __\n"
        "  ___/ o\\_\n"
        " /  _    _>\n"
        "/  / \\__/\n"
        "\\_/   /\n"
        "     _/\\_\n"
        "    VRAVEN",
        "      __\n"
        "  ___/ -\\_\n"
        " /  _    _>\n"
        "/  / \\__/\n"
        "\\_/   /\n"
        "     _/\\_\n"
        "    VRAVEN",
    ),
    "dance": (
        "   \\  __\n"
        "  ___/ o\\_\n"
        " /  _    _>\n"
        "/  / \\__/\n"
        "\\_/   /\n"
        "     _/\\_\n"
        "   < VRAVEN",
        "      __  /\n"
        "  ___/o \\_\n"
        " /  _    _>\n"
        "/  / \\__/\n"
        "\\_/   /\n"
        "     _/\\_\n"
        "    VRAVEN >",
    ),
}


def _colour_enabled(stream: TextIO) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ


def _paint(value: str, colour: str, *, enabled: bool) -> str:
    return f"{colour}{value}{_RESET}" if enabled else value


@dataclass(slots=True)
class TerminalUI:
    """Render restrained terminal panels and an optional raven mascot."""

    stream: TextIO = field(default_factory=lambda: sys.stdout)
    sleep: Callable[[float], None] = time.sleep
    force: bool | None = None

    @property
    def interactive(self) -> bool:
        if self.force is not None:
            return self.force
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def colour(self) -> bool:
        return self.interactive and _colour_enabled(self.stream)

    def panel(
        self,
        title: str,
        rows: Mapping[str, object] | Sequence[tuple[str, object]],
        *,
        footer: str | None = None,
    ) -> None:
        """Write a compact result panel sized for the current terminal."""

        items = list(rows.items()) if isinstance(rows, Mapping) else list(rows)
        terminal_width = shutil.get_terminal_size((88, 24)).columns
        width = max(44, min(88, terminal_width - 2))
        top = "╭" + "─" * (width - 2) + "╮"
        bottom = "╰" + "─" * (width - 2) + "╯"
        title_text = f" VRAVEN · {title} "[: width - 4]
        title_line = "│" + title_text.ljust(width - 2) + "│"
        lines = [top, title_line, "├" + "─" * (width - 2) + "┤"]
        label_width = min(24, max((len(str(label)) for label, _ in items), default=8))
        for label, value in items:
            text = f"  {str(label).ljust(label_width)}  {value}"
            lines.append("│" + text[: width - 3].ljust(width - 2) + "│")
        if footer:
            lines.extend(["├" + "─" * (width - 2) + "┤"])
            lines.append("│" + f"  {footer}"[: width - 3].ljust(width - 2) + "│")
        lines.append(bottom)
        rendered = "\n".join(lines)
        if self.colour:
            rendered = rendered.replace("VRAVEN", _paint("VRAVEN", _RED + _BOLD, enabled=True), 1)
        print(rendered, file=self.stream)

    def mascot(self, *, style: str = "blink", cycles: int = 2, delay: float = 0.18) -> None:
        """Show the raven; animate only when output is an interactive terminal."""

        if style not in _RAVEN_FRAMES:
            raise ValueError(f"Unknown raven style {style!r}")
        frames = _RAVEN_FRAMES[style]
        if not self.interactive or style == "still":
            print(_paint(frames[0], _WHITE + _BOLD, enabled=self.colour), file=self.stream)
            return
        repeats = max(1, int(cycles))
        line_count = frames[0].count("\n") + 1
        for index in range(repeats * len(frames)):
            frame = frames[index % len(frames)]
            print(_paint(frame, _WHITE + _BOLD, enabled=self.colour), file=self.stream, flush=True)
            self.sleep(max(0.0, float(delay)))
            if index + 1 < repeats * len(frames):
                self.stream.write(f"\x1b[{line_count}A")
        print(_paint("Ready to inspect the mechanism.", _GREY, enabled=self.colour), file=self.stream)


def emit_result(
    payload: Mapping[str, object],
    *,
    title: str,
    rows: Mapping[str, object] | Sequence[tuple[str, object]],
    footer: str,
    stream: TextIO | None = None,
) -> None:
    """Show a human panel on a TTY and preserve JSON on pipes and in automation."""

    destination = stream or sys.stdout
    ui = TerminalUI(stream=destination)
    if ui.interactive:
        ui.panel(title, rows, footer=footer)
    else:
        import json

        print(json.dumps(payload, indent=2), file=destination)
