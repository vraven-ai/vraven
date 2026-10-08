from __future__ import annotations

import io

from vraven.terminal import TerminalUI, emit_result


def test_terminal_panel_and_dancing_mascot() -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, sleep=lambda _delay: None, force=True)
    ui.panel("ADVERSARIAL AUDIT", {"Samples": 8, "Robust accuracy": "75.00%"})
    ui.mascot(style="dance", cycles=1, delay=0)
    output = stream.getvalue()
    assert "ADVERSARIAL AUDIT" in output
    assert "Robust accuracy" in output
    assert "VRAVEN" in output
    assert "\x1b[" in output


def test_emit_result_remains_json_when_not_interactive() -> None:
    stream = io.StringIO()
    emit_result(
        {"status": "ok"},
        title="RESULT",
        rows={"Status": "ok"},
        footer="saved",
        stream=stream,
    )
    assert stream.getvalue() == '{\n  "status": "ok"\n}\n'
