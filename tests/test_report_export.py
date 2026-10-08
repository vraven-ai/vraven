from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vraven.core.errors import ExportError
from vraven.report import export as report_export


def test_resolve_report_destination_infers_or_adds_extension(tmp_path: Path) -> None:
    inferred, inferred_format = report_export.resolve_report_destination(
        tmp_path / "report.html"
    )
    assert inferred == tmp_path / "report.html"
    assert inferred_format == "html"

    explicit, explicit_format = report_export.resolve_report_destination(
        tmp_path / "report",
        "pdf",
    )
    assert explicit == tmp_path / "report.pdf"
    assert explicit_format == "pdf"


def test_resolve_report_destination_rejects_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="does not match"):
        report_export.resolve_report_destination(tmp_path / "report.html", "pdf")


def test_html_to_pdf_requires_local_browser(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "report.html"
    source.write_text("<html><body>VRAVEN</body></html>", encoding="utf-8")
    monkeypatch.setattr(report_export, "find_chromium_executable", lambda: None)
    with pytest.raises(ExportError, match="requires Google Chrome"):
        report_export.html_to_pdf(source, tmp_path / "report.pdf")


def test_html_to_pdf_writes_one_valid_pdf(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "report.html"
    destination = tmp_path / "report.pdf"
    source.write_text("<html><body>VRAVEN</body></html>", encoding="utf-8")
    monkeypatch.setattr(
        report_export,
        "find_chromium_executable",
        lambda: Path("/mock/chrome"),
    )

    def fake_run(command, **_kwargs):
        output_argument = next(item for item in command if item.startswith("--print-to-pdf="))
        Path(output_argument.split("=", 1)[1]).write_bytes(b"%PDF-1.7\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(report_export.subprocess, "run", fake_run)
    written = report_export.html_to_pdf(source, destination)
    assert written == destination.resolve()
    assert destination.read_bytes().startswith(b"%PDF-")


def test_html_to_pdf_accepts_valid_file_written_before_browser_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "report.html"
    destination = tmp_path / "report.pdf"
    source.write_text("<html><body>VRAVEN</body></html>", encoding="utf-8")
    monkeypatch.setattr(
        report_export,
        "find_chromium_executable",
        lambda: Path("/mock/chrome"),
    )

    def fake_run(command, **_kwargs):
        output_argument = next(item for item in command if item.startswith("--print-to-pdf="))
        Path(output_argument.split("=", 1)[1]).write_bytes(b"%PDF-1.7\n")
        raise subprocess.TimeoutExpired(command, 45)

    monkeypatch.setattr(report_export.subprocess, "run", fake_run)
    assert report_export.html_to_pdf(source, destination) == destination.resolve()
