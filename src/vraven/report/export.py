"""Single-file report export helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from vraven.core.errors import ExportError

REPORT_FORMATS = ("html", "pdf", "json")


def resolve_report_destination(
    path: str | Path,
    report_format: str | None = None,
) -> tuple[Path, str]:
    """Resolve and validate a single report destination and format."""

    destination = Path(path)
    requested = report_format.lower().lstrip(".") if report_format else None
    suffix_format = destination.suffix.lower().lstrip(".") or None
    resolved = requested or suffix_format
    if resolved not in REPORT_FORMATS:
        supported = ", ".join(REPORT_FORMATS)
        raise ExportError(
            f"Unsupported report format {resolved!r}. Choose one of: {supported}."
        )
    if suffix_format is not None and suffix_format != resolved:
        raise ExportError(
            f"Report format {resolved!r} does not match destination extension "
            f"{destination.suffix!r}."
        )
    if suffix_format is None:
        destination = destination.with_suffix(f".{resolved}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination, resolved


def find_chromium_executable() -> Path | None:
    """Find a Chrome or Chromium executable without downloading software."""

    configured = os.environ.get("VRAVEN_CHROME_PATH")
    candidates = [configured] if configured else []
    candidates.extend(
        shutil.which(name)
        for name in (
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "chrome",
            "msedge",
        )
    )
    candidates.extend(
        (
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        )
    )
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        root = os.environ.get(variable)
        if root:
            candidates.append(str(Path(root) / "Google/Chrome/Application/chrome.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def html_to_pdf(html_path: str | Path, pdf_path: str | Path) -> Path:
    """Print one self-contained local HTML report to a single PDF file."""

    source = Path(html_path).resolve()
    destination = Path(pdf_path).resolve()
    browser = find_chromium_executable()
    if browser is None:
        raise ExportError(
            "PDF report export requires Google Chrome, Chromium or Microsoft Edge. "
            "Install a compatible browser or set VRAVEN_CHROME_PATH to its executable. "
            "HTML and JSON report export do not require a browser."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vraven-pdf-profile-") as profile:
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=15000",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={destination}",
            source.as_uri(),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=45,
            )
        except subprocess.TimeoutExpired as error:
            # Chrome on macOS can finish writing the PDF but keep its parent
            # process alive. The artefact, not process shutdown timing, is the
            # authoritative success condition.
            if _is_valid_pdf(destination):
                return destination
            destination.unlink(missing_ok=True)
            raise ExportError(f"Chrome timed out before producing the VRAVEN PDF report: {error}") from error
        except OSError as error:
            raise ExportError(f"Chrome could not render the VRAVEN PDF report: {error}") from error
    if completed.returncode != 0 or not destination.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        if len(detail) > 500:
            detail = detail[-500:]
        raise ExportError(
            "Chrome could not render the VRAVEN PDF report. "
            f"Exit code: {completed.returncode}. {detail}"
        )
    if not _is_valid_pdf(destination):
        destination.unlink(missing_ok=True)
        raise ExportError("Chrome did not produce a valid PDF report.")
    return destination


def _is_valid_pdf(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= 5 and path.read_bytes()[:5] == b"%PDF-"
