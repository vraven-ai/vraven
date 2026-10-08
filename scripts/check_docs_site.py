#!/usr/bin/env python3
"""Validate local links and essential metadata in the static documentation site."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[tuple[str, str]] = []
        self.title = ""
        self.description = ""
        self.has_viewport = False
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "meta" and values.get("name") == "description":
            self.description = values.get("content") or ""
        if tag == "meta" and values.get("name") == "viewport":
            self.has_viewport = "width=device-width" in (values.get("content") or "")
        for attribute in ("href", "src"):
            value = values.get(attribute)
            if value:
                self.references.append((attribute, value))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def local_target(page: Path, reference: str) -> Path | None:
    parts = urlsplit(reference)
    if parts.scheme or parts.netloc or reference.startswith(("#", "mailto:")):
        return None
    path = unquote(parts.path)
    if not path:
        return None
    if path.startswith("/"):
        return SITE / path.lstrip("/")
    return (page.parent / path).resolve()


def main() -> int:
    failures: list[str] = []

    root_clutter = (
        "START_HERE.md",
        "MIGRATION_NOTE.md",
        "REFERENCES.bib",
        "requirements.txt",
        "render_all.py",
        "render_one.py",
        "kronodroid_vraven_explain.ipynb",
        "winlolbin_vraven_explain.ipynb",
        "winlolbin_vraven_darkside.ipynb",
    )
    for filename in root_clutter:
        if (ROOT / filename).exists():
            failures.append(f"repository root clutter returned: {filename}")
    if not (ROOT / "docs" / "references" / "VRAVEN_REFERENCES.bib").is_file():
        failures.append("canonical bibliography is missing")

    pages = sorted(SITE.rglob("*.html"))
    if not pages:
        failures.append("site contains no HTML pages")

    titles: dict[str, Path] = {}
    for page in pages:
        parser = References()
        parser.feed(page.read_text(encoding="utf-8"))
        title = parser.title.strip()
        if not title:
            failures.append(f"{page.relative_to(ROOT)}: missing <title>")
        elif title in titles:
            failures.append(
                f"{page.relative_to(ROOT)}: duplicate title also used by "
                f"{titles[title].relative_to(ROOT)}"
            )
        else:
            titles[title] = page
        if not parser.description.strip():
            failures.append(f"{page.relative_to(ROOT)}: missing meta description")
        if not parser.has_viewport:
            failures.append(f"{page.relative_to(ROOT)}: missing responsive viewport metadata")

        source = page.read_text(encoding="utf-8")
        for forbidden in ("file:///Users/", "/Users/admin", "github_pat_", "ghp_"):
            if forbidden in source:
                failures.append(
                    f"{page.relative_to(ROOT)}: contains private or unsafe text {forbidden!r}"
                )

        for attribute, reference in parser.references:
            target = local_target(page, reference)
            if target is not None and not target.exists():
                failures.append(
                    f"{page.relative_to(ROOT)}: broken {attribute}={reference!r}"
                )

    required = [
        SITE / "index.html",
        SITE / "404.html",
        SITE / ".nojekyll",
        SITE / "assets" / "styles.css",
        SITE / "assets" / "app.js",
        SITE / "assets" / "vraven-avatar.png",
        SITE / "capabilities.html",
        SITE / "visuals.html",
        SITE / "gallery.html",
        SITE / "python-api.html",
        SITE / "api" / "index.html",
        SITE / "api" / "inventory.json",
        SITE / "cli" / "index.html",
        SITE / "cli" / "inventory.json",
        SITE / "outputs" / "index.html",
        SITE / "outputs" / "inventory.json",
        SITE / "science.html",
        SITE / "research.html",
        SITE / "identity.html",
        SITE / "walkthroughs" / "mobilenet-v2.html",
        SITE / "examples" / "mobilenet-v2" / "vraven-report.html",
        SITE / "examples" / "mobilenet-v2" / "summary.json",
        SITE / "guides" / "getting-started.html",
        SITE / "guides" / "model-folder.html",
        SITE / "guides" / "source-checkout.html",
    ]
    for path in required:
        if not path.exists():
            failures.append(f"missing required site file: {path.relative_to(ROOT)}")

    inventory_checks = [
        (SITE / "outputs" / "inventory.json", SITE / "outputs", 50),
        (SITE / "api" / "inventory.json", SITE / "api", 100),
        (SITE / "cli" / "inventory.json", SITE / "cli", 20),
    ]
    for inventory_path, directory, minimum in inventory_checks:
        if not inventory_path.exists():
            continue
        data = json.loads(inventory_path.read_text(encoding="utf-8"))
        count = int(data.get("count", 0))
        generated = len(list(directory.glob("*.html"))) - 1
        if directory.name == "outputs":
            generated = len(list(directory.glob("*/*.html")))
        if count < minimum:
            failures.append(f"{inventory_path.relative_to(ROOT)}: unexpectedly small inventory ({count})")
        if generated != count:
            failures.append(f"{directory.relative_to(ROOT)}: inventory says {count}, found {generated} generated pages")

    artifact_bytes = sum(path.stat().st_size for path in (SITE / "assets" / "artifacts").glob("*.png"))
    if artifact_bytes > 12 * 1024 * 1024:
        failures.append(f"site/assets/artifacts: image budget exceeded ({artifact_bytes / 1024 / 1024:.1f} MiB)")

    styles = (SITE / "assets" / "styles.css").read_text(encoding="utf-8")
    for breakpoint in ("max-width: 1200px", "max-width: 900px", "max-width: 720px", "max-width: 420px"):
        if breakpoint not in styles:
            failures.append(f"site/assets/styles.css: missing responsive breakpoint {breakpoint}")

    app = (SITE / "assets" / "app.js").read_text(encoding="utf-8")
    for required_text in (
        "VRAVEN · Visual Reasoning and Activation Visualisation for Explainable Networks · Apache-2.0",
        "© 2026 VRAVEN",
        "Identity &amp; origin",
        "Research lineage",
        "detectCodeLanguage",
        "highlightPython",
        "highlightCode",
        "prefers-reduced-motion",
    ):
        if required_text not in app:
            failures.append(f"site/assets/app.js: missing site identity text {required_text!r}")

    homepage = (SITE / "index.html").read_text(encoding="utf-8")
    if 'class="count"' in homepage:
        failures.append("site/index.html: decorative capability numbering returned")
    if 'href="research.html"' not in homepage:
        failures.append("site/index.html: missing research-foundations entry point")
    if 'href="identity.html"' not in homepage:
        failures.append("site/index.html: missing identity-and-origin entry point")
    if 'class="vraven-wordmark hero-wordmark"' not in homepage:
        failures.append("site/index.html: missing VRAVEN signature wordmark")
    if 'class="raven-portrait"' not in homepage:
        failures.append("site/index.html: raven image is not placed in the signature composition")
    if "Figure 3" in homepage:
        failures.append("site/index.html: removed manuscript figure numbering returned")
    gallery = (SITE / "gallery.html").read_text(encoding="utf-8")
    if "One model, four evidence views" in gallery:
        failures.append("site/gallery.html: misleading four-view limitation returned")
    if gallery.count('class="shot" href="outputs/') < 4:
        failures.append("site/gallery.html: gallery previews must link to their output guides")
    visuals = (SITE / "visuals.html").read_text(encoding="utf-8")
    if "Select any entry" not in visuals or "visualRoutes" not in app:
        failures.append("site/visuals.html: visual catalogue entries are not routed to usage guides")
    if "Why this walkthrough looks familiar" in (SITE / "walkthroughs" / "mobilenet-v2.html").read_text(encoding="utf-8"):
        failures.append("site/walkthroughs/mobilenet-v2.html: removed comparison callout returned")

    getting_started = (SITE / "guides" / "getting-started.html").read_text(encoding="utf-8")
    if "Guide 01" in getting_started or 'class="steps"' in getting_started:
        failures.append("site/guides/getting-started.html: decorative guide numbering returned")

    source_only_command = "python3 scripts/run_vraven_artifact.py Models/image_pattern_cnn2d"
    generated_output_pages = list((SITE / "outputs").glob("*/*.html"))
    for output_page in generated_output_pages:
        if source_only_command in output_page.read_text(encoding="utf-8"):
            failures.append(
                f"{output_page.relative_to(ROOT)}: source-only fixture command used as public CLI"
            )

    if failures:
        print("Documentation validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Validated {len(pages)} HTML pages and their local assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
