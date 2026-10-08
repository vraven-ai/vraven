"""HTML report dark/light theme toggle."""

from __future__ import annotations

from pathlib import Path

import torch

import vraven


def test_export_html_includes_theme_toggle(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )
    samples = torch.randn(12, 8)
    report = vraven.explain(
        model,
        samples,
        target=1,
        mode="gradient",
        class_names=["benign", "malware"],
    )
    path = tmp_path / "report.html"
    report.export_html(
        path,
        visuals=("input_attribution", "decision_flow"),
        mode="interactive_dark",
        theme_toggle=True,
    )
    html = path.read_text(encoding="utf-8")
    assert 'data-theme-choice="dark"' in html
    assert 'data-theme-choice="light"' in html
    assert 'data-theme="dark"' in html
    assert html.count('class="viz-panel"') >= 2
    assert "vraven-report-theme" in html
    assert html.count("class='figure-evidence'") >= 2
    assert "Scientific question" in html
    assert "Methods, mappings and limitations" in html
    assert "Download exact plotted data (.json)" in html
    assert "Copy journal caption" in html
    assert "data-publication-caption=" in html
    assert "Strongest displayed" in html
    assert "--vraven-canvas: #030A12" in html
    assert "--vraven-panel: #07111F" in html
    assert "--vraven-border: #1C344D" in html
    assert "--vraven-accent: #00D4FF" in html


def test_export_html_can_disable_theme_toggle(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    report = vraven.explain(model, torch.randn(4, 4), target=0, mode="fast")
    path = tmp_path / "single.html"
    report.export_html(
        path,
        visuals=("activation_energy",),
        mode="publication_light",
        theme_toggle=False,
    )
    html = path.read_text(encoding="utf-8")
    assert "theme-toggle" not in html
    assert 'data-theme-choice="light"' not in html
