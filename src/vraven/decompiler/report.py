"""Focused HTML Decision Proof report for the VRAVEN causal decompiler."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.io as pio

from vraven.report.export import html_to_pdf, resolve_report_destination

if TYPE_CHECKING:  # pragma: no cover
    from .decompiler import DecompilationResult

from .feature_audit import ReverseFeatureAudit, build_reverse_feature_audits


class DecisionProofReport:
    """Render a compact, evidence-led report rather than a generic metrics dashboard."""

    visual_order = (
        ("model_xray", "What computation was recovered?", "Raven Eye · Full-stack model X-ray"),
        (
            "concept_genome",
            "What internal concepts formed and evolved?",
            "Raven Flight · Concept genome",
        ),
        (
            "causal_proof",
            "Which mechanisms survived intervention tests?",
            "Raven Brain · Causal proof graph",
        ),
        (
            "programme_proof",
            "Can the mechanism execute as a compact programme?",
            "Raven Code · Executable decision programme",
        ),
        (
            "contrastive_proof",
            "Why this class rather than the alternative?",
            "Raven Contrast · Why this rather than that?",
        ),
        (
            "counterfactual_twin",
            "What change reverses the decision?",
            "Raven Shadow · Counterfactual twin",
        ),
        ("dark_matter", "What remains unexplained?", "Raven Dark Matter · Explanation coverage"),
    )

    def __init__(self, result: DecompilationResult) -> None:
        self.result = result
        self.feature_audits = build_reverse_feature_audits(result)

    def write(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        initial_mode: str = "interactive_dark",
        include_plotlyjs: bool = True,
    ) -> Path:
        """Write exactly one Decision Proof report in HTML, PDF or JSON format."""

        destination, resolved_format = resolve_report_destination(path, format)
        if resolved_format == "html":
            return self.write_html(
                destination,
                initial_mode=initial_mode,
                include_plotlyjs=include_plotlyjs,
            )
        if resolved_format == "json":
            return self.write_json(destination)
        return self.write_pdf(destination)

    def write_feature(
        self,
        visual: str,
        path: str | Path,
        *,
        format: str | None = None,
        initial_mode: str = "interactive_dark",
        include_plotlyjs: bool = True,
    ) -> Path:
        """Write exactly one evidence-led feature report."""

        destination, resolved_format = resolve_report_destination(path, format)
        if resolved_format == "html":
            return self.write_feature_html(
                visual,
                destination,
                initial_mode=initial_mode,
                include_plotlyjs=include_plotlyjs,
            )
        if resolved_format == "json":
            return self.write_feature_json(visual, destination)
        return self.write_feature_pdf(visual, destination)

    @staticmethod
    def _metric_value(name: str, value: object) -> str:
        if value is None:
            return "not measured"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            percentage_terms = (
                "fraction",
                "coverage",
                "rate",
                "agreement",
                "stability",
                "cancellation",
            )
            if any(term in name for term in percentage_terms) and 0.0 <= value <= 1.0:
                return f"{100.0 * value:.1f}%"
            return f"{value:.4g}"
        return str(value)

    def _audit_html(self, audit: ReverseFeatureAudit) -> str:
        metrics = "".join(
            "<div class='audit-metric'><span>{}</span><strong>{}</strong></div>".format(
                html.escape(name.replace("_", " ")),
                html.escape(self._metric_value(name, value)),
            )
            for name, value in audit.metrics.items()
        )
        definitions = "".join(
            f"<li>{html.escape(definition)}</li>" for definition in audit.definitions
        )
        return (
            "<article class='feature-result'>"
            "<div class='result-head'><span>Primary scientific result</span>"
            f"<span class='status'>{html.escape(audit.evidence_status)}</span></div>"
            f"<div class='result-headline'>{html.escape(audit.headline)}</div>"
            f"<p class='result-meaning'>{html.escape(audit.scientific_interpretation)}</p>"
            "<div class='evidence-label'>Measured evidence</div>"
            f"<p class='measured-evidence'>{html.escape(audit.finding)}</p>"
            f"<div class='audit-grid'>{metrics}</div>"
            "<div class='result-details'>"
            "<details><summary>Scope and limitations</summary>"
            f"<p>{html.escape(audit.caution)}</p></details>"
            f"<details><summary>Metric definitions</summary><ul>{definitions}</ul></details>"
            f"<span class='tier'>Evidence tier · {html.escape(audit.evidence_tier)}</span>"
            "</div></article>"
        )

    def write_html(
        self,
        path: str | Path,
        *,
        initial_mode: str = "interactive_dark",
        include_plotlyjs: bool = True,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        dark = initial_mode != "publication_light"
        snippets: list[str] = []
        include_js: bool | str = include_plotlyjs
        for visual, question, heading in self.visual_order:
            dark_figure = self._prepare_embedded_figure(
                self.result.figure(visual, mode="interactive_dark")
            )
            light_figure = self._prepare_embedded_figure(
                self.result.figure(visual, mode="publication_light")
            )
            dark_html = pio.to_html(
                dark_figure,
                include_plotlyjs=include_js,
                full_html=False,
                config={"responsive": True, "displaylogo": False},
            )
            include_js = False
            light_html = pio.to_html(
                light_figure,
                include_plotlyjs=False,
                full_html=False,
                config={"responsive": True, "displaylogo": False},
            )
            snippets.append(
                "<section class='proof-section'>"
                f"<div class='section-kicker'>{html.escape(question)}</div>"
                f"<h3 class='section-title'>{html.escape(heading)}</h3>"
                f"{self._audit_html(self.feature_audits[visual])}"
                f"<div class='viz mode-dark'>{dark_html}</div>"
                f"<div class='viz mode-light'>{light_html}</div></section>"
            )
        destination.write_text(
            self._document("\n".join(snippets), initial_dark=dark),
            encoding="utf-8",
        )
        return destination

    def write_json(self, path: str | Path) -> Path:
        """Write the complete Decision Proof evidence as one machine-readable report."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = self.result
        payload = {
            "schema_version": "1.0",
            "report_type": "vraven_decision_proof",
            "summary": result.summary(),
            "manifest": result.to_manifest(),
            "feature_audits": {
                name: audit.to_dict() for name, audit in self.feature_audits.items()
            },
            "mechanism_certificate": result.certificate.to_dict(),
            "causal_mechanism_graph": result.causal_graph.to_dict(),
            "executable_programme": result.programme.to_dict(),
            "contrastive_explanation": result.contrastive.to_dict(),
            "counterfactual_proof": (
                result.counterfactual.to_dict() if result.counterfactual is not None else None
            ),
            "mechanism_profile": (
                result.profiler.to_dict() if result.profiler is not None else None
            ),
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    def write_pdf(self, path: str | Path) -> Path:
        """Write one publication-light PDF using a local Chromium-family browser."""

        destination = Path(path)
        with tempfile.TemporaryDirectory(prefix="vraven-report-") as directory:
            html_path = Path(directory) / "decision_proof.html"
            self.write_html(
                html_path,
                initial_mode="publication_light",
                include_plotlyjs=True,
            )
            return html_to_pdf(html_path, destination)

    def write_feature_html(
        self,
        visual: str,
        path: str | Path,
        *,
        initial_mode: str = "interactive_dark",
        include_plotlyjs: bool = True,
    ) -> Path:
        """Write one evidence-led standalone feature report with both render modes."""

        headings = {name: (question, heading) for name, question, heading in self.visual_order}
        if visual not in headings:
            supported = ", ".join(headings)
            raise KeyError(f"Unknown Reverse visual {visual!r}. Supported: {supported}")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        question, heading = headings[visual]
        dark_figure = self._prepare_embedded_figure(
            self.result.figure(visual, mode="interactive_dark")
        )
        light_figure = self._prepare_embedded_figure(
            self.result.figure(visual, mode="publication_light")
        )
        dark_html = pio.to_html(
            dark_figure,
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        light_html = pio.to_html(
            light_figure,
            include_plotlyjs=False,
            full_html=False,
            config={"responsive": True, "displaylogo": False},
        )
        destination.write_text(
            self._feature_document(
                question=question,
                heading=heading,
                audit=self.feature_audits[visual],
                dark_html=dark_html,
                light_html=light_html,
                initial_dark=initial_mode != "publication_light",
            ),
            encoding="utf-8",
        )
        return destination

    def write_feature_json(self, visual: str, path: str | Path) -> Path:
        """Write one feature audit and its plotted publication data as JSON."""

        headings = {name: (question, heading) for name, question, heading in self.visual_order}
        if visual not in headings:
            supported = ", ".join(headings)
            raise KeyError(f"Unknown Reverse visual {visual!r}. Supported: {supported}")
        question, heading = headings[visual]
        figure = self.result.figure(visual, mode="publication_light")
        payload = {
            "schema_version": "1.0",
            "report_type": "vraven_feature_report",
            "visual": visual,
            "question": question,
            "heading": heading,
            "summary": self.result.summary(),
            "feature_audit": self.feature_audits[visual].to_dict(),
            "figure": json.loads(figure.to_json()),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    def write_feature_pdf(self, visual: str, path: str | Path) -> Path:
        """Write one publication-light feature report as a PDF."""

        destination = Path(path)
        with tempfile.TemporaryDirectory(prefix="vraven-feature-report-") as directory:
            html_path = Path(directory) / f"{visual}.html"
            self.write_feature_html(
                visual,
                html_path,
                initial_mode="publication_light",
                include_plotlyjs=True,
            )
            return html_to_pdf(html_path, destination)

    def _feature_document(
        self,
        *,
        question: str,
        heading: str,
        audit: ReverseFeatureAudit,
        dark_html: str,
        light_html: str,
        initial_dark: bool,
    ) -> str:
        theme = "dark" if initial_dark else "light"
        return f"""<!doctype html>
<html lang='en' data-theme='{theme}'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(heading)} · VRAVEN scientific result</title>
<style>
:root{{--canvas:#030A12;--panel:#07111F;--raised:#0B1728;--border:#1C344D;--text:#F1F5F9;--muted:#94A3B8;--accent:#00D4FF;--result:#32D74B;--result-bg:#09251B}}
html[data-theme='light']{{--canvas:#fff;--panel:#fff;--raised:#f6f8fa;--border:#cbd5e1;--text:#111827;--muted:#526173;--accent:#006b8f;--result:#00796b;--result-bg:#e8f5f2}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--canvas);color:var(--text);font:15px/1.55 Inter,Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}header,.feature-result,.visual{{border:1px solid var(--border);background:var(--panel);border-radius:15px}}
header{{padding:24px;margin-bottom:18px;display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap}}
.kicker{{color:var(--accent);font-weight:750;text-transform:uppercase;letter-spacing:.09em;font-size:11px}}h1{{font-size:25px;margin:7px 0}}p{{color:var(--muted);max-width:92ch}}
.toggle{{display:flex;gap:4px;padding:4px;border:1px solid var(--border);border-radius:10px;background:var(--raised)}}button{{border:0;background:transparent;color:var(--muted);padding:8px 13px;border-radius:7px;cursor:pointer;font-weight:700}}button.active{{background:var(--text);color:var(--canvas)}}
.feature-result{{padding:22px 22px 19px;margin-bottom:18px;border-left:5px solid var(--result);background:linear-gradient(135deg,var(--result-bg),var(--panel) 48%)}}.result-head{{display:flex;justify-content:space-between;gap:12px;align-items:center;color:var(--result);font-weight:800;text-transform:uppercase;letter-spacing:.08em;font-size:11px}}
.status{{border:1px solid var(--result);border-radius:999px;padding:5px 10px;color:var(--result);background:var(--panel)}}.result-headline{{color:var(--text);font-size:clamp(22px,3vw,32px);font-weight:800;line-height:1.2;max-width:32ch;margin:15px 0 10px}}
.result-meaning{{font-size:16px;line-height:1.7;color:var(--text);max-width:105ch}}.evidence-label{{color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.07em;font-size:10px;margin-top:18px}}.measured-evidence{{color:var(--text);font-size:16px;margin-top:5px}}
.audit-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:9px}}.audit-metric{{border:1px solid var(--border);background:var(--raised);padding:10px 12px;border-radius:9px;display:flex;flex-direction:column;gap:5px}}
.audit-metric span{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}}.audit-metric strong{{font:700 16px ui-monospace,monospace;overflow-wrap:anywhere}}.result-details{{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-top:14px;padding-top:13px;border-top:1px solid var(--border)}}details,summary{{color:var(--muted)}}details p,details li{{font-size:13px}}.tier{{font-size:11px;color:var(--muted);margin-left:auto}}
.visual{{padding:14px;overflow:visible}}.viz{{width:100%;min-height:520px}}.viz .js-plotly-plot,.viz .plotly-graph-div{{width:100%!important;max-width:100%}}
html[data-theme='dark'] .mode-light{{display:none}}html[data-theme='light'] .mode-dark{{display:none}}
footer{{color:var(--muted);font-size:12px;padding:20px 4px}}@media(max-width:620px){{main{{padding:12px}}.toggle{{width:100%}}button{{flex:1}}}}
@media print{{@page{{size:A4 landscape;margin:10mm}}*{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}html,body{{background:#fff!important}}main{{max-width:none;padding:0}}.toggle{{display:none!important}}header,.feature-result,.visual{{box-shadow:none!important}}html[data-theme] .mode-dark{{display:none!important}}html[data-theme] .mode-light{{display:block!important}}details>*{{display:block!important}}}}
</style><script>function setTheme(t){{document.documentElement.dataset.theme=t;document.querySelectorAll('[data-theme-choice]').forEach(b=>b.classList.toggle('active',b.dataset.themeChoice===t));setTimeout(()=>document.querySelectorAll('.js-plotly-plot').forEach(g=>{{try{{Plotly.Plots.resize(g)}}catch(e){{}}}}),50)}}document.addEventListener('DOMContentLoaded',()=>{{document.querySelectorAll('[data-theme-choice]').forEach(b=>b.onclick=()=>setTheme(b.dataset.themeChoice));setTheme('{theme}')}})</script>
</head><body><main><header><div><div class='kicker'>VRAVEN Reverse · scientific result</div><h1>{html.escape(heading)}</h1><p>{html.escape(question)}</p></div><div class='toggle'><button data-theme-choice='dark'>Dark interactive</button><button data-theme-choice='light'>Publication light</button></div></header>
{self._audit_html(audit)}<section class='visual'><div class='viz mode-dark'>{dark_html}</div><div class='viz mode-light'>{light_html}</div></section>
<footer>VRAVEN presents the established result first, then preserves its evidence tier, denominator and scope for reproducible scientific interpretation.</footer></main></body></html>"""

    @staticmethod
    def _prepare_embedded_figure(figure):
        """Drop Plotly titles in the report shell so they cannot collide with HTML headings.

        Standalone figure HTML files keep their titles; only the Decision Proof report
        uses section kickers + ``h3`` headings instead.
        """

        annotations = figure.layout.annotations or ()
        has_overhead = any(
            str(getattr(annotation, "yref", "paper") or "paper").endswith("paper")
            and isinstance(getattr(annotation, "y", None), (int, float))
            and float(annotation.y) >= 0.995
            for annotation in annotations
        )
        margin = figure.layout.margin
        figure.update_layout(
            title=None,
            margin={
                "l": margin.l if margin and margin.l is not None else 84,
                "r": margin.r if margin and margin.r is not None else 48,
                "t": 72 if has_overhead else 28,
                "b": margin.b if margin and margin.b is not None else 72,
            },
        )
        return figure

    def _document(self, visuals: str, *, initial_dark: bool) -> str:
        result = self.result
        certificate = result.certificate
        summary = result.summary()
        target_label = result.class_names.get(result.target, f"class {result.target}")
        alternative_label = result.class_names.get(result.alternative, "alternative")
        claims = "".join(
            "<tr><td>{}</td><td><span class='tier'>{}</span></td><td>{}</td></tr>".format(
                html.escape(claim.claim),
                html.escape(claim.evidence_tier.replace("_", " ")),
                "Supported" if claim.supported else "Not established",
            )
            for claim in certificate.claims
        )
        warnings = "".join(f"<li>{html.escape(item)}</li>" for item in certificate.warnings)
        programme = html.escape(result.programme.to_pseudocode())
        plain = html.escape(result.contrastive.plain_english)
        causal_metrics = self.feature_audits["causal_proof"].metrics
        causal_tested = int(causal_metrics["tested_concepts"])
        causal_detected = round(causal_tested * causal_metrics["node_ci_exclusion_rate"])
        retained_mass = self.feature_audits["concept_genome"].metrics[
            "retained_transport_mass_fraction"
        ]
        data_json = html.escape(json.dumps(summary, indent=2))
        theme = "dark" if initial_dark else "light"
        return f"""<!doctype html>
<html lang='en' data-theme='{theme}'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VRAVEN Decision Proof</title>
<style>
:root{{
  --canvas:#030A12;--panel:#07111F;--raised:#0B1728;--border:#1C344D;
  --text:#F1F5F9;--muted:#94A3B8;--ink:#F1F5F9;
  --accent:#00D4FF;--good:#32D74B;--warn:#FF8A00;--bad:#FF2D55;
  --shadow:0 18px 50px rgba(0,0,0,.28);
}}
html[data-theme='light']{{
  --canvas:#f4f6f8;--panel:#ffffff;--raised:#f0f3f7;--border:#d0d7e2;
  --text:#152033;--muted:#5b6b7f;--ink:#152033;
  --accent:#152033;--good:#1f7a3d;--warn:#9a5b12;--bad:#a12835;
  --shadow:0 12px 36px rgba(15,23,42,.08);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--canvas);color:var(--text);font-family:"IBM Plex Sans",Inter,Arial,sans-serif;line-height:1.55}}
main{{max-width:1480px;margin:auto;padding:32px 28px 48px}}
header,.proof-section,.card{{background:var(--panel);border:1px solid var(--border);box-shadow:var(--shadow)}}
header{{border-radius:18px;padding:28px 28px 24px;margin-bottom:22px}}
.header-row{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap}}
h1{{font-size:clamp(24px,3vw,30px);margin:10px 0 8px;letter-spacing:.01em;line-height:1.2;color:var(--ink)}}
h2{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;margin:0 0 14px;padding-bottom:8px;border-bottom:1px solid var(--border);line-height:1.35;color:var(--ink)}}
p{{color:var(--muted);overflow-wrap:anywhere;max-width:78ch}}
.tag{{display:inline-block;padding:5px 10px;border:1px solid var(--border);border-radius:8px;font-size:12px;margin:0 6px 6px 0;color:var(--muted);background:var(--raised)}}
.tag.good{{color:var(--good);border-color:color-mix(in srgb,var(--good) 55%, var(--border))}}
.tag.warn{{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 55%, var(--border))}}
.toggle{{display:flex;gap:4px;padding:4px;border:1px solid var(--border);border-radius:10px;background:var(--raised);flex-shrink:0}}
button{{border:0;background:transparent;color:var(--muted);padding:8px 14px;border-radius:8px;font-weight:700;cursor:pointer}}
button.active{{background:var(--ink);color:var(--canvas)}}
.discovery-summary{{background:linear-gradient(135deg,color-mix(in srgb,var(--good) 12%,var(--panel)),var(--panel) 55%);border:1px solid color-mix(in srgb,var(--good) 48%,var(--border));border-left:5px solid var(--good);border-radius:16px;padding:23px 24px;margin-bottom:18px;box-shadow:var(--shadow)}}
.discovery-summary .kicker{{color:var(--good);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.09em}}.discovery-summary h2{{border:0;text-transform:none;letter-spacing:0;font-size:clamp(21px,2.7vw,30px);line-height:1.25;margin:9px 0;color:var(--ink);max-width:35ch}}.discovery-summary p{{color:var(--text);font-size:16px;max-width:100ch;margin-bottom:0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:22px}}
.card{{border-radius:14px;padding:18px 18px 16px;min-width:0}}
.metric{{font-size:clamp(18px,2.2vw,24px);font-weight:750;line-height:1.25;overflow-wrap:anywhere;color:var(--ink)}}
.label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;line-height:1.4;margin-bottom:10px}}
.narrative{{font-size:16px;color:var(--ink);line-height:1.65}}
.proof-section{{border-radius:14px;margin:26px 0;padding:22px 20px 16px;overflow:visible}}
.section-kicker{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;font-weight:750;padding:0;margin:0 0 6px;color:var(--muted);line-height:1.4}}
.section-title{{font-size:20px;font-weight:700;letter-spacing:.01em;margin:0 0 16px;padding:0 0 10px;border-bottom:1px solid var(--border);line-height:1.3;color:var(--ink)}}
.feature-result{{border:1px solid var(--border);border-left:5px solid var(--good);border-radius:12px;background:linear-gradient(135deg,color-mix(in srgb,var(--good) 8%,var(--raised)),var(--raised) 55%);padding:18px 18px 16px;margin:4px 0 16px}}
.result-head{{display:flex;justify-content:space-between;align-items:center;gap:12px;color:var(--good);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}}
.status{{border:1px solid var(--good);border-radius:999px;padding:4px 9px;color:var(--good);background:var(--panel)}}.result-headline{{color:var(--ink);font-size:clamp(18px,2vw,24px);font-weight:800;line-height:1.25;max-width:38ch;margin:12px 0 8px}}
.result-meaning{{color:var(--text);font-size:15px;line-height:1.65;max-width:105ch}}.evidence-label{{color:var(--muted);font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.07em;margin-top:15px}}.measured-evidence{{color:var(--ink);font-size:14px;margin:5px 0 12px;max-width:105ch}}
.audit-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:8px}}
.audit-metric{{border:1px solid var(--border);background:var(--panel);border-radius:8px;padding:9px 10px;display:flex;flex-direction:column;gap:4px}}
.audit-metric span{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em}}
.audit-metric strong{{color:var(--ink);font:700 14px ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}}
.result-details{{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-top:13px;padding-top:12px;border-top:1px solid var(--border)}}.result-details details{{margin:0}}.result-details details p,.result-details details li{{font-size:13px}}.result-details .tier{{margin-left:auto;color:var(--muted)}}
.viz{{width:100%;overflow:visible;min-height:420px;margin-top:6px}}
.viz .js-plotly-plot,.viz .plotly-graph-div{{width:100% !important;max-width:100%}}
html[data-theme='dark'] .mode-light{{display:none}}
html[data-theme='light'] .mode-dark{{display:none}}
table{{width:100%;border-collapse:collapse}}
td,th{{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top;overflow-wrap:anywhere}}
th{{color:var(--muted);font-weight:600}}
.tier{{font-size:11px;padding:4px 8px;border-radius:8px;background:var(--raised);border:1px solid var(--border);white-space:nowrap;color:var(--ink)}}
pre{{white-space:pre-wrap;background:var(--raised);border:1px solid var(--border);padding:16px;border-radius:12px;color:var(--ink);overflow:auto}}
details{{margin-top:16px}}
summary{{cursor:pointer;color:var(--ink);font-weight:700}}
.warning-list{{color:var(--warn)}}
footer{{color:var(--muted);font-size:12px;padding:20px 3px 40px;line-height:1.5}}
@media(max-width:980px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:620px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}.toggle{{width:100%;justify-content:stretch}} .toggle button{{flex:1}}}}
@media print{{@page{{size:A4 landscape;margin:10mm}}*{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}html,body{{background:#fff!important}}main{{max-width:none;padding:0}}.toggle{{display:none!important}}header,.proof-section,.card,.feature-result{{box-shadow:none!important}}html[data-theme] .mode-dark{{display:none!important}}html[data-theme] .mode-light{{display:block!important}}.proof-section{{break-before:page;break-inside:avoid}}details>*{{display:block!important}}}}
</style>
<script>
function setTheme(t){{document.documentElement.dataset.theme=t;document.querySelectorAll('[data-theme-choice]').forEach(b=>b.classList.toggle('active',b.dataset.themeChoice===t));setTimeout(()=>{{document.querySelectorAll('.js-plotly-plot').forEach(g=>{{try{{Plotly.Plots.resize(g)}}catch(e){{}}}})}},50);}}
document.addEventListener('DOMContentLoaded',()=>{{document.querySelectorAll('[data-theme-choice]').forEach(b=>b.onclick=()=>setTheme(b.dataset.themeChoice));setTheme('{theme}')}});
</script>
</head>
<body><main>
<header><div class='header-row'><div><div class='tag'>VRAVEN neural decompiler</div><div class='tag good'>{html.escape(result.captured.ir.capture_route.value)}</div><h1>Decision Proof · {html.escape(target_label)}</h1><p>A compact, executable and intervention-tested account of why the model preferred {html.escape(target_label)} over {html.escape(alternative_label)}.</p></div><div class='toggle'><button data-theme-choice='dark'>Dark interactive</button><button data-theme-choice='light'>Publication light</button></div></div></header>
<section class='discovery-summary'><div class='kicker'>Recovered scientific result</div><h2>VRAVEN reconstructed, tested and compressed this model's decision mechanism.</h2><p>For this artefact, VRAVEN recovered the executed graph with numerical equivalence, traced {100 * retained_mass:.1f}% of aligned representation mass and reproduced held-out decisions with {100 * result.programme.fidelity.validation_decision_agreement:.1f}% agreement using {len(result.programme.rules)} executable rules. Direct interventions identified non-zero effects in {causal_detected} of {causal_tested} tested concepts.</p></section>
<div class='grid'>
<div class='card'><div class='label'>Numerical equivalence</div><div class='metric'>{"Passed" if result.captured.ir.coverage.numeric_equivalence_passed else "Not passed"}</div></div>
<div class='card'><div class='label'>Held-out decision agreement</div><div class='metric'>{100 * result.programme.fidelity.validation_decision_agreement:.1f}%</div></div>
<div class='card'><div class='label'>Held-out programme R²</div><div class='metric'>{result.programme.fidelity.validation_r2:.3f}</div></div>
<div class='card'><div class='label'>Causal concept effects detected</div><div class='metric'>{causal_detected} / {causal_tested}</div></div>
<div class='card'><div class='label'>Aligned transport mass retained</div><div class='metric'>{100 * retained_mass:.1f}%</div></div>
<div class='card'><div class='label'>Executable programme</div><div class='metric'>{len(result.programme.rules)} rules</div></div>
</div>
<div class='card'><h2>Decision conclusion</h2><p class='narrative'>{plain}</p></div>
{visuals}
<div class='card'><h2>Executable programme</h2><pre>{programme}</pre></div>
<div class='card'><h2>Evidence claim lattice</h2><table><thead><tr><th>Claim</th><th>Evidence tier</th><th>Status</th></tr></thead><tbody>{claims}</tbody></table></div>
<div class='card'><h2>Reproducibility and evidence boundaries</h2><details><summary>Scope, unresolved evidence and next validation targets</summary><ul class='warning-list'>{warnings}</ul></details><details><summary>Machine-readable summary</summary><pre>{data_json}</pre></details></div>
<footer>VRAVEN reports the recovered result at its strongest supported evidence tier and preserves every denominator, validation boundary and unresolved test for reproducibility.</footer>
</main></body></html>"""
