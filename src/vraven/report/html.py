"""Explicit composition of standalone figures into a self-contained HTML report."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as plotly_io

from vraven.core.errors import CapabilityError
from vraven.visualisation.palette import RenderMode
from vraven.visualisation.theme import THEMES

if TYPE_CHECKING:
    from vraven.report.report import VRavenReport


DEFAULT_VISUAL_ORDER = (
    "input_attribution",
    "feature_sequence_heatmap",
    "neuron_importance",
    "neuron_impact",
    "decision_sankey",
    "decision_flow",
    "latent_flight",
    "relevance_flow",
    "counterfactual_delta",
    "perturbation_stability",
    "reasoning_dna",
    "spectral_trace",
    "conflict_field",
    "evidence_trajectory",
    "causal_pathway",
    "mechanism_concordance",
    "activation_energy",
    "reasoning_graph",
    "community_graph",
    "attention_structure",
)

TITLE_MAP = {
    "input_attribution": "1. Local Feature Attribution",
    "feature_sequence_heatmap": "2. Feature Attribution Heatmap",
    "decision_sankey": "3. First-order Contribution Flow",
    "decision_flow": "4. Decision Flow",
    "neuron_impact": "5. Compressed-unit Influence Field",
    "neuron_importance": "6. Top Local Unit Contributions",
    "latent_flight": "7. Representation Landscape (t-SNE / PCA)",
    "relevance_flow": "8. Layer-wise Relevance Propagation",
    "counterfactual_delta": "9. Counterfactual Explanation",
    "perturbation_stability": "10. Perturbation Test (Prediction Stability)",
    "consistency_matrix": "11. Explanation Consistency Matrix",
    "reasoning_dna": "12. Reasoning DNA",
    "spectral_trace": "13. Spectral Trace",
    "conflict_field": "14. Separated Uncertainty Diagnostics",
    "evidence_trajectory": "15. Evidence Trajectory",
    "causal_pathway": "16. Single-layer Intervention Effects",
    "mechanism_concordance": "17. Mechanism Concordance",
    "activation_energy": "18. Activation Energy",
    "reasoning_graph": "19. Reasoning Graph",
    "community_graph": "20. Activation Community Graph",
    "attention_structure": "21. Attention Structure",
}

_MODE_TO_THEME_KEY = {
    "interactive_dark": "dark",
    "publication_light": "light",
}


def _summary_rows(summary: dict[str, object]) -> str:
    return "".join(
        f"<tr><th>{escape(str(key).replace('_', ' ').title())}</th>"
        f"<td>{escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )


def _number(value: float | None, *, digits: int = 3) -> str:
    return "Not estimable" if value is None else f"{value:.{digits}f}"


def _percent(value: float | None) -> str:
    return "Not estimable" if value is None else f"{value:.1%}"


_EVIDENCE_TIER_LABELS = {
    "predictive": "Predictive",
    "observational": "Observational",
    "comparative": "Comparative",
    "behavioural": "Behavioural stress test",
    "interventional": "Targeted interventional",
    "observational_interventional": "Observational ↔ interventional",
    "formal": "Formal certificate",
}


def _figure_scientific_metadata(figure: go.Figure) -> dict[str, Any]:
    layout_meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
    scientific = layout_meta.get("scientific")
    return dict(scientific) if isinstance(scientific, dict) else {}


def _publication_caption(title: str, scientific: dict[str, Any]) -> str:
    question = str(scientific.get("scientific_question") or "Measured VRAVEN result.")
    interpretation = str(scientific.get("interpretation") or "")
    metrics = ", ".join(str(value) for value in scientific.get("metric_names") or ())
    normalisation = scientific.get("normalisation")
    limitations = [str(value) for value in scientific.get("limitations") or ()]
    parts = [f"{title}. {question}"]
    if interpretation:
        parts.append(interpretation)
    if metrics:
        parts.append(f"Metrics: {metrics}.")
    if normalisation:
        parts.append(f"Normalisation: {normalisation}.")
    if limitations:
        parts.append(f"Leading limitation: {limitations[0]}")
    return " ".join(parts)


def _publication_evidence_html(figure: go.Figure, title: str) -> str:
    """Expose the claim boundary and exact quantitative takeaways beside a figure."""

    scientific = _figure_scientific_metadata(figure)
    if not scientific:
        return ""
    tier = str(scientific.get("evidence_tier") or "observational")
    tier_label = _EVIDENCE_TIER_LABELS.get(tier, tier.replace("_", " ").title())
    question = str(scientific.get("scientific_question") or "Not documented")
    interpretation = str(scientific.get("interpretation") or "Not documented")
    metrics = [str(value) for value in scientific.get("metric_names") or ()]
    encodings = [str(value) for value in scientific.get("visual_encodings") or ()]
    definitions = [str(value) for value in scientific.get("metric_definitions") or ()]
    limitations = [str(value) for value in scientific.get("limitations") or ()]
    normalisation = scientific.get("normalisation")
    results = [value for value in scientific.get("key_results") or () if isinstance(value, dict)]

    result_html = ""
    if results:
        cards = []
        for result in results:
            unit = result.get("unit")
            unit_html = f" <span>{escape(str(unit))}</span>" if unit else ""
            cards.append(
                "<div class='figure-result'>"
                f"<div class='figure-result-label'>{escape(str(result.get('label', 'Result')))}</div>"
                f"<div class='figure-result-value'>{escape(str(result.get('value', '—')))}{unit_html}</div>"
                f"<div class='figure-result-note'>{escape(str(result.get('interpretation', '')))}</div>"
                "</div>"
            )
        result_html = f"<div class='figure-result-grid'>{''.join(cards)}</div>"

    method_items = []
    if metrics:
        method_items.append(("Measured quantities", "; ".join(metrics)))
    if definitions:
        method_items.append(("Definitions", "; ".join(definitions)))
    if encodings:
        method_items.append(("Visual mapping", "; ".join(encodings)))
    if normalisation:
        method_items.append(("Normalisation", str(normalisation)))
    if limitations:
        method_items.append(("Claim boundary", "; ".join(limitations)))
    method_html = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in method_items
    )
    caption = _publication_caption(title, scientific)
    visual_id = str(scientific.get("visual_id") or title).replace("vraven.", "")
    return f"""
<div class='figure-evidence' data-figure-id='{escape(visual_id, quote=True)}'
     data-publication-caption='{escape(caption, quote=True)}'>
  <div class='figure-evidence-head'>
    <div><span class='figure-tier'>{escape(tier_label)}</span>
    <h3>Scientific question</h3><p>{escape(question)}</p></div>
    <div><h3>Interpretation</h3><p>{escape(interpretation)}</p></div>
  </div>
  {result_html}
  <details class='figure-methods'><summary>Methods, mappings and limitations</summary>
    <dl>{method_html}</dl>
  </details>
  <div class='figure-actions'>
    <button type='button' data-copy-caption>Copy journal caption</button>
    <button type='button' data-download-figure>Download exact plotted data (.json)</button>
    <span role='status' aria-live='polite'></span>
  </div>
</div>"""


def _scientific_audit_html(report: VRavenReport) -> str:
    """Render an evidence-led synthesis without manufacturing a trust score."""

    audit = report.scientific_audit()
    concentration = audit.concentration
    concordance = audit.concordance
    tier_label = audit.evidence_tier.replace("_", " ").title()
    effective = (
        f"{concentration.effective_layers:.2f} / {audit.gradient_layer_count}"
        if concentration.effective_layers is not None
        else "Unavailable"
    )
    half_mass = (
        f"{concentration.half_mass_layer} ({_percent(concentration.half_mass_depth)} depth)"
        if concentration.half_mass_layer
        else "Unavailable"
    )
    rho = _number(concordance.spearman_rho)
    p_value = (
        f"{concordance.permutation_p_value:.4f}"
        if concordance.permutation_p_value is not None
        else "Not estimable"
    )
    cards = (
        ("Evidence tier", tier_label, "Highest evidence type present; not a quality grade"),
        ("Effective evidence layers", effective, "exp(Shannon entropy) of absolute layer mass"),
        ("50% evidence crossing", half_mass, "Earliest depth at which cumulative mass reaches 0.5"),
        (
            "Zero-ablation coverage",
            _percent(audit.causal_coverage),
            "Uniquely matched gradient-bearing layers",
        ),
        ("Gradient ↔ ablation ρ", rho, "Spearman rank concordance on matched layers"),
        ("Permutation p", p_value, concordance.permutation_method or "Insufficient matched layers"),
    )
    card_html = "".join(
        f"<div class='audit-card'><div class='audit-label'>{escape(label)}</div>"
        f"<div class='audit-value'>{escape(value)}</div>"
        f"<div class='audit-caption'>{escape(caption)}</div></div>"
        for label, value, caption in cards
    )

    hypotheses: list[tuple[str, str, str, str]] = []
    if audit.median_prediction_margin is not None:
        hypotheses.append(
            (
                "Output separation",
                f"median class margin = {audit.median_prediction_margin:.3f}",
                "Predictive",
                "Measures output separation only; it does not validate the explanation.",
            )
        )
    if concentration.normalised_entropy is not None:
        hypotheses.append(
            (
                "Depth concentration",
                (
                    f"H/Hmax = {concentration.normalised_entropy:.3f}; "
                    f"normalised Gini = {_number(concentration.normalised_gini)}"
                ),
                "Observational",
                "Reports dispersion of absolute first-order evidence across captured depth.",
            )
        )
    if concentration.signed_cancellation is not None:
        hypotheses.append(
            (
                "Evidence cancellation",
                f"cancellation ratio = {concentration.signed_cancellation:.3f}",
                "Observational",
                "Quantifies support/opposition cancellation; high values are not uncertainty.",
            )
        )
    if concordance.layer_names:
        hypotheses.append(
            (
                "Local causal corroboration",
                (
                    f"ρ = {_number(concordance.spearman_rho)}; "
                    f"sign agreement = {_percent(concordance.sign_agreement)}; "
                    f"L1 discrepancy = {_number(concordance.normalised_l1_discrepancy)}"
                ),
                "Targeted interventional",
                "Compares scale-normalised gradients with actual zero-output ablations.",
            )
        )
    else:
        hypotheses.append(
            (
                "Local causal corroboration",
                "not tested or not uniquely matchable",
                "Unavailable",
                "Observational rankings must not be promoted to causal findings.",
            )
        )
    hypothesis_rows = "".join(
        "<tr><td>{}</td><td class='numeric'>{}</td><td><span class='evidence-tier'>{}</span>"
        "</td><td>{}</td></tr>".format(*(escape(item) for item in row))
        for row in hypotheses
    )

    causal_by_name = dict(zip(concordance.layer_names, concordance.causal_raw, strict=True))
    ledger_rows: list[str] = []
    for index, name in enumerate(concentration.layer_names):
        observed = concentration.signed_influence[index]
        causal = causal_by_name.get(name)
        if causal is None:
            status = "Not causally tested"
            status_class = "untested"
            causal_text = "—"
        elif abs(observed) <= 1e-12 or abs(causal) <= 1e-12:
            status = "Contains numerical zero"
            status_class = "untested"
            causal_text = f"{causal:.5g}"
        elif np.sign(observed) == np.sign(causal):
            status = "Direction agrees"
            status_class = "agrees"
            causal_text = f"{causal:.5g}"
        else:
            status = "Direction contradicts"
            status_class = "contradicts"
            causal_text = f"{causal:.5g}"
        ledger_rows.append(
            f"<tr><td>{index + 1}</td><td>{escape(name)}</td>"
            f"<td class='numeric'>{observed:.5g}</td>"
            f"<td class='numeric'>{concentration.mass_share[index]:.2%}</td>"
            f"<td class='numeric'>{concentration.cumulative_mass[index]:.2%}</td>"
            f"<td class='numeric'>{escape(causal_text)}</td>"
            f"<td><span class='ledger-status {status_class}'>{escape(status)}</span></td></tr>"
        )
    if not ledger_rows:
        ledger_rows.append(
            "<tr><td colspan='7'>Gradient evidence is unavailable in this trace. "
            "Activation summaries remain descriptive.</td></tr>"
        )

    findings: list[str] = []
    if concentration.effective_layers is not None:
        findings.append(
            f"Absolute first-order evidence occupies {concentration.effective_layers:.2f} "
            f"effective layers across {audit.gradient_layer_count} measured layers; "
            f"80% of its mass is reached at {concentration.eighty_mass_layer}."
        )
    if concordance.layer_names:
        findings.append(
            f"{len(concordance.layer_names)} layer effects were uniquely paired with zero "
            f"ablation: rank concordance ρ={rho}, sign agreement "
            f"{_percent(concordance.sign_agreement)}, and scale-normalised L1 discrepancy "
            f"{_number(concordance.normalised_l1_discrepancy)}."
        )
    elif audit.gradient_layer_count:
        findings.append(
            "No first-order layer effect could be uniquely paired with a zero-ablation result; "
            "the layer ranking is observational."
        )
    if audit.causal_coverage is not None:
        findings.append(
            f"Zero ablation produced {audit.zero_ablation_layer_count} unique module tests; "
            f"{len(concordance.layer_names)} of {audit.gradient_layer_count} gradient-bearing "
            f"layers were uniquely matched ({audit.causal_coverage:.1%}). All remaining "
            "layer-level causal roles are unresolved."
        )
    if audit.median_predictive_entropy is not None:
        findings.append(
            f"Median normalised predictive entropy is {audit.median_predictive_entropy:.3f} "
            f"and median class margin is {_number(audit.median_prediction_margin)}; these "
            "describe the output distribution, not explanation validity."
        )
    finding_html = "".join(f"<li>{escape(item)}</li>" for item in findings)

    limitations = [
        "Activation × gradient is a local first-order approximation and is not causal evidence.",
        "Layer summaries are batch-reduced; they do not provide per-sample confidence intervals.",
        "Zero-output ablation may produce an internal state outside the training distribution.",
        "Targeted top-k interventions leave untested layers and multi-layer interactions unresolved.",
        "The permutation p-value tests exchangeability under the matched-layer null; it is not the probability that the mechanism claim is true.",
    ]
    if concordance.excluded_nonzero_interventions:
        limitations.append(
            "Non-zero interventions were excluded from concordance because activation × gradient "
            "is specifically the first-order approximation to zero scaling."
        )
    if concordance.excluded_ambiguous_layers:
        limitations.append(
            "Reused modules with multiple execution records were excluded from matching."
        )
    limitation_html = "".join(f"<li>{escape(item)}</li>" for item in limitations)
    machine = escape(json.dumps(audit.as_dict(), indent=2))
    return f"""
<section class='scientific-audit' aria-labelledby='scientific-audit-title'>
  <div class='audit-kicker'>Inference before illustration</div>
  <h2 id='scientific-audit-title'>Scientific Evidence Audit</h2>
  <p class='audit-lede'>This section separates prediction, observational evidence and intervention
  evidence. It reports complementary quantities directly and deliberately does not manufacture a
  single explanation-confidence or trust score.</p>
  <div class='audit-grid'>{card_html}</div>
  <h3>Critical findings</h3><ul class='finding-list'>{finding_html or '<li>No inferential findings are available.</li>'}</ul>
  <h3>Hypothesis and evidence ledger</h3>
  <div class='table-scroll'><table class='audit-table'><thead><tr><th>Question</th><th>Estimate</th>
  <th>Evidence tier</th><th>Interpretation boundary</th></tr></thead><tbody>{hypothesis_rows}</tbody></table></div>
  <h3>Layer evidence ledger</h3>
  <div class='table-scroll'><table class='audit-table'><thead><tr><th>Depth</th><th>Layer</th>
  <th>Signed first-order</th><th>Absolute mass</th><th>Cumulative mass</th>
  <th>Zero-ablation Δ</th><th>Cross-check</th></tr></thead><tbody>{''.join(ledger_rows)}</tbody></table></div>
  <details class='audit-details'><summary>Mathematical definitions, assumptions and machine-readable audit</summary>
  <div class='equations'>
    <p><code>p_l = ||a_l ⊙ ∂s_t/∂a_l||₁ / Σ_j ||a_j ⊙ ∂s_t/∂a_j||₁</code></p>
    <p><code>N_eff = exp(−Σ_l p_l log p_l)</code>; participation ratio <code>= 1/Σ_l p_l²</code>.</p>
    <p>Concordance compares <code>Σ(a_l ⊙ ∂s_t/∂a_l)</code> with measured
    <code>s_t(x) − s_t(do(a_l=0))</code> after separate signed L1 normalisation.</p>
  </div><h4>Known limitations</h4><ul>{limitation_html}</ul>
  <h4>Machine-readable values</h4><pre>{machine}</pre></details>
</section>"""


def _scientific_audit_css() -> str:
    return """
.scientific-audit { border-top: 3px solid var(--vraven-accent); }
.audit-kicker { color: var(--vraven-accent); font-size: 11px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 7px; }
.audit-lede { max-width: 92ch; margin: -4px 0 20px; }
.scientific-audit h3 { margin: 26px 0 10px; font-size: 14px; color: var(--vraven-text); letter-spacing: .02em; }
.scientific-audit h4 { margin: 20px 0 8px; color: var(--vraven-text); }
.audit-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; }
.audit-card { min-width: 0; padding: 15px 16px; border: 1px solid var(--vraven-border); border-radius: 12px; background: var(--vraven-raised, var(--vraven-panel)); }
.audit-label { color: var(--vraven-muted); text-transform: uppercase; letter-spacing: .07em; font-size: 10px; font-weight: 700; }
.audit-value { margin: 7px 0 5px; color: var(--vraven-text); font-family: "JetBrains Mono", monospace; font-size: 19px; font-weight: 700; overflow-wrap: anywhere; }
.audit-caption { color: var(--vraven-muted); font-size: 11px; line-height: 1.45; }
.finding-list { margin: 8px 0 0; padding-left: 20px; color: var(--vraven-text); }
.finding-list li { margin: 7px 0; line-height: 1.55; }
.table-scroll { overflow-x: auto; border: 1px solid var(--vraven-border); border-radius: 10px; }
.audit-table { min-width: 760px; margin: 0; }
.audit-table th { width: auto; white-space: nowrap; background: var(--vraven-raised, var(--vraven-panel)); }
.audit-table tr:last-child td { border-bottom: 0; }
.numeric { font-variant-numeric: tabular-nums; white-space: nowrap; }
.evidence-tier,.ledger-status { display: inline-block; padding: 3px 7px; border: 1px solid var(--vraven-border); border-radius: 7px; font-size: 10px; font-weight: 700; white-space: nowrap; }
.ledger-status.agrees { color: #32D74B; border-color: #32D74B; }
html[data-theme='light'] .ledger-status.agrees { color: #15803D; border-color: #15803D; }
.ledger-status.contradicts { color: #FF2D55; border-color: #FF2D55; }
html[data-theme='light'] .ledger-status.contradicts { color: #B42318; border-color: #B42318; }
.ledger-status.untested { color: var(--vraven-muted); }
.audit-details { margin-top: 24px; padding: 14px 16px; border: 1px solid var(--vraven-border); border-radius: 10px; }
.audit-details summary { cursor: pointer; color: var(--vraven-text); font-weight: 700; }
.audit-details pre { max-height: 440px; overflow: auto; padding: 14px; border-radius: 8px; background: var(--vraven-raised, var(--vraven-panel)); color: var(--vraven-text); font-size: 11px; }
.equations code { color: var(--vraven-text); font-family: "JetBrains Mono", monospace; }
.figure-evidence { margin: 0 0 14px; padding: 14px 16px; border: 1px solid var(--vraven-border); border-left: 3px solid var(--vraven-accent); border-radius: 10px; background: var(--vraven-raised, var(--vraven-panel)); }
.figure-evidence-head { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 18px; }
.figure-evidence h3 { margin: 8px 0 4px; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--vraven-text); }
.figure-evidence p { margin: 0; max-width: none; font-size: 12px; line-height: 1.5; }
.figure-tier { display: inline-block; padding: 3px 7px; border: 1px solid var(--vraven-accent); border-radius: 999px; color: var(--vraven-accent); font-size: 10px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
.figure-result-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 8px; margin-top: 12px; }
.figure-result { min-width: 0; padding: 10px 11px; border: 1px solid var(--vraven-border); border-radius: 8px; background: var(--vraven-panel); }
.figure-result-label { color: var(--vraven-muted); font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.figure-result-value { margin: 4px 0; color: var(--vraven-text); font-family: "JetBrains Mono", monospace; font-size: 15px; font-weight: 700; overflow-wrap: anywhere; }
.figure-result-value span { color: var(--vraven-muted); font-family: "Inter", sans-serif; font-size: 10px; font-weight: 500; }
.figure-result-note { color: var(--vraven-muted); font-size: 10px; line-height: 1.4; }
.figure-methods { margin-top: 10px; border-top: 1px solid var(--vraven-border); padding-top: 8px; }
.figure-methods summary { cursor: pointer; color: var(--vraven-text); font-size: 11px; font-weight: 700; }
.figure-methods dl { display: grid; grid-template-columns: minmax(130px,.25fr) minmax(0,1fr); gap: 6px 12px; margin: 10px 0 0; font-size: 11px; }
.figure-methods dt { color: var(--vraven-muted); font-weight: 700; }
.figure-methods dd { margin: 0; color: var(--vraven-text); }
.figure-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin-top: 10px; }
.figure-actions button { appearance: none; border: 1px solid var(--vraven-border); border-radius: 7px; padding: 6px 9px; background: transparent; color: var(--vraven-text); font: 600 10px "Inter", sans-serif; cursor: pointer; }
.figure-actions button:hover { border-color: var(--vraven-accent); color: var(--vraven-accent); }
.figure-actions span { color: var(--vraven-muted); font-size: 10px; }
@media (max-width: 900px) { .audit-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } }
@media (max-width: 720px) { .figure-evidence-head,.figure-result-grid { grid-template-columns: 1fr; } }
@media (max-width: 620px) { .audit-grid { grid-template-columns: 1fr; } .figure-methods dl { grid-template-columns: 1fr; } }
"""


def _figure_html(
    figure: go.Figure,
    *,
    include_plotlyjs: bool,
) -> str:
    return plotly_io.to_html(
        figure,
        full_html=False,
        include_plotlyjs=bool(include_plotlyjs),
        config={"responsive": True, "displaylogo": False},
    )


def dual_mode_section(
    title: str,
    *,
    dark_figure: go.Figure,
    light_figure: go.Figure,
    include_plotlyjs: bool = False,
) -> str:
    """Build one report section with paired dark/light Plotly panels."""

    dark_html = _figure_html(dark_figure, include_plotlyjs=include_plotlyjs)
    light_html = _figure_html(light_figure, include_plotlyjs=False)
    evidence_html = _publication_evidence_html(dark_figure, title)
    return (
        f"<section data-vraven-section>"
        f"<h2>{escape(title)}</h2>"
        f"{evidence_html}"
        f'<div class="viz-panel" data-theme="dark">{dark_html}</div>'
        f'<div class="viz-panel" data-theme="light">{light_html}</div>'
        f"</section>"
    )


def _resolve_title(visual: str, figure: go.Figure | None = None) -> str:
    title = TITLE_MAP.get(visual, visual.replace("_", " ").title())
    if visual == "input_attribution" and figure is not None:
        presentation = ((figure.layout.meta or {}).get("scientific") or {}).get(
            "presentation"
        )
        if presentation != "named_feature_bars":
            title = "Local Input Attribution"
    return title


def _theme_css(*, theme_toggle: bool, initial_theme: str) -> str:
    dark = THEMES["interactive_dark"]
    light = THEMES["publication_light"]
    active = dark if initial_theme == "dark" else light
    active_accent = active.semantic["selected"]
    active_glow = (
        "rgba(0, 212, 255, 0.15)"
        if initial_theme == "dark"
        else "rgba(14, 116, 144, 0.12)"
    )
    active_accent_gradient = (
        "linear-gradient(135deg, #00D4FF 0%, #8B5CF6 50%, #FF2D55 100%)"
        if initial_theme == "dark"
        else "linear-gradient(135deg, #0057B8 0%, #6D28D9 50%, #B42318 100%)"
    )
    if not theme_toggle:
        return f"""
:root {{
  color-scheme: {'dark' if initial_theme == 'dark' else 'light'};
  --vraven-canvas: {active.canvas};
  --vraven-panel: {active.panel};
  --vraven-raised: {active.raised_panel};
  --vraven-border: {active.border};
  --vraven-text: {active.primary_text};
  --vraven-muted: {active.secondary_text};
  --vraven-accent: {active_accent};
  --vraven-accent-gradient: {active_accent_gradient};
  --vraven-glow: {active_glow};
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 32px 28px;
  background: var(--vraven-canvas);
  background-image: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(22, 139, 255, 0.12), transparent 70%), radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139, 92, 246, 0.08), transparent 60%);
  background-attachment: fixed;
  color: var(--vraven-text);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}}
main {{ max-width: 1280px; margin: 0 auto; }}
header, section {{
  background: var(--vraven-panel);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--vraven-border);
  box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px 0 var(--vraven-glow);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 24px;
}}
h1 {{
  margin: 0 0 10px 0;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: var(--vraven-accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
h2 {{
  margin: 0 0 16px 0;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--vraven-accent);
  border-bottom: 1px solid var(--vraven-border);
  padding-bottom: 10px;
}}
p {{ color: var(--vraven-muted); line-height: 1.6; font-size: 14px; }}
table {{ border-collapse: separate; border-spacing: 0; width: 100%; margin-top: 12px; }}
th, td {{
  text-align: left;
  border-bottom: 1px solid var(--vraven-border);
  padding: 12px 14px;
  font-size: 13px;
}}
th {{ width: 30%; color: var(--vraven-muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }}
td {{ font-family: "JetBrains Mono", monospace; color: var(--vraven-text); }}
"""

    return """

:root, html[data-theme="dark"] {
  color-scheme: dark;
  --vraven-canvas: #030A12;
  --vraven-panel: #07111F;
  --vraven-raised: #0B1728;
  --vraven-border: #1C344D;
  --vraven-border-glow: rgba(0, 212, 255, 0.3);
  --vraven-text: #F1F5F9;
  --vraven-muted: #94A3B8;
  --vraven-accent: #00D4FF;
  --vraven-accent-gradient: linear-gradient(135deg, #00D4FF 0%, #8B5CF6 50%, #FF2D55 100%);
  --vraven-toggle-bg: #0B1728;
  --vraven-toggle-active: #00D4FF;
  --vraven-shadow: 0 12px 36px -10px rgba(0, 0, 0, 0.6), 0 0 20px 0 rgba(0, 212, 255, 0.05);
}
html[data-theme="light"] {
  color-scheme: light;
  --vraven-canvas: #FFFFFF;
  --vraven-panel: #FFFFFF;
  --vraven-raised: #F1F5F9;
  --vraven-border: #CBD5E1;
  --vraven-border-glow: rgba(14, 116, 144, 0.3);
  --vraven-text: #111827;
  --vraven-muted: #475569;
  --vraven-accent: #0E7490;
  --vraven-accent-gradient: linear-gradient(135deg, #0057B8 0%, #6D28D9 50%, #B42318 100%);
  --vraven-toggle-bg: #F1F5F9;
  --vraven-toggle-active: #0E7490;
  --vraven-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 24px 64px;
  background: var(--vraven-canvas);
  background-image: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(22, 139, 255, 0.12), transparent 70%), radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139, 92, 246, 0.08), transparent 60%);
  background-attachment: fixed;
  color: var(--vraven-text);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
  transition: background 200ms ease, color 200ms ease;
}
main { max-width: 1360px; margin: 0 auto; }
header, section, .card {
  background: var(--vraven-panel);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--vraven-border);
  box-shadow: var(--vraven-shadow);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 24px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
header {
  position: relative;
  overflow: hidden;
}
header::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--vraven-accent-gradient);
}
.header-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  flex-wrap: wrap;
}
.header-copy { flex: 1 1 300px; }
h1 {
  margin: 0 0 10px 0;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: clamp(22px, 3vw, 28px);
  font-weight: 800;
  letter-spacing: -0.02em;
  background: var(--vraven-accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.2;
}
h2 {
  margin: 0 0 18px 0;
  padding: 0 0 12px 0;
  border-bottom: 1px solid var(--vraven-border);
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--vraven-accent);
  line-height: 1.35;
}
p { color: var(--vraven-muted); line-height: 1.6; font-size: 14px; }
table { border-collapse: separate; border-spacing: 0; width: 100%; }
th, td {
  text-align: left;
  border-bottom: 1px solid var(--vraven-border);
  padding: 12px 14px;
  vertical-align: top;
  overflow-wrap: anywhere;
  font-size: 13px;
}
th { width: 30%; color: var(--vraven-muted); font-weight: 600; font-family: "Inter", sans-serif; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; }
td { font-family: "JetBrains Mono", monospace; font-size: 12px; }
.note { font-size: 13px; color: var(--vraven-muted); line-height: 1.6; max-width: 76ch; }
.theme-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px;
  border: 1px solid var(--vraven-border);
  border-radius: 12px;
  background: var(--vraven-toggle-bg);
  flex: 0 0 auto;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}
.theme-toggle button {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--vraven-muted);
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 12px;
  font-weight: 700;
  padding: 8px 16px;
  border-radius: 9px;
  cursor: pointer;
  transition: all 180ms ease;
}
.theme-toggle button[aria-pressed="true"] {
  background: var(--vraven-accent-gradient);
  color: #FFFFFF;
  box-shadow: 0 4px 12px var(--vraven-border-glow);
}
.viz-panel {
  width: 100%;
  overflow: visible;
  margin-top: 8px;
  border-radius: 12px;
}
.viz-panel .js-plotly-plot,
.viz-panel .plotly-graph-div {
  width: 100% !important;
  border-radius: 12px;
}
.viz-panel[hidden] { display: none !important; }
@media (max-width: 720px) {
  body { padding: 16px; }
  .theme-toggle { width: 100%; justify-content: center; }
  h2 { letter-spacing: 0.04em; }
}
"""



def _theme_script(initial_theme: str) -> str:
    return f"""
<script>
(function () {{
  var STORAGE_KEY = "vraven-report-theme";
  var initial = {initial_theme!r};
  function storedTheme() {{
    try {{ return localStorage.getItem(STORAGE_KEY); }} catch (e) {{ return null; }}
  }}
  function currentTheme() {{
    return document.documentElement.getAttribute("data-theme") || initial;
  }}
  function resizeVisiblePlots() {{
    if (typeof Plotly !== "undefined") {{
      document.querySelectorAll('.viz-panel:not([hidden]) .js-plotly-plot').forEach(function (gd) {{
        try {{ Plotly.Plots.resize(gd); }} catch (e) {{}}
      }});
    }}
  }}
  function setTheme(theme) {{
    var next = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    document.querySelectorAll(".viz-panel").forEach(function (panel) {{
      panel.hidden = panel.getAttribute("data-theme") !== next;
    }});
    document.querySelectorAll("[data-theme-choice]").forEach(function (button) {{
      button.setAttribute(
        "aria-pressed",
        button.getAttribute("data-theme-choice") === next ? "true" : "false"
      );
    }});
    try {{ localStorage.setItem(STORAGE_KEY, next); }} catch (e) {{}}
    window.requestAnimationFrame(resizeVisiblePlots);
  }}
  document.addEventListener("DOMContentLoaded", function () {{
    setTheme(storedTheme() || initial);
    document.querySelectorAll("[data-theme-choice]").forEach(function (button) {{
      button.addEventListener("click", function () {{
        setTheme(button.getAttribute("data-theme-choice"));
      }});
    }});
    window.addEventListener("resize", resizeVisiblePlots);
  }});
  // Apply early to avoid a flash of the wrong chrome colors.
  setTheme(storedTheme() || initial);
}})();
</script>
"""


def _publication_action_script() -> str:
    return """
<script>
(function () {
  function setStatus(button, message) {
    var status = button.closest('.figure-actions').querySelector('[role="status"]');
    if (status) status.textContent = message;
  }
  function visiblePanel(section) {
    return section.querySelector('.viz-panel:not([hidden])') || section;
  }
  function dataPayload(section) {
    var panel = visiblePanel(section);
    var graph = panel.querySelector('.js-plotly-plot');
    if (graph && graph.data) {
      return {
        engine: 'plotly',
        scientific_metadata: graph.layout && graph.layout.meta
          ? graph.layout.meta.scientific : null,
        traces: graph.data
      };
    }
    throw new Error('No rendered figure data found');
  }
  function jsonReplacer(_key, value) {
    if (value && ArrayBuffer.isView(value)) return Array.from(value);
    return value;
  }
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-copy-caption]').forEach(function (button) {
      button.addEventListener('click', async function () {
        var evidence = button.closest('.figure-evidence');
        var caption = evidence.getAttribute('data-publication-caption') || '';
        try {
          await navigator.clipboard.writeText(caption);
          setStatus(button, 'Caption copied');
        } catch (_error) {
          var area = document.createElement('textarea');
          area.value = caption;
          document.body.appendChild(area);
          area.select();
          document.execCommand('copy');
          area.remove();
          setStatus(button, 'Caption copied');
        }
      });
    });
    document.querySelectorAll('[data-download-figure]').forEach(function (button) {
      button.addEventListener('click', function () {
        var section = button.closest('[data-vraven-section]');
        var evidence = button.closest('.figure-evidence');
        try {
          var payload = dataPayload(section);
          var blob = new Blob([JSON.stringify(payload, jsonReplacer, 2)], {type: 'application/json'});
          var url = URL.createObjectURL(blob);
          var link = document.createElement('a');
          link.href = url;
          link.download = (evidence.getAttribute('data-figure-id') || 'vraven-figure') + '.json';
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
          setStatus(button, 'Exact data downloaded');
        } catch (error) {
          setStatus(button, 'Data export unavailable: ' + error.message);
        }
      });
    });
  });
})();
</script>
"""


def export_html_report(
    report: VRavenReport,
    path: str | Path,
    *,
    visuals: list[str] | tuple[str, ...] | None = None,
    mode: RenderMode = "interactive_dark",
    figure_kwargs: dict[str, dict[str, Any]] | None = None,
    theme_toggle: bool = True,
    include_scientific_audit: bool = True,
) -> Path:
    """Compose requested standalone figures. This is opt-in and not the default output.

    When ``theme_toggle`` is True (default), each figure is rendered in both
    ``interactive_dark`` and ``publication_light`` and the report header exposes a
    Dark / Light control. Preference is remembered in ``localStorage``.
    """

    selected = tuple(visuals or DEFAULT_VISUAL_ORDER)
    capabilities = report.capabilities()
    per_figure = figure_kwargs or {}
    fragments: list[str] = []
    included_plotly = False
    initial_theme = _MODE_TO_THEME_KEY.get(mode, "dark")
    alt_mode: RenderMode = (
        "publication_light" if mode == "interactive_dark" else "interactive_dark"
    )

    for visual in selected:
        if visual in capabilities and not capabilities[visual]:
            continue
        kwargs = dict(per_figure.get(visual, {}))
        # HTML section headings own the title; suppress in-chart titles so they
        # cannot stack on / collide with the report h2.
        kwargs.setdefault("show_title", False)
        title = _resolve_title(visual)

        try:
            primary = report.figure(visual, mode=mode, **kwargs)
        except (CapabilityError, ValueError, TypeError):
            continue
        title = _resolve_title(visual, primary)
        if theme_toggle:
            try:
                secondary = report.figure(visual, mode=alt_mode, **kwargs)
            except (CapabilityError, ValueError, TypeError):
                # Fall back to a single-mode panel if the alternate render fails.
                fragment = _figure_html(primary, include_plotlyjs=not included_plotly)
                included_plotly = True
                fragments.append(
                    f"<section data-vraven-section><h2>{escape(title)}</h2>"
                    f"{_publication_evidence_html(primary, title)}{fragment}</section>"
                )
                continue
            dark_fig = primary if mode == "interactive_dark" else secondary
            light_fig = secondary if mode == "interactive_dark" else primary
            fragments.append(
                dual_mode_section(
                    title,
                    dark_figure=dark_fig,
                    light_figure=light_fig,
                    include_plotlyjs=not included_plotly,
                )
            )
            included_plotly = True
        else:
            fragment = _figure_html(primary, include_plotlyjs=not included_plotly)
            included_plotly = True
            fragments.append(
                f"<section data-vraven-section><h2>{escape(title)}</h2>"
                f"{_publication_evidence_html(primary, title)}{fragment}</section>"
            )

    toggle_markup = ""
    if theme_toggle:
        toggle_markup = """
<div class="theme-toggle" role="group" aria-label="Colour theme">
  <button type="button" data-theme-choice="dark" aria-pressed="true">Dark</button>
  <button type="button" data-theme-choice="light" aria-pressed="false">Light</button>
</div>
"""

    html = f"""<!doctype html>
<html lang="en" data-theme="{initial_theme}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Self-contained VRAVEN explanation report with explicit evidence boundaries.">
<title>VRAVEN trace {escape(report.trace.trace_id)}</title>
<style>{_theme_css(theme_toggle=theme_toggle, initial_theme=initial_theme)}{_scientific_audit_css()}</style>
{_theme_script(initial_theme) if theme_toggle else ""}
{_publication_action_script()}
</head>
<body><main>
<header>
<div class="header-row">
  <div class="header-copy">
    <h1>VRAVEN Explanation Report</h1>
    <p>Trace {escape(report.trace.trace_id)}</p>
  </div>
  {toggle_markup}
</div>
<table>{_summary_rows(report.summary())}</table>
<p class="note">Figures are standalone scientific views explicitly composed into this local report.
Signal statistics and first-order gradients must not be interpreted as causal unless an intervention view is present.
{"Use the Dark / Light control to switch page chrome and figure themes." if theme_toggle else ""}</p>
</header>
{_scientific_audit_html(report) if include_scientific_audit else ""}
{''.join(fragments) if fragments else '<section>No compatible visualisations were available.</section>'}
</main></body></html>"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
