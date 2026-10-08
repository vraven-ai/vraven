#!/usr/bin/env python3
"""Build the public VRAVEN output, Python API, and CLI reference pages.

The reference is deliberately generated from the checked-out package and CLI
parser. That keeps signatures and command options tied to the code rather than
to a hand-maintained marketing page.
"""

from __future__ import annotations

import argparse
import html
import importlib
import inspect
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class Output:
    family: str
    title: str
    image: str
    question: str
    call: str
    result: str
    boundary: str
    machine: str
    api: str


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def code(value: str) -> str:
    return f"<pre><code>{esc(value)}</code></pre>"


def cli_bundle_command() -> str:
    return """vraven analyse ./my-model \\
  --task explain \\
  --preset standard \\
  --format html \\
  --output vraven-report.html"""


def output(
    family: str,
    title: str,
    image: str,
    question: str,
    call: str,
    result: str,
    boundary: str,
    machine: str,
    api: str,
) -> Output:
    return Output(family, title, image, question, call, result, boundary, machine, api)


TRACE_CALLS = {
    "Activation Energy": 'report.figure("activation_energy", mode="publication_light")',
    "Attention Structure": 'report.figure("attention_structure", mode="publication_light")',
    "Causal Pathway": 'report.figure("causal_pathway", mode="publication_light")',
    "Conflict Field": 'report.figure("conflict_field", mode="publication_light")',
    "Decision Flow": 'report.figure("decision_flow", mode="publication_light")',
    "Decision Sankey": 'report.figure("decision_sankey", mode="publication_light")',
    "Evidence Trajectory": 'report.figure("evidence_trajectory", mode="publication_light")',
    "Feature Sequence Heatmap": 'report.figure("feature_sequence_heatmap", mode="publication_light")',
    "Input Attribution": 'report.figure("input_attribution", mode="publication_light")',
    "Mechanism Concordance": 'report.figure("mechanism_concordance", mode="publication_light")',
    "Neuron Impact": 'report.figure("neuron_impact", mode="publication_light")',
    "Neuron Importance": 'report.figure("neuron_importance", mode="publication_light")',
    "Reasoning Dna": 'report.figure("reasoning_dna", mode="publication_light")',
    "Reasoning Graph": 'report.figure("reasoning_graph", mode="publication_light")',
    "Spectral Trace": 'report.figure("spectral_trace", mode="publication_light")',
}


def output_catalogue() -> list[Output]:
    common_boundary = "This is evidence for this input and model state, not a universal statement about every input."
    items = [
        output("Reverse", "Decision Proof report", "decision_proof_report.png", "Can the prediction be expressed as an auditable decision mechanism?", 'reverse.write_report(output_path="vraven_decision_proof.html")', "A joined report of the recovered mechanism, interventions, and decision evidence.", "A recovered programme is an approximation whose fidelity must be checked against the source model.", "reverse/vraven_decision_proof.html", "vraven.decompiler.ReverseEngineeringPipeline"),
        output("Reverse", "Adversarial Mechanism Audit", "adversarial_audit_report.png", "Which mechanism changes under bounded adversarial pressure?", "AdversarialAuditor(model).audit_sample(input_tensor, target=target).write_html(output_path)", "A perturbation-by-perturbation view of prediction and internal mechanism stability.", common_boundary, "reverse/adversarial_audit.html", "vraven.AdversarialAuditor"),
        output("Reverse", "Formal Robustness Certificate", "formal_robustness_certificate_report.png", "Can the local prediction be bounded formally inside the selected region?", "IntervalBoundVerifier(model).certify(input_tensor, target=target).write_html(output_path)", "An interval-bound certificate with its assumptions and verified radius.", "A certificate applies only to its stated norm, radius, model, input, and verifier assumptions.", "reverse/formal_robustness_certificate.html", "vraven.IntervalBoundVerifier"),
        output("Reverse", "Formal Causal-Mechanism Certificate", "formal_causal_mechanism_certificate_report.png", "Which recovered causal statements can be certified under the chosen intervention model?", "FormalCausalMechanismVerifier(model).certify(input_tensor, target=target).write_html(output_path)", "A formal record of the mechanism claims that passed verification.", "The certificate does not validate causal claims outside the encoded intervention assumptions.", "reverse/formal_causal_mechanism_certificate.html", "vraven.FormalCausalMechanismVerifier"),
        output("Reverse", "Empirical Robustness Profile", "adversarial_robustness_profile_report.png", "How quickly does the decision degrade as perturbation strength increases?", "AdversarialAuditor(model).audit_robustness_curve(input_tensor, target=target).write_html(output_path)", "An empirical curve linking perturbation strength to prediction and mechanism behaviour.", "Empirical sampling can miss attacks; it is not a formal robustness guarantee.", "reverse/adversarial_robustness_profile.html", "vraven.AdversarialAuditor"),
    ]

    reverse_views = [
        ("Raven Brain · causal proof", "causal_proof_publication_light.png", "Which recovered links carry causal evidence?", "causal_proof"),
        ("Raven Flight · concept genome", "concept_genome_publication_light.png", "How do internal concepts appear and evolve through the network?", "concept_genome"),
        ("Raven Contrast", "contrastive_proof_publication_light.png", "Why this decision rather than the strongest alternative?", "contrastive_proof"),
        ("Raven Shadow · counterfactual", "counterfactual_twin_publication_light.png", "What smallest tested change changes the decision?", "counterfactual_twin"),
        ("Raven Dark Matter", "dark_matter_publication_light.png", "Which internal activity is present but absent from the recovered programme?", "dark_matter"),
        ("Raven Eye · model X-ray", "model_xray_publication_light.png", "What is the structural and activation profile of the model?", "model_xray"),
        ("Raven Code · programme proof", "programme_proof_publication_light.png", "How faithfully does the executable programme reproduce the model?", "programme_proof"),
    ]
    for title, image, question, name in reverse_views:
        items.append(output("Reverse", title, image, question, f'reverse.write_feature_report("{name}", output_path, initial_mode="publication_light")', "A publication-light view extracted from the reverse-engineering result.", common_boundary, f"reverse/{name}_publication_light.html", "vraven.decompiler.ReverseEngineeringResult"))

    proof_specs = [
        ("Explanation report", "proof_report.png", "What evidence supports this single prediction?", 'report.write_report("vraven_proof.html")', "The complete single-prediction explanation report."),
        ("Activation Energy", "activation_energy_publication_light.png", "Where is activation energy concentrated?", TRACE_CALLS["Activation Energy"], "Relative activation intensity by captured layer."),
        ("Attention Structure", "attention_structure_publication_light.png", "Which spatial or token relationships receive attention?", TRACE_CALLS["Attention Structure"], "The attention structure available in the captured trace."),
        ("Causal Pathway", "causal_pathway_publication_light.png", "Which internal path most supports the selected output?", TRACE_CALLS["Causal Pathway"], "A traced pathway through influential internal components."),
        ("Conflict Field", "conflict_field_publication_light.png", "Where does evidence for competing classes conflict?", TRACE_CALLS["Conflict Field"], "Regions or components where class evidence opposes."),
        ("Counterfactual Delta", "counterfactual_delta_publication_light.png", "What changed between the original and counterfactual?", 'counterfactual.figure("counterfactual_delta", mode="publication_light")', "The measured change associated with the tested counterfactual."),
        ("Decision Flow", "decision_flow_publication_light.png", "How does evidence move toward the final decision?", TRACE_CALLS["Decision Flow"], "A directed summary of evidence flow across the trace."),
        ("Decision Sankey", "decision_sankey_publication_light.png", "How much evidence flows between major stages?", TRACE_CALLS["Decision Sankey"], "Weighted evidence transfers between stages."),
        ("Evidence Trajectory", "evidence_trajectory_publication_light.png", "How does target evidence evolve across depth?", TRACE_CALLS["Evidence Trajectory"], "The selected target's evidence trajectory through layers."),
        ("Feature Sequence Heatmap", "feature_sequence_heatmap_publication_light.png", "Which features dominate at each captured step?", TRACE_CALLS["Feature Sequence Heatmap"], "Feature intensity across the captured sequence."),
        ("Input Attribution", "input_attribution_publication_light.png", "Which input regions support or oppose the prediction?", TRACE_CALLS["Input Attribution"], "Signed attribution over the input."),
        ("Mechanism Concordance", "mechanism_concordance_publication_light.png", "Do independent explanation signals agree?", TRACE_CALLS["Mechanism Concordance"], "Agreement among available mechanism signals."),
        ("Neuron Impact", "neuron_impact_publication_light.png", "Which unit interventions most change the output?", TRACE_CALLS["Neuron Impact"], "Measured output change from tested unit interventions."),
        ("Neuron Importance", "neuron_importance_publication_light.png", "Which units rank highest for this prediction?", TRACE_CALLS["Neuron Importance"], "A local importance ranking for captured units."),
        ("Perturbation Stability", "perturbation_stability_publication_light.png", "Does the explanation remain stable under tested perturbations?", 'stability.figure("perturbation_stability", mode="publication_light")', "Similarity of explanation evidence across sampled perturbations."),
        ("Reasoning Dna", "reasoning_dna_publication_light.png", "What compact signature describes this prediction path?", TRACE_CALLS["Reasoning Dna"], "A compact fingerprint of the captured reasoning pattern."),
        ("Reasoning Graph", "reasoning_graph_publication_light.png", "Which internal components form the reasoning graph?", TRACE_CALLS["Reasoning Graph"], "A graph of influential captured components and relationships."),
        ("Relevance Flow", "relevance_flow_publication_light.png", "How does relevance propagate back toward the input?", 'lrp_result.figure("relevance_flow", mode="publication_light")', "Layer-wise relevance propagation for the selected target."),
        ("Spectral Trace", "spectral_trace_publication_light.png", "What frequency structure appears in internal evidence?", TRACE_CALLS["Spectral Trace"], "A spectral view of captured activation or attribution structure."),
    ]
    for title, image, question, call, result in proof_specs:
        items.append(output("Proof", title, image, question, call, result, common_boundary, f"proof/{slug(title).replace('-', '_')}.html", "vraven.VRavenReport"))

    atlas = [
        ("Activation Community Graph", "activation_community_graph_publication_light.png", "Which samples share activation communities?", 'atlas.figure("activation_community_graph", mode="publication_light")'),
        ("Reasoning landscape", "reasoning_landscape_publication_light.png", "How are sample-level reasoning signatures arranged?", 'atlas.figure("reasoning_landscape", mode="publication_light")'),
        ("Reasoning outliers", "reasoning_outliers_publication_light.png", "Which samples use atypical reasoning patterns?", 'atlas.figure("reasoning_outliers", mode="publication_light")'),
        ("Reasoning similarity", "reasoning_similarity_publication_light.png", "Which predictions use similar internal reasoning?", 'atlas.figure("reasoning_similarity", mode="publication_light")'),
    ]
    for title, image, question, call in atlas:
        items.append(output("Atlas", title, image, question, call, "A cohort-level map derived from per-sample reasoning signatures.", "Cohort geometry depends on sampling, representation, and distance choices.", f"atlas/{slug(title).replace('-', '_')}.html", "vraven.DatasetAtlas"))

    items.append(output("Watch", "Model evolution", "model_evolution_publication_light.png", "How does model behaviour change across checkpoints?", 'evolution.figure("model_evolution", mode="publication_light")', "A checkpoint-ordered view of prediction and mechanism change.", "Checkpoint differences may reflect training data, optimisation, or stochasticity; the plot alone does not identify cause.", "watch/model_evolution_publication_light.html", "vraven.ModelEvolution"))
    items.append(output("Compare", "Model comparison", "model_comparison_publication_light.png", "How do matched models differ on the same evidence?", 'comparison.figure("model_comparison", mode="publication_light")', "A matched comparison of model outputs and internal evidence.", "A fair claim requires matched inputs, targets, preprocessing, and compatible capture points.", "compare/model_comparison_publication_light.html", "vraven.ModelComparison"))

    darkside_names = [
        ("Activation Mediation", "activation_mediation_interactive_dark.png", "Which intermediate activations mediate the output?", "analyse_activation_mediation", "model, clean_inputs, corrupted_inputs, target=target, layer_names=layer_names"),
        ("Adversarial Rerouting", "adversarial_rerouting_interactive_dark.png", "Does an attack reroute the internal decision path?", "analyse_adversarial_rerouting", "clean_traces, altered_traces"),
        ("Causal Synergy", "causal_synergy_interactive_dark.png", "Which component pairs have non-additive causal effects?", "analyse_causal_synergy", "model, inputs, target=target, groups=unit_groups"),
        ("Circuit Multiplicity", "circuit_multiplicity_interactive_dark.png", "Does the model retain multiple routes to the same output?", "analyse_circuit_multiplicity", "circuit_masks"),
        ("Circuit Wiring", "circuit_wiring_publication_light.png", "How are influential internal units wired?", "build_circuit_wiring", "model, inputs, circuitry"),
        ("Class-conditional Internals", "class_conditional_internals_interactive_dark.png", "How do internal states differ by class?", "compare_class_conditionals", "trace_dataset, class_a=0, class_b=1"),
        ("Class-conditional Unit Field", "class_conditional_unit_field_interactive_dark.png", "Which units are selectively associated with classes?", "map_class_conditional_unit_field", "trace_dataset, class_a=0, class_b=1"),
        ("Cost–Evidence Trade-off", "cost_evidence_tradeoff_interactive_dark.png", "How much evidence is retained at each explanation cost?", "analyse_cost_evidence_tradeoff", "trace_dataset"),
        ("Decision Circuitry", "decision_circuitry_interactive_dark.png", "Which circuit supports the selected decision?", "discover_decision_circuitry", "model, inputs, target=target"),
        ("Explanation Uncertainty", "explanation_uncertainty_interactive_dark.png", "How uncertain are the explanation estimates?", "analyse_explanation_uncertainty", "repeated_attributions"),
        ("High-confidence OOD", "high_confidence_ood_interactive_dark.png", "Where is the model confident outside familiar internal regions?", "analyse_high_confidence_ood", "reference_fingerprints, query_fingerprints, query_confidence=confidence"),
        ("Internal Anatomy", "internal_anatomy_interactive_dark.png", "What are the model's major internal structural regions?", "inspect_internal_anatomy", "model, inputs, target=target"),
        ("Mechanism Transfer", "mechanism_transfer_interactive_dark.png", "Does the mechanism transfer across matched samples?", "analyse_mechanism_transfer", "effect_vectors"),
        ("Prediction Depth", "prediction_depth_interactive_dark.png", "At what depth does the decision become recoverable?", "analyse_prediction_depth", "reference_traces, query_traces"),
        ("Representation Drift", "representation_drift_interactive_dark.png", "How do representations drift across checkpoints or conditions?", "analyse_representation_drift", "reference_traces, comparison_traces"),
        ("Shortcut Reliance", "shortcut_reliance_interactive_dark.png", "Does the model rely on a tested shortcut signal?", "detect_shortcut_reliance", "model, inputs, target=target, feature_groups=feature_groups"),
        ("Stability Geometry", "stability_geometry_interactive_dark.png", "What is the geometry of local explanation stability?", "analyse_stability_geometry", "reference_traces, perturbed_traces"),
        ("Topological Shift", "topological_shift_interactive_dark.png", "Does internal topology change across conditions?", "analyse_topological_shift", "reference_traces, comparison_traces"),
        ("Unused Capacity", "unused_capacity_interactive_dark.png", "Which captured capacity remains inactive for the tested data?", "audit_unused_capacity", "trace_dataset"),
    ]
    for title, image, question, function, arguments in darkside_names:
        method = slug(title).replace("-", "_").replace("cost_evidence", "cost_evidence")
        call = f"from vraven.darkside import {function}\n\nresult = {function}({arguments})\nfigure = result.figure(mode=\"publication_light\")\nfigure.write_html(\"result.html\")"
        items.append(output("DarkSide", title, image, question, call, "An internal audit view for the synthetic CNN-2D fixture.", "This diagnostic is hypothesis-generating unless paired with appropriate controls or formal verification.", f"darkside/{method}.html", f"vraven.darkside.{function}"))

    items.append(output("Status", "Scientific Synthesis", "scientific_synthesis.png", "What can be concluded from the complete run, and what remains unresolved?", 'pipeline.write_scientific_synthesis(output_path="SCIENTIFIC_SYNTHESIS.html")', "A run-level synthesis of successful, skipped, and bounded claims.", "The synthesis inherits the limitations of every included analysis and must be read with its status notes.", "SCIENTIFIC_SYNTHESIS.html", "vraven.decompiler.ReverseEngineeringPipeline"))
    return items


def evidence_level(item: Output) -> str:
    lowered = item.title.lower()
    if "formal" in lowered or "certificate" in lowered:
        return "Formal — within stated verifier assumptions"
    if any(term in lowered for term in ("causal", "counterfactual", "adversarial", "neuron impact", "mediation", "synergy", "shortcut")):
        return "Interventional or perturbational"
    if item.family == "Status":
        return "Synthesis of mixed evidence"
    return "Observational or descriptive"


def page_shell(*, title: str, description: str, body: str, root: str, page: str) -> str:
    return f'''<!doctype html>
<html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{esc(description)}"><title>{esc(title)} · VRAVEN</title>
<link rel="stylesheet" href="{root}/assets/styles.css"></head>
<body data-page="{esc(page)}" data-root="{root}"><div class="page"><main id="content" class="content guide">
{body}
<footer>VRAVEN documentation · Evidence before ornament.</footer></main></div>
<script src="{root}/assets/app.js"></script></body></html>'''


def build_outputs() -> dict[str, Any]:
    target = SITE / "outputs"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    items = output_catalogue()
    by_family: dict[str, list[Output]] = {}
    for item in items:
        by_family.setdefault(item.family, []).append(item)

    all_links: list[tuple[Output, str]] = []
    for family, group in by_family.items():
        family_dir = target / slug(family)
        family_dir.mkdir(parents=True)
        for index, item in enumerate(group):
            href = f"outputs/{slug(family)}/{slug(item.title)}.html"
            all_links.append((item, href))
            prev_item = group[index - 1] if index else None
            next_item = group[index + 1] if index + 1 < len(group) else None
            prev_link = f'<a rel="prev" title="{esc(prev_item.title)}" href="{slug(prev_item.title)}.html">← Prev (K)</a>' if prev_item else '<span>Start of family</span>'
            next_link = f'<a rel="next" title="{esc(next_item.title)}" href="{slug(next_item.title)}.html">Next (J) →</a>' if next_item else '<a rel="next" href="../index.html">Output index →</a>'
            if item.family == "DarkSide":
                focused = item.call
            elif ".figure(" in item.call:
                focused = "# Use the typed result object returned by the matching analysis.\nfigure = " + item.call + "\nfigure.write_html(\"result.html\")"
            else:
                focused = item.call
            body = f'''
<a class="backlink" href="../../index.html">← Observatory home</a>
<div class="eyebrow">{esc(item.family)} · {index + 1} of {len(group)}</div>
<h1>{esc(item.title)}</h1>
<p class="lede">{esc(item.question)}</p>
<div class="output-nav">{prev_link}<span>image_pattern_cnn2d · <a href="../../assets/artifacts/{esc(item.image)}">Open result image ↗</a></span>{next_link}</div>
<figure class="evidence-figure"><a href="../../assets/artifacts/{esc(item.image)}"><img src="../../assets/artifacts/{esc(item.image)}" alt="{esc(item.title)} result for the image_pattern_cnn2d fixture"></a><figcaption>Real output from the small deterministic CNN-2D documentation fixture.</figcaption></figure>
<section class="reading-grid"><div><h2>What it shows</h2><p>{esc(item.result)}</p></div><div><h2>Claim boundary</h2><p>{esc(item.boundary)}</p></div></section>
<h2>Run this with the installed CLI</h2>
<p>Point VRAVEN at a validated model bundle. This produces one focused HTML report and works after a normal PyPI installation.</p>{code(cli_bundle_command())}
<p>Start with the <a href="../../guides/model-folder.html">model-folder guide</a> if your bundle is not prepared yet. The larger documentation fixture is a contributor workflow described in <a href="../../guides/source-checkout.html">Develop from source</a>.</p>
<h2>Generate this view</h2>{code(focused)}
<div class="parameter-grid"><div><strong>Input</strong><span>Model bundle, one reference input, selected target</span></div><div><strong>Display mode</strong><span><code>publication_light</code></span></div><div><strong>Evidence level</strong><span>{esc(evidence_level(item))}</span></div><div><strong>Machine output</strong><span><code>{esc(item.machine)}</code></span></div><div><strong>Related API</strong><span><a href="../../api/index.html?q={esc(item.api)}">{esc(item.api)}</a></span></div><div><strong>Fixture</strong><span><code>image_pattern_cnn2d</code></span></div></div>
<h2>How to read the result</h2><ol><li>Start with the title and target metadata in the generated report.</li><li>Read the strongest structures first, then inspect weaker or opposing evidence.</li><li>Check the stated evidence level and status before treating a relationship as causal or formal.</li><li>Retain the trace, model version, preprocessing, and VRAVEN version with any published claim.</li></ol>
<div class="callout"><strong>Reproducibility note</strong>The screenshot is public documentation evidence. The source run remains local-first; raw provenance files are not published because they can contain machine paths.</div>
<div class="next">{prev_link}{next_link}</div>'''
            (family_dir / f"{slug(item.title)}.html").write_text(page_shell(title=item.title, description=item.question, body=body, root="../..", page="outputs"), encoding="utf-8")

    family_cards = []
    for family, group in by_family.items():
        links = "".join(f'<li><a href="{slug(family)}/{slug(item.title)}.html"><strong>{esc(item.title)}</strong><span>{esc(item.question)}</span></a></li>' for item in group)
        family_cards.append(f'<section class="output-family" id="{slug(family)}"><div class="section-head"><h2>{esc(family)} <small>({len(group)})</small></h2><p>{esc({"Reverse":"Mechanism decompilation and verification","Proof":"Single-prediction evidence","Atlas":"Cohort reasoning maps","Watch":"Checkpoint evolution","Compare":"Matched model comparison","DarkSide":"Internal mechanism audits","Status":"Run-level synthesis"}[family])}</p></div><ol>{links}</ol></section>')
    body = f'''<div class="eyebrow">Output guide</div><h1>One question. One page. One reproducible result.</h1>
<p class="lede">Every successful output from the <code>image_pattern_cnn2d</code> fixture has a permanent page with runnable code, the real result, interpretation guidance, and a claim boundary.</p>
<div class="output-summary"><strong>{len(items)}</strong><span>documented outputs</span><strong>{len(by_family)}</strong><span>families</span></div>
{''.join(family_cards)}'''
    (target / "index.html").write_text(page_shell(title="Output guide", description="Every VRAVEN output explained with runnable code and a real result.", body=body, root="..", page="outputs"), encoding="utf-8")
    manifest = {"count": len(items), "families": {key: len(value) for key, value in by_family.items()}, "pages": [{"family": item.family, "title": item.title, "path": href, "image": item.image} for item, href in all_links]}
    (target / "inventory.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


CURATED_USAGE = {
    "vraven.explain": "import vraven\n\nreport = vraven.explain(model=model, inputs=x, target=target, mode=\"causal\")\nreport.write_report(\"explanation.html\")",
    "vraven.Explainer": "from vraven import Explainer\n\nexplainer = Explainer(model, mode=\"gradient\")\nreport = explainer.explain(x, target=target)\nreport.write_report(\"explanation.html\")",
    "vraven.load_model_bundle": "from vraven import load_model_bundle\n\nbundle = load_model_bundle(\"./my_model\")",
    "vraven.CounterfactualExplainer": "from vraven import CounterfactualExplainer\n\nengine = CounterfactualExplainer(model)\nresult = engine.explain(x, target=target, clamp=(0.0, 1.0))",
    "vraven.AdversarialAuditor": "from vraven import AdversarialAuditor\n\naudit = AdversarialAuditor(model).audit_sample(x, target=target)",
    "vraven.PerturbationStabilityAnalyzer": "from vraven import PerturbationStabilityAnalyzer\n\nresult = PerturbationStabilityAnalyzer(model).analyze(x, target=target)",
}


def safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def humanise_symbol(name: str) -> str:
    """Convert an exported Python name into a readable phrase."""
    words = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
    words = words.replace("_", " ").strip()
    return words.lower()


def public_documentation(obj: Any, name: str, kind: str) -> tuple[str, str]:
    """Return concise documentation without leaking built-in container prose."""
    raw = inspect.getdoc(obj) or ""
    if kind == "constant":
        summary = f"Exported VRAVEN constant {name}."
        return summary, summary
    inherited_builtin_doc = raw.startswith(("str(object=", "dict() ->", "Built-in "))
    if inspect.isclass(obj) and (raw.startswith(f"{name}(") or inherited_builtin_doc):
        summary = f"Structured VRAVEN data model for {humanise_symbol(name)}."
        return summary, summary
    doc = raw or "Public VRAVEN API symbol."
    summary = doc.split("\n\n", 1)[0].replace("\n", " ")
    return summary, doc


def api_symbols() -> list[dict[str, Any]]:
    symbols = []
    for module_name in ("vraven", "vraven.decompiler", "vraven.darkside"):
        module = importlib.import_module(module_name)
        names = list(getattr(module, "__all__", []))
        for name in names:
            if name.startswith("_") or not hasattr(module, name):
                continue
            obj = getattr(module, name)
            qualified = f"{module_name}.{name}"
            kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "constant"
            summary, doc = public_documentation(obj, name, kind)
            methods = []
            if inspect.isclass(obj):
                for method_name, method in inspect.getmembers(obj):
                    if method_name.startswith("_") or not callable(method):
                        continue
                    methods.append({"name": method_name, "signature": safe_signature(method), "summary": (inspect.getdoc(method) or "Public method.").split("\n\n", 1)[0].replace("\n", " ")})
            symbols.append({"module": module_name, "name": name, "qualified": qualified, "kind": kind, "signature": safe_signature(obj), "summary": summary, "doc": doc, "methods": methods})
    return symbols


def build_api() -> dict[str, Any]:
    # Resolve the package before touching the last known-good generated pages.
    # An unsupported interpreter or broken import must not erase the reference.
    symbols = api_symbols()
    target = SITE / "api"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in symbols:
        grouped.setdefault(item["module"], []).append(item)
        filename = f'{item["module"].replace(".", "-")}-{slug(item["name"])}.html'
        method_rows = ""
        if item["methods"]:
            method_rows = '<h2>Methods</h2><div class="api-members">' + "".join(f'<details><summary><code>{esc(m["name"])}{esc(m["signature"])}</code></summary><p>{esc(m["summary"])}</p></details>' for m in item["methods"]) + "</div>"
        usage = CURATED_USAGE.get(item["qualified"])
        if usage is None:
            if item["kind"] == "class":
                usage = f"from {item['module']} import {item['name']}\n\nvalue = {item['name']}(...)"
            elif item["kind"] == "function":
                usage = f"from {item['module']} import {item['name']}\n\nresult = {item['name']}(... )"
            else:
                usage = f"from {item['module']} import {item['name']}"
        signature = f'{item["name"]}{item["signature"]}' if item["signature"] else item["name"]
        doc_paragraphs = "".join(f"<p>{esc(paragraph).replace(chr(10), '<br>')}</p>" for paragraph in item["doc"].split("\n\n")[:4])
        body = f'''<a class="backlink" href="index.html">← Python API</a><div class="eyebrow">{esc(item["module"])} · {esc(item["kind"])}</div>
<h1>{esc(item["name"])}</h1><div class="signature"><code>{esc(signature)}</code></div>
<section class="api-doc">{doc_paragraphs}</section><h2>Usage</h2>{code(usage)}
{method_rows}
<div class="callout"><strong>Source of truth</strong>This page is generated from the checked-out public symbol, signature, and docstring. Pin the package version for reproducible work.</div>
<div class="next"><a href="index.html">← API index</a><a href="../outputs/index.html">Output guide →</a></div>'''
        (target / filename).write_text(page_shell(title=item["qualified"], description=f'{item["qualified"]}: {item["summary"]}', body=body, root="..", page="api"), encoding="utf-8")
        item["path"] = filename

    sections = []
    for module_name, group in grouped.items():
        rows = "".join(f'<a class="api-row" href="{esc(item["path"])}"><span><code>{esc(item["name"])}</code><small>{esc(item["kind"])}</small></span><p>{esc(item["summary"])}</p></a>' for item in sorted(group, key=lambda x: x["name"].lower()))
        sections.append(f'<section><div class="section-head"><h2>{esc(module_name)}</h2><p>{len(group)} public symbols</p></div><div class="api-list">{rows}</div></section>')
    body = f'''<div class="eyebrow">Reference</div><h1>Python API</h1><p class="lede">The complete exported surface of <code>vraven</code>, <code>vraven.decompiler</code>, and <code>vraven.darkside</code>, generated from the package itself.</p>
<label class="reference-search">Filter the API <input id="reference-filter" type="search" placeholder="explain, verifier, graph…"></label>
<div class="callout"><strong>{len(symbols)} public symbols</strong>Each symbol has its own signature, docstring, methods, and minimal usage page.</div>{''.join(sections)}'''
    (target / "index.html").write_text(page_shell(title="Python API", description="Complete generated VRAVEN Python API reference.", body=body, root="..", page="api"), encoding="utf-8")
    inventory = {"count": len(symbols), "modules": {key: len(value) for key, value in grouped.items()}, "symbols": symbols}
    (target / "inventory.json").write_text(json.dumps(inventory, indent=2, default=str) + "\n", encoding="utf-8")
    return inventory


def parser_commands(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            commands = []
            seen: set[int] = set()
            help_by_name = {choice.dest: choice.help for choice in action._choices_actions}
            for name, child in action.choices.items():
                if id(child) in seen:
                    continue
                seen.add(id(child))
                aliases = sorted(key for key, value in action.choices.items() if value is child and key != name)
                options = []
                for option in child._actions:
                    if option.dest == "help":
                        continue
                    options.append({"flags": option.option_strings or [option.dest], "dest": option.dest, "required": bool(getattr(option, "required", False)), "default": option.default if isinstance(option.default, (str, int, float, bool, type(None))) else str(option.default), "help": option.help or ""})
                commands.append({"name": name, "aliases": aliases, "help": child.description or help_by_name.get(name) or "VRAVEN command.", "options": options})
            return commands
    return []


def build_cli() -> dict[str, Any]:
    from vraven.cli import build_parser

    target = SITE / "cli"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    commands = parser_commands(build_parser())
    for command in commands:
        option_rows = "".join(f'<tr><td><code>{esc(", ".join(opt["flags"]))}</code></td><td>{esc(opt["help"] or "—")}</td><td><code>{esc(opt["default"])}</code></td></tr>' for opt in command["options"])
        positional = [opt["dest"] for opt in command["options"] if not str(opt["flags"][0]).startswith("-")]
        example = f"vraven {command['name']}" + "".join(f" <{name}>" for name in positional)
        aliases = f'<p class="muted">Aliases: {", ".join(f"<code>{esc(a)}</code>" for a in command["aliases"])}</p>' if command["aliases"] else ""
        body = f'''<a class="backlink" href="index.html">← CLI reference</a><div class="eyebrow">Command</div><h1>vraven {esc(command["name"])}</h1>
<p class="lede">{esc(command["help"])}</p>{aliases}<h2>Minimal form</h2>{code(example)}
<h2>Arguments and options</h2><div class="table-scroll"><table class="command-table"><thead><tr><th>Option</th><th>Meaning</th><th>Default</th></tr></thead><tbody>{option_rows}</tbody></table></div>
<div class="callout"><strong>Exact local help</strong>Run <code>vraven {esc(command["name"])} --help</code> against your installed version before automating a release workflow.</div>'''
        (target / f'{slug(command["name"])}.html').write_text(page_shell(title=f'vraven {command["name"]}', description=f'CLI reference for vraven {command["name"]}.', body=body, root="..", page="cli"), encoding="utf-8")
    rows = "".join(f'<a class="api-row" href="{slug(command["name"])}.html"><span><code>vraven {esc(command["name"])}</code><small>{len(command["options"])} {"option" if len(command["options"]) == 1 else "options"}</small></span><p>{esc(command["help"])}</p></a>' for command in commands)
    body = f'''<div class="eyebrow">Reference</div><h1>Command-line interface</h1><p class="lede">Every command and parser option generated from the current CLI.</p><div class="api-list">{rows}</div>'''
    (target / "index.html").write_text(page_shell(title="CLI reference", description="Complete generated VRAVEN command-line reference.", body=body, root="..", page="cli"), encoding="utf-8")
    inventory = {"count": len(commands), "commands": commands}
    (target / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def main() -> int:
    outputs = build_outputs()
    api = build_api()
    cli = build_cli()
    print(f"Built {outputs['count']} output pages, {api['count']} API pages, and {cli['count']} CLI pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
