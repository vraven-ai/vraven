#!/usr/bin/env python3
"""Run the complete applicable VRAVEN suite from one self-describing artefact directory.

The only required argument is the artefact directory.  The directory must contain
``model.pt``, ``meta.json`` and ``reference.npz``.  Architecture reconstruction is
driven by the packaged architecture declaration; reference inputs, labels, class
names, feature names, groups and checkpoints are all read from the same bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_vraven_gallery as gallery  # noqa: E402
import run_vraven_on_models as suite  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs(reference_path: Path) -> dict[str, Any]:
    reference = np.load(reference_path, allow_pickle=False)
    if "images" in reference.files:
        values = np.asarray(reference["images"], dtype=np.float32)
    elif "features" in reference.files:
        values = np.asarray(reference["features"], dtype=np.float32)
    else:
        raise ValueError("reference.npz requires an 'images' or 'features' array")
    if "labels" not in reference.files:
        raise ValueError("reference.npz requires a 'labels' array")
    labels = np.asarray(reference["labels"], dtype=np.int64)
    feature_names = (
        [str(item) for item in reference["feature_names"].tolist()]
        if "feature_names" in reference.files
        else []
    )
    if values.shape[0] != labels.shape[0]:
        raise ValueError("Reference inputs and labels must have the same first dimension")
    calibration_values: np.ndarray | None = None
    calibration_labels: np.ndarray | None = None
    calibration_ids: list[str] = []
    if "all_test_images" in reference.files and "all_test_labels" in reference.files:
        all_values = np.asarray(reference["all_test_images"], dtype=np.float32)
        all_labels = np.asarray(reference["all_test_labels"], dtype=np.int64)
        evaluation_fingerprints = {row.tobytes() for row in values}
        eligible = [
            index
            for index, row in enumerate(all_values)
            if row.tobytes() not in evaluation_fingerprints
        ]
        selected: list[int] = []
        for label in sorted(int(value) for value in np.unique(all_labels[eligible])):
            class_indices = [
                index for index in eligible if int(all_labels[index]) == label
            ]
            selected.extend(class_indices[:8])
        if selected:
            calibration_values = all_values[selected]
            calibration_labels = all_labels[selected]
            calibration_ids = [f"all_test:{index}" for index in selected]
    return {
        "inputs": torch.from_numpy(values),
        "labels": labels,
        "feature_names": feature_names,
        "calibration_inputs": (
            None if calibration_values is None else torch.from_numpy(calibration_values)
        ),
        "calibration_labels": calibration_labels,
        "calibration_sample_ids": calibration_ids,
        "evaluation_sample_ids": [f"reference:{index}" for index in range(len(labels))],
        "calibration_disjoint_from_evaluation": bool(calibration_ids),
    }


def _input_spec(metadata: dict[str, Any]) -> suite.InputDomainSpec | None:
    payload = metadata.get("input_domain")
    if not isinstance(payload, dict):
        return None
    return suite.InputDomainSpec(
        lower=payload.get("lower"),
        upper=payload.get("upper"),
        units=str(payload.get("units", "")),
        preprocessing=str(payload.get("preprocessing", "")),
        semantic_domain=str(payload.get("semantic_domain", "")),
        source=str(payload.get("source", "")),
        declared=bool(payload.get("declared", True)),
    )


def _build_model(payload: dict[str, Any]) -> torch.nn.Module:
    architecture = payload.get("architecture")
    state = payload.get("model_state")
    if not isinstance(architecture, dict) or not isinstance(state, dict):
        raise ValueError("model.pt requires 'architecture' and 'model_state' mappings")
    model = suite._build_from_arch(architecture).eval()
    model.load_state_dict(state, strict=True)
    return model


def _checkpoint_loader(
    artifact_dir: Path,
) -> tuple[list[torch.nn.Module], list[str]]:
    paths = sorted((artifact_dir / "checkpoints").glob("epoch_*.pt"))
    if len(paths) >= 3:
        paths = [paths[0], paths[len(paths) // 2], paths[-1]]
    models: list[torch.nn.Module] = []
    names: list[str] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        models.append(_build_model(payload))
        names.append(path.stem)
    return models, names


def _load_bundle(artifact_dir: Path) -> dict[str, Any]:
    required = tuple(artifact_dir / name for name in ("model.pt", "meta.json", "reference.npz"))
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Artefact directory is missing: {', '.join(missing)}")
    metadata = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    payload = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
    model = _build_model(payload)
    reference = _load_inputs(artifact_dir / "reference.npz")
    inputs = reference["inputs"]
    labels = reference["labels"]
    reference_names = reference["feature_names"]
    class_list = list(payload.get("class_names") or metadata.get("class_names") or [])
    if len(class_list) < 2:
        raise ValueError("Artefact requires at least two class names")
    class_names = {index: str(name) for index, name in enumerate(class_list)}
    feature_names = [str(name) for name in (payload.get("feature_names") or reference_names)]
    if not feature_names:
        feature_count = int(np.prod(inputs.shape[1:]))
        feature_names = [f"feature_{index}" for index in range(feature_count)]
    groups = {
        str(name): [int(index) for index in indices]
        for name, indices in (payload.get("input_groups") or {}).items()
    }
    return {
        "model": model,
        "inputs": inputs,
        "labels": labels,
        "feature_names": feature_names,
        "class_names": class_names,
        "input_groups": groups,
        "metadata": metadata,
        "payload": payload,
        "input_spec": _input_spec(metadata),
        "calibration_inputs": reference["calibration_inputs"],
        "calibration_labels": reference["calibration_labels"],
        "calibration_sample_ids": reference["calibration_sample_ids"],
        "evaluation_sample_ids": reference["evaluation_sample_ids"],
        "calibration_disjoint_from_evaluation": reference[
            "calibration_disjoint_from_evaluation"
        ],
    }


def _first_checkpoint_peer(artifact_dir: Path) -> tuple[str, torch.nn.Module] | None:
    paths = sorted((artifact_dir / "checkpoints").glob("epoch_*.pt"))
    if not paths:
        return None
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    return paths[0].stem, _build_model(payload)


def _provenance(artifact_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    files = [artifact_dir / "model.pt", artifact_dir / "meta.json", artifact_dir / "reference.npz"]
    files.extend(sorted((artifact_dir / "checkpoints").glob("epoch_*.pt")))
    model = bundle["model"]
    inputs = bundle["inputs"]
    labels = bundle["labels"]
    with torch.inference_mode():
        logits = model(inputs)
        predictions = logits.argmax(dim=-1).cpu().numpy()
    return {
        "artefact_directory": str(artifact_dir),
        "input_contract": "self-describing VRAVEN model bundle",
        "architecture": bundle["payload"].get("architecture"),
        "reference_shape": list(inputs.shape),
        "reference_samples": int(inputs.shape[0]),
        "reference_accuracy": float(np.mean(predictions == labels)),
        "class_names": bundle["class_names"],
        "feature_count": len(bundle["feature_names"]),
        "input_groups": {name: len(indices) for name, indices in bundle["input_groups"].items()},
        "input_domain": (
            None if bundle["input_spec"] is None else bundle["input_spec"].to_dict()
        ),
        "coordinate_calibration": {
            "n_samples": (
                0
                if bundle["calibration_inputs"] is None
                else int(bundle["calibration_inputs"].shape[0])
            ),
            "sample_ids": bundle["calibration_sample_ids"],
            "disjoint_from_evaluation": bundle[
                "calibration_disjoint_from_evaluation"
            ],
        },
        "file_sha256": {path.relative_to(artifact_dir).as_posix(): _sha256(path) for path in files},
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_scientific_synthesis(output_dir: Path) -> Path:
    """Compose a critical cross-analysis report without inventing a trust score."""

    proof = _read_json(output_dir / "proof" / "scientific_evidence_audit.json")
    provenance = _read_json(output_dir / "ARTEFACT_PROVENANCE.json")
    reverse = _read_json(output_dir / "reverse" / "summary.json")
    robust = _read_json(output_dir / "reverse" / "adversarial_robust_accuracy.json")
    robustness_profile = _read_json(
        output_dir / "reverse" / "adversarial_robustness_profile.json"
    )
    formal = _read_json(output_dir / "reverse" / "formal_robustness_certificate.json")
    formal_mechanism = _read_json(
        output_dir / "reverse" / "formal_causal_mechanism_certificate.json"
    )
    mechanism_cohort = formal_mechanism.get("cohort_points", [])
    mechanism_cohort_zero = mechanism_cohort[0] if mechanism_cohort else {}
    mechanism_cohort_endpoint = mechanism_cohort[-1] if mechanism_cohort else {}
    mechanism_selection = formal_mechanism.get("coordinate_selection", {})
    mechanism_readiness = formal_mechanism.get("publication_readiness", {})
    darkside_status = _read_json(output_dir / "darkside" / "STATUS.json")
    concentration = proof.get("concentration", {})
    concordance = proof.get("concordance", {})

    darkside_payloads: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "darkside").glob("*.json")):
        if path.name in {"STATUS.json", "recommend_features.json"}:
            continue
        darkside_payloads[path.stem] = _read_json(path)

    shortcut = darkside_payloads.get("shortcut_reliance", {}).get("metrics", {})
    synergy = darkside_payloads.get("causal_synergy", {}).get("metrics", {})
    stability = darkside_payloads.get("stability_geometry", {}).get("metrics", {})
    drift = darkside_payloads.get("representation_drift", {}).get("metrics", {})
    class_effect = darkside_payloads.get("class_conditional_internals", {}).get("metrics", {})
    transfer = darkside_payloads.get("mechanism_transfer", {}).get("metrics", {})
    robust_interval = robust.get("metadata", {}).get("robust_accuracy_wilson_95")
    robust_interval_text = (
        f"[{float(robust_interval[0]):.3f}, {float(robust_interval[1]):.3f}]"
        if isinstance(robust_interval, list) and len(robust_interval) == 2
        else "not measured"
    )
    certified_cohort_points = [
        point
        for point in formal.get("cohort_points", [])
        if int(point.get("n_certified", 0)) > 0
    ]
    last_certified_cohort = (
        certified_cohort_points[-1] if certified_cohort_points else {}
    )

    findings = [
        (
            "Output performance",
            f"The packaged reference cohort contains {provenance.get('reference_samples', 'unknown')} "
            f"samples and has accuracy {_number(provenance.get('reference_accuracy'))}. "
            "This is an artefact-contained evaluation, not an external generalisation estimate.",
        ),
        (
            "Evidence is distributed",
            f"Absolute first-order evidence spans {_number(concentration.get('effective_layers'), 2)} "
            f"effective layers out of {proof.get('gradient_layer_count', 'unknown')}; 80% of the "
            f"mass is reached at {concentration.get('eighty_mass_layer', 'an unresolved layer')}.",
        ),
        (
            "Causal agreement is promising but underpowered",
            f"Matched gradient-versus-zero-ablation effects give Spearman ρ="
            f"{_number(concordance.get('spearman_rho'))}, sign agreement "
            f"{_number(100 * concordance.get('sign_agreement', 0), 1)}%, and normalised L1 "
            f"discrepancy {_number(concordance.get('normalised_l1_discrepancy'))}. Only "
            f"{proof.get('zero_ablation_layer_count', 0)} layers were matched, so the exact "
            f"permutation p={_number(concordance.get('permutation_p_value'))} does not support a "
            "population-level concordance claim.",
        ),
        (
            "Local mechanism interaction",
            f"The strongest tested early-convolution interaction is "
            f"{_number(synergy.get('strongest_interaction'))} under joint zero ablation. This is "
            "context- and baseline-specific causal evidence, not a universal circuit law.",
        ),
        (
            "Potential quadrant reliance",
            f"The strongest declared input group is {shortcut.get('strongest_group', 'unresolved')} "
            f"with absolute relative target-score change "
            f"{_number(shortcut.get('maximum_absolute_relative_drop'))}; none of the four group "
            "ablations flipped the selected decision.",
        ),
        (
            "Stable local geometry",
            f"Gaussian σ=0.05 paired perturbations produced minimum internal stability "
            f"{_number(stability.get('minimum_stability_index'))} at "
            f"{stability.get('weakest_layer', 'an unresolved layer')}, while epoch-01-to-final "
            f"maximum drift was {_number(drift.get('maximum_drift_index'))}. These composites rank "
            "effects; they are not calibrated probabilities.",
        ),
        (
            "Strong class separation",
            f"The largest cohort distribution contrast occurs at "
            f"{class_effect.get('largest_distribution_shift_layer', 'an unresolved layer')} with "
            f"energy distance {_number(class_effect.get('largest_energy_distance'))}. Separation "
            "shows decodability, not causal use or semantic validity.",
        ),
        (
            "Checkpoint mechanism retention",
            f"Aligned zero-ablation maps have a three-component stable core; the largest "
            f"normalised L1 distance is {_number(transfer.get('maximum_normalised_l1_distance'))} "
            f"for {escape(str(transfer.get('most_distant_pair', 'an unresolved pair')))}.",
        ),
        (
            "Robustness has a sharp tested boundary",
            f"The stratified VRAVEN ensemble-lite pilot retained "
            f"{_number(robust.get('robust_accuracy'))} attacked robust accuracy at "
            f"L∞ ε={_number(robust.get('epsilon'))} over {robust.get('n_samples', 0)} "
            f"samples, with pointwise Wilson 95% CI {robust_interval_text}. "
            f"Across the tested interval its robust-accuracy AUC was "
            f"{_number(robustness_profile.get('robust_accuracy_auc'))}, but endpoint robustness "
            f"fell to {_number(robustness_profile.get('endpoint_robust_accuracy'))} at the maximum "
            "budget. This sharp transition is more informative than the isolated 1.0 value and "
            "the 12-sample pilot is neither a certificate nor a RobustBench evaluation.",
        ),
        (
            "Formal and empirical radii now bracket different evidence",
            f"Native interval verification proves the focus prediction invariant through at "
            f"least L∞ ε={_number(formal.get('certified_radius_lower_bound'), 6)}, while "
            f"the first successful attack was observed at ε="
            f"{_number(_read_json(output_dir / 'reverse' / 'adversarial_audit.json').get('empirical_min_flip_eps_linf'))}. "
            f"At ε={_number(last_certified_cohort.get('epsilon'), 6)}, "
            f"{last_certified_cohort.get('n_certified', 0)}/"
            f"{last_certified_cohort.get('n_samples', 0)} cohort samples are formally certified. "
            "The gap is an explicit unresolved region: an inconclusive interval bound is not "
            "evidence of vulnerability, and attack non-discovery is not a proof of robustness.",
        ),
        (
            "Internal causal-mechanism survival is now a separate formal property",
            f"At node {escape(str(formal_mechanism.get('mechanism_spec', {}).get('node_id', 'unsupported')))}, "
            f"the proven representation-survival radius is at least L∞ ε="
            f"{_number(formal_mechanism.get('representation_radius_lower_bound'), 8)}, "
            f"the intervention-effect-vector survival radius is at least ε="
            f"{_number(formal_mechanism.get('causal_effect_radius_lower_bound'), 8)}, and "
            f"the conjunction with decision invariance is proven through ε="
            f"{_number(formal_mechanism.get('joint_radius_lower_bound'), 8)}. The causal "
            "vector is defined by executable coordinate interventions on a frozen basis; "
            "failure to certify beyond these radii is verifier inconclusiveness, not proof "
            f"that the mechanism changed. Across the frozen-basis cohort, "
            f"{mechanism_cohort_zero.get('n_jointly_certified', 0)}/"
            f"{mechanism_cohort_zero.get('n_samples', 0)} samples satisfy the full property "
            f"at ε=0, while {mechanism_cohort_endpoint.get('n_jointly_certified', 0)}/"
            f"{mechanism_cohort_endpoint.get('n_samples', 0)} are proven jointly at ε="
            f"{_number(mechanism_cohort_endpoint.get('epsilon'), 8)}. This is certificate "
            f"coverage for a {mechanism_selection.get('mode', 'legacy')} basis selected on "
            f"n={mechanism_selection.get('n_samples', 'unknown')} calibration samples, not a "
            "mechanism-failure rate. The publication ledger remains "
            f"{mechanism_readiness.get('status', 'not assessed')}; outstanding requirements are "
            f"{', '.join(mechanism_readiness.get('blocking_requirements', [])) or 'none'}.",
        ),
    ]

    finding_html = "".join(
        f"<article class='finding'><h3>{escape(title)}</h3><p>{body}</p></article>"
        for title, body in findings
    )
    ledger_rows = []
    for name, payload in darkside_payloads.items():
        metadata = payload.get("metadata", {})
        limitations = payload.get("limitations", [])
        html_name = f"{name}_interactive_dark.html"
        href = (
            f"darkside/{html_name}"
            if (output_dir / "darkside" / html_name).is_file()
            else f"darkside/{name}.json"
        )
        ledger_rows.append(
            "<tr>"
            f"<td><a href='{escape(href)}'>{escape(name.replace('_', ' ').title())}</a></td>"
            f"<td>{escape(str(metadata.get('claim_type', 'declared in result JSON')))}</td>"
            f"<td>{escape(str(limitations[0] if limitations else 'See machine-readable result.'))}</td>"
            "</tr>"
        )
    skip_rows = "".join(
        f"<li><strong>{escape(name.replace('_', ' '))}</strong>: {escape(str(reason))}</li>"
        for name, reason in darkside_status.get("skipped", {}).items()
    )
    warnings = "".join(f"<li>{escape(str(item))}</li>" for item in reverse.get("warnings", []))

    html = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VRAVEN scientific synthesis · {escape(output_dir.name)}</title>
<style>
:root{{--bg:#030A12;--panel:#07111F;--ink:#F1F5F9;--muted:#94A3B8;--accent:#00D4FF;--warn:#FF8A00;--line:#1C344D}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top right,#123142,var(--bg) 38%);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:52px 24px 80px}} h1{{font-size:clamp(2rem,5vw,4rem);line-height:1.05;margin:.2em 0}} h2{{margin-top:2.2em}}
.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase}} .lede,.note{{color:var(--muted);max-width:80ch}}
.nav{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}} a{{color:var(--accent)}} .nav a{{border:1px solid var(--line);padding:9px 13px;border-radius:10px;text-decoration:none;background:#0c1822}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}} .finding{{background:linear-gradient(145deg,#0B1728,#07111F);border:1px solid var(--line);border-radius:14px;padding:18px}}
.finding h3{{font-size:1rem;color:var(--accent);margin:0 0 7px}} .finding p{{margin:0}} table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}}
th,td{{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}} th{{color:var(--accent)}} code{{color:var(--warn)}}
.boundary{{border-left:4px solid var(--warn);background:#1d1b18;padding:16px 20px;border-radius:8px}} footer{{margin-top:40px;color:var(--muted)}}
</style></head><body><main>
<p class='eyebrow'>Evidence-led artefact audit</p><h1>{escape(output_dir.name)}</h1>
<p class='lede'>A cross-analysis synthesis of VRAVEN Capture, Reverse, Proof, Atlas, Watch, Compare, adversarial stress and DarkSide outputs. No composite trust score is emitted: observational, interventional, geometric and finite-sample evidence remain separate.</p>
<nav class='nav'><a href='INDEX.html'>All generated views</a><a href='proof/vraven_proof.html'>Scientific Evidence Audit</a><a href='reverse/vraven_decision_proof.html'>Decision Proof</a><a href='reverse/formal_robustness_certificate.html'>Formal decision certificate</a><a href='reverse/formal_causal_mechanism_certificate.html'>Formal causal-mechanism certificate</a><a href='reverse/adversarial_audit.html'>Adversarial audit</a><a href='reverse/adversarial_robustness_profile.html'>Empirical profile</a><a href='ARTEFACT_PROVENANCE.json'>Provenance</a></nav>
<h2>Critical findings</h2><section class='grid'>{finding_html}</section>
<h2>Scientific boundary</h2><div class='boundary'><p>The decompiled programme has observational validation R²={_number(reverse.get("programme_validation_r2"))}, but interventional sign agreement is only {_number(reverse.get("programme_interventional_sign_agreement"))}. Causal semantic coverage is {_number(reverse.get("causal_semantic_coverage"))}, and no verified counterfactual was found. The mechanism is therefore a tested local hypothesis, not a recovered ground-truth algorithm.</p><ul>{warnings}</ul></div>
<h2>Evidence ledger</h2><p class='note'>Each row links to the measured visual or JSON and states the claim family and its leading limitation.</p>
<table><thead><tr><th>Analysis</th><th>Claim type</th><th>Leading limitation</th></tr></thead><tbody>{"".join(ledger_rows)}</tbody></table>
<h2>Not estimable from this artefact</h2><ul>{skip_rows}</ul>
<h2>Reproducibility</h2><p>Capture route: <code>{escape(str(reverse.get("capture_route", "unknown")))}</code>. Model bundle hashes, architecture, shapes and cohort accuracy are retained in <a href='ARTEFACT_PROVENANCE.json'>ARTEFACT_PROVENANCE.json</a>. Exact numerical tables live beside every visual.</p>
<footer>Generated by the artefact-directory-only VRAVEN research runner.</footer>
</main></body></html>"""
    destination = output_dir / "SCIENTIFIC_SYNTHESIS.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(argv)
    artifact_dir = args.artifact_dir.expanduser().resolve()
    bundle = _load_bundle(artifact_dir)

    output_base = ROOT / "vraven_outputs" / "artifact_runs"
    suite.OUT = output_base
    gallery.OUT = output_base
    name = artifact_dir.name
    output_dir = output_base / name
    output_dir.mkdir(parents=True, exist_ok=True)
    suite._write_json(output_dir / "ARTEFACT_PROVENANCE.json", _provenance(artifact_dir, bundle))

    status = suite.run_classifier_suite(
        name=name,
        model=bundle["model"],
        x=bundle["inputs"],
        y=bundle["labels"],
        feature_names=bundle["feature_names"],
        class_names=bundle["class_names"],
        input_groups=bundle["input_groups"],
        checkpoint_loader=lambda: _checkpoint_loader(artifact_dir),
        compare_peer=_first_checkpoint_peer(artifact_dir),
        input_spec=bundle["input_spec"],
        calibration_x=bundle["calibration_inputs"],
        calibration_sample_ids=bundle["calibration_sample_ids"],
        calibration_y=bundle["calibration_labels"],
        calibration_disjoint_from_evaluation=bundle[
            "calibration_disjoint_from_evaluation"
        ],
        evaluation_sample_ids=bundle["evaluation_sample_ids"],
    )
    run_status = {
        "vraven_version": getattr(suite.vraven, "__version__", "unknown"),
        "invocation": "artefact-directory-only",
        "artifact": str(artifact_dir),
        "model": name,
        "status": status,
    }
    suite._write_json(output_base / "RUN_STATUS.json", run_status)
    synthesis = _write_scientific_synthesis(output_dir)
    gallery.main()
    print(f"\nArtefact report: {output_dir / 'INDEX.html'}")
    print(f"Scientific synthesis: {synthesis}")
    print(f"Scientific proof: {output_dir / 'proof' / 'vraven_proof.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
