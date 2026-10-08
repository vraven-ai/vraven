#!/usr/bin/env python3
"""Exercise every implemented VRAVEN 0.9 capability on Models/.

Stacks covered:
  Reverse   vraven.reverse / Decision Proof + Raven visuals + Adversarial Mechanism Audit
  Proof     vraven.explain (+ LRP, perturbation, counterfactual) + all capable figures
  Atlas     vraven.explain_dataset + landscape/similarity/outliers
  Watch     vraven.compare_checkpoints (+ DarkSide representation_drift when possible)
  Compare   vraven.compare_models
  Capture   Mechanism IR only (vraven.decompiler.ModelCaptureRouter)
  DarkSide  circuitry, anatomy, wiring, shortcuts, synergy, failure, OOD, depth,
            capacity, cost-evidence, multiplicity, class-conditional, mediation

Autoencoder packages use the deployed reconstruction-MSE anomaly score
(Benign vs Anomaly logits) plus the paper charmap injection reference rows.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "examples" / "kronodroid"))
sys.path.insert(0, str(ROOT / "examples" / "windows_malware_mlp"))

from ae_models import load_ae_anomaly_bundle
from model import MalwareMLP as LolbinMLP
from probe_models import build_probe
from publication_models import build_publication_model

import vraven
from models import MODEL_BUILDERS
from vraven.decompiler import (
    AdversarialAuditor,
    CaptureConfig,
    CausalEffectPropertySpec,
    DecompileConfig,
    FormalCausalMechanismVerifier,
    InputDomainSpec,
    IntervalBoundVerifier,
    MechanismPropertySpec,
    ModelCaptureRouter,
    VRavenDecompiler,
)
from vraven.report.html import export_html_report

OUT = ROOT / "vraven_outputs"
MODELS = ROOT / "Models"

KRONO_GROUPS = {
    "Linux system-call behaviour": tuple(range(0, 289)),
    "Android permissions": tuple(range(289, 462)),
    "APK packaging and submission metadata": tuple(range(462, 466)),
    "Network contact behaviour": (466,),
    "Android component structure": tuple(range(467, 476)),
}

PROOF_VISUALS = (
    "activation_energy",
    "decision_flow",
    "decision_sankey",
    "spectral_trace",
    "neuron_impact",
    "neuron_importance",
    "latent_flight",
    "reasoning_dna",
    "conflict_field",
    "evidence_trajectory",
    "causal_pathway",
    "mechanism_concordance",
    "input_attribution",
    "feature_sequence_heatmap",
    "reasoning_graph",
    "community_graph",
    "relevance_flow",
    "perturbation_stability",
    "counterfactual_delta",
    "attention_structure",
)

REVERSE_VISUALS = (
    "model_xray",
    "concept_genome",
    "causal_proof",
    "programme_proof",
    "contrastive_proof",
    "counterfactual_twin",
    "dark_matter",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _linear_names(model: torch.nn.Module, limit: int = 6) -> tuple[str, ...]:
    linears = tuple(n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n)
    if linears:
        return linears[:limit]
    # Fall back to Conv layers for pure convolutional stacks.
    convs = tuple(
        n
        for n, m in model.named_modules()
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)) and n
    )
    return convs[:limit]


def _formal_mechanism_node(model: torch.nn.Module) -> str:
    """Choose the last explicit monotone activation before the model readout."""

    supported = (
        torch.nn.ReLU,
        torch.nn.LeakyReLU,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
    )
    candidates = [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, supported)
    ]
    if not candidates:
        raise ValueError("no supported internal activation node is available")
    return candidates[-1]


def _build_from_arch(arch: dict[str, Any]) -> torch.nn.Module:
    kind = arch["kind"]
    kwargs = {k: v for k, v in arch.items() if k not in {"kind", "task", "seed"}}
    if kind == "cnn1d" and "hidden" in kwargs:
        kwargs.pop("hidden")
    for key in ("hidden", "channels"):
        if key in kwargs and isinstance(kwargs[key], list):
            kwargs[key] = tuple(kwargs[key])
    if kind.startswith("publication_"):
        return build_publication_model(kind, **kwargs)
    return build_probe(kind, **kwargs)


def _save_figure(result_or_fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result_or_fig, "write_html"):
        result_or_fig.write_html(path, include_plotlyjs=True)
    else:
        result_or_fig.figure(mode="interactive_dark").write_html(path, include_plotlyjs=True)


def _save_reverse(result, dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    result.save(dest / "evidence")
    result.write_report(dest / "vraven_decision_proof.html", include_plotlyjs=True)
    figures = dest / "figures"
    figures.mkdir(exist_ok=True)
    for visual in REVERSE_VISUALS:
        for mode in ("interactive_dark", "publication_light"):
            try:
                result.write_feature_report(
                    visual,
                    figures / f"{visual}_{mode}.html",
                    initial_mode=mode,
                    include_plotlyjs=True,
                )
            except Exception as error:  # noqa: BLE001
                print(f"  Reverse figure {visual}/{mode} skipped: {error}")
    summary = result.summary()
    _write_json(dest / "summary.json", summary)
    return summary


def _infer_clip_range(sample: torch.Tensor) -> tuple[float | None, float | None]:
    """Clamp adversarial examples when the sample looks like a bounded domain."""

    lo = float(sample.detach().min().item())
    hi = float(sample.detach().max().item())
    if lo >= -1e-5 and hi <= 1.0 + 1e-5:
        return 0.0, 1.0
    if lo >= -1.0 - 1e-5 and hi <= 1.0 + 1e-5 and lo < -0.05:
        return -1.0, 1.0
    return None, None


def _save_adversarial_audit(
    model: torch.nn.Module,
    focus: torch.Tensor,
    dest: Path,
    *,
    model_name: str,
    feature_names: list[str] | None = None,
    target: int = 1,
    batch_x: torch.Tensor | None = None,
    batch_y: np.ndarray | None = None,
    autoattack_epsilon: float = 0.1,
    robust_max_samples: int = 12,
    input_spec: InputDomainSpec | None = None,
    calibration_x: torch.Tensor | None = None,
    calibration_sample_ids: list[str] | None = None,
    calibration_y: torch.Tensor | np.ndarray | None = None,
    calibration_disjoint_from_evaluation: bool | None = None,
    evaluation_sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run focus stress and a stratified cohort robustness profile."""

    dest.mkdir(parents=True, exist_ok=True)
    clip_min, clip_max = (
        _infer_clip_range(focus if batch_x is None else batch_x)
        if input_spec is None
        else (input_spec.lower, input_spec.upper)
    )
    flat_n = int(focus.detach().reshape(-1).numel())
    concept_names = None
    if feature_names and len(feature_names) == flat_n:
        concept_names = list(feature_names)
    auditor = AdversarialAuditor(model, model_name=model_name)
    report = auditor.audit_sample(
        focus,
        target=target,
        epsilon_grid=(0.01, 0.05, 0.1, 0.25, 0.5),
        pgd_steps=10,
        clip_min=clip_min,
        clip_max=clip_max,
        concept_names=concept_names,
        seed=24032026,
        run_pgd=True,
        fab_iters=15,
        fab_restarts=1,
        square_queries=60,
        n_target_classes=2,
    )
    aa = auditor.audit_autoattack(
        focus,
        epsilon=float(autoattack_epsilon),
        steps=15,
        clip_min=clip_min,
        clip_max=clip_max,
        seed=24032026,
        square_queries=120,
        fab_iters=25,
        fab_restarts=2,
        n_target_classes=2,
        target=target,
    )
    report.metadata["autoattack_epsilon"] = aa.metadata.get("autoattack_epsilon")
    report.metadata["autoattack_flipped_by"] = aa.metadata.get("autoattack_flipped_by")
    report.metadata["autoattack_robust"] = aa.metadata.get("autoattack_robust")
    report.metadata["autoattack_trials"] = aa.to_dict()["attacks_tested"]

    robust_payload = None
    formal_payload = None
    formal_mechanism_payload = None
    if batch_x is not None and int(batch_x.shape[0]) > 0:
        robust = auditor.audit_batch(
            batch_x,
            batch_y,
            epsilon=float(autoattack_epsilon),
            protocol="vraven_ensemble_lite",
            max_samples=int(robust_max_samples),
            sampling="stratified",
            clip_min=clip_min,
            clip_max=clip_max,
            seed=24032026,
            steps=8,
            square_queries=40,
            fab_iters=10,
            fab_restarts=1,
            n_target_classes=2,
        )
        robust_payload = robust.to_dict()
        profile = auditor.audit_robustness_curve(
            batch_x,
            batch_y,
            epsilon_grid=(0.01, 0.05, float(autoattack_epsilon), 0.25),
            protocol="vraven_ensemble_lite",
            max_samples=int(robust_max_samples),
            sampling="stratified",
            clip_min=clip_min,
            clip_max=clip_max,
            seed=24032026,
            steps=8,
            square_queries=40,
            fab_iters=10,
            fab_restarts=1,
            n_target_classes=2,
        )
        profile.save(dest / "adversarial_robustness_profile.json")
        profile.write_html(dest / "adversarial_robustness_profile.html")
        report.metadata["robustness_profile"] = profile.to_dict()
        report.metadata["robust_accuracy"] = {
            "protocol": robust.protocol,
            "epsilon": robust.epsilon,
            "n_samples": robust.n_samples,
            "n_correct_clean": robust.n_correct_clean,
            "n_robust": robust.n_robust,
            "clean_accuracy": robust.clean_accuracy,
            "robust_accuracy": robust.robust_accuracy,
            "conditional_robust_accuracy": robust.conditional_robust_accuracy,
            "attack_success_rate": robust.attack_success_rate,
            "per_attack_flips": robust.per_attack_flips,
            "per_attack_successes": robust.per_attack_successes,
            "per_attack_unique_flips": robust.per_attack_unique_flips,
            "sampling_protocol": robust.sampling_protocol,
            "selected_indices": robust.selected_indices,
            "per_class": robust.per_class,
            "macro_robust_accuracy": robust.macro_robust_accuracy,
            "evidence_scope": robust.metadata.get("evidence_scope"),
            "robust_accuracy_wilson_95": robust.metadata.get(
                "robust_accuracy_wilson_95"
            ),
        }
        (dest / "adversarial_robust_accuracy.json").write_text(
            json.dumps(robust_payload, indent=2),
            encoding="utf-8",
        )

        formal = IntervalBoundVerifier(model, model_name=model_name).certify(
            focus,
            target=target,
            epsilon_grid=(
                0.0,
                0.00001,
                0.00005,
                0.0001,
                0.00025,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.25,
            ),
            clip_min=clip_min,
            clip_max=clip_max,
            cohort_x=batch_x if batch_y is not None else None,
            cohort_y=None if batch_y is None else torch.as_tensor(batch_y),
            selected_indices=(
                None
                if batch_y is None
                else list(range(int(batch_x.shape[0])))
            ),
            radius_search_steps=24,
            input_spec=input_spec,
        )
        formal.save(dest / "formal_robustness_certificate.json")
        formal.write_html(dest / "formal_robustness_certificate.html")
        formal_payload = formal.to_dict()
        report.metadata["formal_robustness"] = {
            "verifier": formal.verifier,
            "certified_radius_lower_bound": formal.certified_radius_lower_bound,
            "first_inconclusive_epsilon": formal.first_inconclusive_epsilon,
            "cohort_points": [
                {
                    "epsilon": point.epsilon,
                    "n_certified": point.n_certified,
                    "n_samples": point.n_samples,
                    "certified_accuracy": point.certified_accuracy,
                }
                for point in formal.cohort_points
            ],
        }
        try:
            mechanism_node = _formal_mechanism_node(model)
            formal_mechanism = FormalCausalMechanismVerifier(
                model, model_name=model_name
            ).certify(
                focus,
                target=target,
                mechanism=MechanismPropertySpec(
                    node_id=mechanism_node,
                    reduction="flatten",
                    max_relative_l2_drift=0.25,
                    min_cosine=0.95,
                    top_k=8,
                    require_top_k_stability=True,
                ),
                causal=CausalEffectPropertySpec(
                    node_id=mechanism_node,
                    baseline=0.0,
                    auto_top_k=4,
                    max_screened_coordinates=256,
                    max_relative_l2_drift=0.50,
                    min_cosine=0.80,
                    min_sign_stable_fraction=1.0,
                ),
                epsilon_grid=(
                    0.0,
                    0.00001,
                    0.00005,
                    0.0001,
                    0.00025,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                ),
                clip_min=clip_min,
                clip_max=clip_max,
                radius_search_upper=0.25,
                radius_search_steps=20,
                cohort_x=batch_x if batch_y is not None else None,
                cohort_y=(
                    None if batch_y is None else torch.as_tensor(batch_y)
                ),
                selected_cohort_indices=(
                    None
                    if batch_y is None
                    else list(range(int(batch_x.shape[0])))
                ),
                cohort_epsilon_grid=(0.0, 0.00001),
                input_spec=input_spec,
                calibration_x=calibration_x,
                calibration_sample_ids=calibration_sample_ids,
                calibration_labels=calibration_y,
                calibration_disjoint_from_evaluation=(
                    calibration_disjoint_from_evaluation
                ),
                evaluation_sample_ids=evaluation_sample_ids,
            )
            formal_mechanism.save(
                dest / "formal_causal_mechanism_certificate.json"
            )
            formal_mechanism.write_html(
                dest / "formal_causal_mechanism_certificate.html"
            )
            formal_mechanism_payload = formal_mechanism.to_dict()
            report.metadata["formal_causal_mechanism"] = {
                "verifier": formal_mechanism.verifier,
                "mechanism_node": mechanism_node,
                "decision_radius_lower_bound": (
                    formal_mechanism.decision_radius_lower_bound
                ),
                "representation_radius_lower_bound": (
                    formal_mechanism.representation_radius_lower_bound
                ),
                "representation_geometry_radius_lower_bound": (
                    formal_mechanism.representation_geometry_radius_lower_bound
                ),
                "representation_top_k_radius_lower_bound": (
                    formal_mechanism.representation_top_k_radius_lower_bound
                ),
                "causal_effect_radius_lower_bound": (
                    formal_mechanism.causal_effect_radius_lower_bound
                ),
                "causal_geometry_radius_lower_bound": (
                    formal_mechanism.causal_geometry_radius_lower_bound
                ),
                "causal_sign_radius_lower_bound": (
                    formal_mechanism.causal_sign_radius_lower_bound
                ),
                "joint_radius_lower_bound": formal_mechanism.joint_radius_lower_bound,
                "selected_causal_coordinates": (
                    formal_mechanism.selected_causal_coordinates
                ),
                "cohort_points": [
                    {
                        "epsilon": point.epsilon,
                        "n_samples": point.n_samples,
                        "n_decision_certified": point.n_decision_certified,
                        "n_representation_certified": (
                            point.n_representation_certified
                        ),
                        "n_causal_effect_certified": (
                            point.n_causal_effect_certified
                        ),
                        "n_jointly_certified": point.n_jointly_certified,
                    }
                    for point in formal_mechanism.cohort_points
                ],
            }
        except Exception as error:  # noqa: BLE001
            _write_json(
                dest / "formal_causal_mechanism_status.json",
                {
                    "status": "unsupported",
                    "reason": f"{type(error).__name__}: {error}",
                    "interpretation": (
                        "No formal causal-mechanism claim was emitted. Unsupported is "
                        "not evidence of mechanism instability."
                    ),
                },
            )

    json_path = dest / "adversarial_audit.json"
    html_path = dest / "adversarial_audit.html"
    report.save(json_path)
    report.write_html(html_path)
    return {
        "json": str(json_path),
        "html": str(html_path),
        "empirical_min_flip_eps_linf": report.empirical_min_flip_eps_linf,
        "empirical_min_flip_eps_l2": report.empirical_min_flip_eps_l2,
        "attacks": len(report.attacks_tested),
        "most_vulnerable_concept": report.most_vulnerable_concept,
        "autoattack_robust": report.metadata.get("autoattack_robust"),
        "robust_accuracy": None
        if robust_payload is None
        else robust_payload.get("robust_accuracy"),
        "robustness_profile_html": None
        if robust_payload is None
        else str(dest / "adversarial_robustness_profile.html"),
        "formal_certificate_html": None
        if formal_payload is None
        else str(dest / "formal_robustness_certificate.html"),
        "certified_radius_lower_bound": None
        if formal_payload is None
        else formal_payload.get("certified_radius_lower_bound"),
        "formal_causal_mechanism_html": None
        if formal_mechanism_payload is None
        else str(dest / "formal_causal_mechanism_certificate.html"),
        "joint_mechanism_radius_lower_bound": None
        if formal_mechanism_payload is None
        else formal_mechanism_payload.get("joint_radius_lower_bound"),
    }


def _save_capture(model, sample: torch.Tensor, dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    captured = ModelCaptureRouter(CaptureConfig()).capture(model, (sample,), {})
    ir_path = dest / "mechanism_ir.json"
    captured.ir.save(ir_path)
    summary = captured.summary()
    _write_json(dest / "capture_summary.json", summary)
    return summary


def _save_proof(
    model,
    x: torch.Tensor,
    *,
    target: int,
    class_names,
    feature_names: list[str],
    dest: Path,
    metadata: dict,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    figures = dest / "figures"
    figures.mkdir(exist_ok=True)
    vraven.VRavenTheme.apply()
    report = vraven.explain(
        model,
        x,
        target=target,
        mode="causal",
        class_names=class_names,
        metadata={**metadata, "feature_names": feature_names},
    )
    try:
        report.lrp_result = vraven.explain_lrp(model, x, target=target)
    except Exception as error:  # noqa: BLE001
        print(f"  LRP skipped: {error}")
    try:
        report.perturbation_result = vraven.run_perturbation_stability(
            model, x, target=target, repeats=3, random_seed=17
        )
    except Exception as error:  # noqa: BLE001
        print(f"  Perturbation skipped: {error}")
    try:
        report.counterfactual_result = vraven.generate_counterfactual(
            model,
            x[:1].clone(),
            target_class=0 if target == 1 else 1,
            steps=60,
            learning_rate=0.05,
            l2_weight=0.01,
            top_changes=12,
        )
    except Exception as error:  # noqa: BLE001
        print(f"  Counterfactual skipped: {error}")

    caps = report.capabilities()
    for visual in PROOF_VISUALS:
        if not caps.get(visual, True) and visual not in (
            "relevance_flow",
            "perturbation_stability",
            "counterfactual_delta",
        ):
            continue
        for mode in ("interactive_dark", "publication_light"):
            try:
                fig = report.figure(visual, mode=mode)
                fig.write_html(figures / f"{visual}_{mode}.html", include_plotlyjs=True)
            except Exception as error:  # noqa: BLE001
                print(f"  Proof figure {visual}/{mode} skipped: {error}")

    path = dest / "vraven_proof.html"
    figure_kwargs = {
        "input_attribution": {
            "feature_names": feature_names,
            "top_k": min(30, max(1, len(feature_names))),
        },
        "feature_sequence_heatmap": {
            "feature_names": feature_names,
            "top_features": min(16, max(1, len(feature_names))),
        },
        "decision_sankey": {
            "feature_names": feature_names,
            "class_names": (
                list(class_names)
                if not isinstance(class_names, dict)
                else [class_names[i] for i in sorted(class_names)]
            ),
            "top_features": min(8, max(1, len(feature_names))),
        },
        "counterfactual_delta": {"feature_names": feature_names},
        "latent_flight": {"method": "pca", "max_layers": 4},
    }
    export_html_report(
        report,
        path,
        visuals=list(PROOF_VISUALS),
        mode="interactive_dark",
        figure_kwargs=figure_kwargs,
        theme_toggle=True,
    )
    report.save_trace(dest / "trace.json")
    _write_json(dest / "summary.json", report.summary())
    _write_json(dest / "capabilities.json", caps)
    _write_json(dest / "scientific_evidence_audit.json", report.scientific_audit().as_dict())
    return path


def _save_atlas(model, items, *, dest: Path, class_names, cohort_x: torch.Tensor | None = None):
    dest.mkdir(parents=True, exist_ok=True)
    dataset = vraven.explain_dataset(
        model,
        items,
        mode="gradient",
        class_names=class_names,
        max_samples=min(64, len(items)),
    )
    json_path = dest / "atlas.json"
    dataset.save(json_path)
    for name, figure_fn in (
        ("reasoning_landscape", dataset.reasoning_landscape),
        ("reasoning_similarity", dataset.reasoning_similarity),
        ("reasoning_outliers", dataset.reasoning_outliers),
    ):
        for mode in ("interactive_dark", "publication_light"):
            try:
                figure_fn(mode=mode).write_html(
                    dest / f"{name}_{mode}.html",
                    include_plotlyjs=True,
                )
            except Exception as error:  # noqa: BLE001
                print(f"  Atlas figure {name}/{mode} skipped: {error}")
    _write_json(dest / "summary.json", dataset.summary())
    if cohort_x is not None and int(cohort_x.shape[0]) >= 2:
        try:
            batch_report = vraven.explain(
                model,
                cohort_x[: min(48, int(cohort_x.shape[0]))],
                target=1,
                mode="gradient",
                class_names=class_names,
                metadata={"analysis_semantics": "joint cohort capture for activation communities"},
            )
            for mode in ("interactive_dark", "publication_light"):
                batch_report.community_graph(mode=mode).write_html(
                    dest / f"activation_community_graph_{mode}.html",
                    include_plotlyjs=True,
                )
        except Exception as error:  # noqa: BLE001
            print(f"  Atlas activation community graph skipped: {error}")
    return dataset


def _save_watch(models, focus, *, dest: Path, names: list[str], class_names, target: int) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    evo = vraven.compare_checkpoints(
        models,
        focus,
        target=target,
        checkpoint_names=names,
        mode="gradient",
        class_names=class_names,
    )
    _write_json(dest / "summary.json", evo.summary())
    evo.figure(mode="interactive_dark").write_html(
        dest / "model_evolution_interactive_dark.html",
        include_plotlyjs=True,
    )
    evo.figure(mode="publication_light").write_html(
        dest / "model_evolution_publication_light.html",
        include_plotlyjs=True,
    )
    return evo.summary()


def _save_compare(left, right, items, *, dest, left_name, right_name, class_names) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    comparison = vraven.compare_models(
        left,
        right,
        items,
        mode="fast",
        class_names=class_names,
        left_name=left_name,
        right_name=right_name,
        max_samples=min(64, len(items)),
    )
    _write_json(dest / "summary.json", comparison.summary())
    comparison.alignment(mode="interactive_dark").write_html(
        dest / "model_comparison_interactive_dark.html",
        include_plotlyjs=True,
    )
    comparison.alignment(mode="publication_light").write_html(
        dest / "model_comparison_publication_light.html",
        include_plotlyjs=True,
    )
    return comparison.summary()


def _save_darkside(
    model,
    focus: torch.Tensor,
    *,
    dest: Path,
    layer_names: tuple[str, ...],
    feature_groups: dict[str, list[int] | tuple[int, ...]] | None,
    cohort_x: torch.Tensor | None = None,
    cohort_y: np.ndarray | None = None,
    class_names=None,
    checkpoint_models: list[torch.nn.Module] | None = None,
    checkpoint_names: list[str] | None = None,
) -> dict[str, Any]:
    from vraven.darkside import (
        DarkSideSession,
        InputFeatureGroup,
        TraceDataset,
        UnitGroup,
        analyse_adversarial_rerouting,
        analyse_circuit_multiplicity,
        analyse_cost_evidence_tradeoff,
        analyse_explanation_uncertainty,
        analyse_failure_anatomy,
        analyse_high_confidence_ood,
        analyse_mechanism_transfer,
        analyse_prediction_depth,
        analyse_representation_drift,
        analyse_stability_geometry,
        analyse_topological_shift,
        audit_unused_capacity,
        compare_class_conditionals,
        map_class_conditional_unit_field,
        recommend_features,
    )

    dest.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"written": [], "skipped": {}}
    _write_json(
        dest / "recommend_features.json",
        {
            community: [f.name for f in recommend_features(community)]
            for community in ("security", "malware", "training", "audit", "research")
        },
    )

    session = DarkSideSession(model, focus, target=1)
    circuit = None
    try:
        circuit = session.decision_circuitry(layer_names=layer_names, group_size=4)
        circuit.save_json(dest / "decision_circuitry.json")
        circuit.figure(mode="interactive_dark").write_html(
            dest / "decision_circuitry_interactive_dark.html", include_plotlyjs=True
        )
        status["written"].append("decision_circuitry")
    except Exception as error:  # noqa: BLE001
        status["skipped"]["decision_circuitry"] = str(error)

    try:
        anatomy = session.internal_anatomy(layer_names=layer_names)
        anatomy.save_json(dest / "internal_anatomy.json")
        anatomy.figure(mode="interactive_dark").write_html(
            dest / "internal_anatomy_interactive_dark.html", include_plotlyjs=True
        )
        status["written"].append("internal_anatomy")
    except Exception as error:  # noqa: BLE001
        status["skipped"]["internal_anatomy"] = str(error)

    if circuit is not None:
        try:
            ref_batch = cohort_x if cohort_x is not None else focus
            # Cap bootstrap cohort for edge cards (plan: up to 100 samples).
            if isinstance(ref_batch, torch.Tensor) and ref_batch.shape[0] > 100:
                ref_batch = ref_batch[:100]
            concept_assoc = None
            reverse_atlas = dest.parent / "reverse" / "evidence" / "concept_atlas" / "concept_atlas.json"
            if reverse_atlas.is_file():
                try:
                    atlas = json.loads(reverse_atlas.read_text(encoding="utf-8"))
                    # Pass through only if associations are already keyed; never invent.
                    if isinstance(atlas, dict) and atlas.get("associations"):
                        concept_assoc = atlas["associations"]
                except Exception:  # noqa: BLE001
                    concept_assoc = None
            wiring = session.circuit_wiring(
                circuit,
                enrich=True,
                reference_batch=ref_batch,
                class_names=class_names,
                top_k_intervene=24,
                bootstrap_samples=min(100, int(ref_batch.shape[0]) if hasattr(ref_batch, "shape") else 32),
                concept_associations=concept_assoc,
            )
            wiring.save_json(dest / "circuit_wiring.json")
            wiring.figure(mode="interactive_dark").write_html(
                dest / "circuit_wiring_interactive_dark.html", include_plotlyjs=True
            )
            with suppress(Exception):
                wiring.figure(mode="publication_light").write_html(
                    dest / "circuit_wiring_publication_light.html", include_plotlyjs=True
                )
            status["written"].append("circuit_wiring")
            status["circuit_wiring_cards"] = wiring.metrics.get("edge_cards")
            status["circuit_wiring_intervened"] = wiring.metrics.get("intervened_edges")
        except Exception as error:  # noqa: BLE001
            status["skipped"]["circuit_wiring"] = str(error)

    groups = None
    if feature_groups:
        groups = tuple(
            InputFeatureGroup(tuple(int(i) for i in idxs), label=label)
            for label, idxs in feature_groups.items()
        )
    else:
        dim = int(focus.shape[-1])
        groups = (
            InputFeatureGroup(tuple(range(0, dim // 3)), label="input_block_a"),
            InputFeatureGroup(tuple(range(dim // 3, 2 * dim // 3)), label="input_block_b"),
            InputFeatureGroup(tuple(range(2 * dim // 3, dim)), label="input_block_c"),
        )
    try:
        shortcut = session.shortcut_detection(groups)
        shortcut.save_json(dest / "shortcut_reliance.json")
        shortcut.figure(mode="interactive_dark").write_html(
            dest / "shortcut_reliance_interactive_dark.html", include_plotlyjs=True
        )
        status["written"].append("shortcut_detection")
    except Exception as error:  # noqa: BLE001
        status["skipped"]["shortcut_detection"] = str(error)

    # Synergy on first linear layer units.
    try:
        candidates = []
        for order, (layer_name, module) in enumerate(model.named_modules()):
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                width = int(module.out_channels)
                axis = 1
                family_priority = 0
            elif isinstance(module, torch.nn.Linear):
                width = int(module.out_features)
                axis = -1
                family_priority = 1
            else:
                continue
            if layer_name and width >= 6:
                candidates.append((family_priority, order, layer_name, axis))
        if candidates:
            _, _, layer, axis = min(candidates)
            synergy = session.causal_synergy(
                (
                    UnitGroup(layer, (0, 1), axis=axis, label="community_a"),
                    UnitGroup(layer, (2, 3), axis=axis, label="community_b"),
                    UnitGroup(layer, (4, 5), axis=axis, label="community_c"),
                )
            )
            synergy.save_json(dest / "causal_synergy.json")
            synergy.figure(mode="interactive_dark").write_html(
                dest / "causal_synergy_interactive_dark.html", include_plotlyjs=True
            )
            status["written"].append("causal_synergy")
        else:
            status["skipped"]["causal_synergy"] = (
                "No candidate layer exposes at least six intervention units"
            )
    except Exception as error:  # noqa: BLE001
        status["skipped"]["causal_synergy"] = str(error)

    # Mediation: corrupt a few features and patch.
    try:
        corrupted = focus.clone()
        corrupted[..., : min(4, corrupted.shape[-1])] = 0
        mediation = session.activation_mediation(corrupted, layer_names=layer_names[:3])
        mediation.save_json(dest / "activation_mediation.json")
        mediation.figure(mode="interactive_dark").write_html(
            dest / "activation_mediation_interactive_dark.html", include_plotlyjs=True
        )
        status["written"].append("activation_mediation")
    except Exception as error:  # noqa: BLE001
        status["skipped"]["activation_mediation"] = str(error)

    # Multiplicity: repeat circuitry a few times.
    try:
        masks = {}
        for run in range(3):
            c = session.decision_circuitry(
                layer_names=layer_names[:2] or layer_names,
                group_size=4,
                relative_score_tolerance=0.10 + 0.02 * run,
            )
            # Fall back to metric vector if binary mask absent.
            table = c.tables.get("group_mask") or c.tables.get("retained_mask")
            if table is None and "groups" in c.tables:
                retained = [
                    1 if row.get("retained") or row.get("kept") else 0
                    for row in c.tables["groups"]
                ]
                masks[f"run_{run}"] = np.asarray(retained, dtype=np.int8)
            elif table is not None:
                masks[f"run_{run}"] = np.asarray(table).astype(np.int8).reshape(-1)
        if len(masks) >= 2 and len({m.shape for m in masks.values()}) == 1:
            multi = analyse_circuit_multiplicity(masks)
            multi.save_json(dest / "circuit_multiplicity.json")
            multi.figure(mode="interactive_dark").write_html(
                dest / "circuit_multiplicity_interactive_dark.html", include_plotlyjs=True
            )
            status["written"].append("circuit_multiplicity")
        else:
            status["skipped"]["circuit_multiplicity"] = "Could not align repeated circuit masks"
    except Exception as error:  # noqa: BLE001
        status["skipped"]["circuit_multiplicity"] = str(error)

    # Cohort DarkSide via TraceDataset.
    if cohort_x is not None and cohort_y is not None and len(cohort_y) >= 8:
        try:
            items = [
                vraven.DatasetItem(
                    inputs=cohort_x[i : i + 1],
                    target=1,
                    label=int(cohort_y[i]),
                    sample_id=f"cohort_{i}",
                )
                for i in range(min(48, len(cohort_y)))
            ]
            atlas = vraven.explain_dataset(
                model, items, mode="gradient", class_names=class_names, max_samples=len(items)
            )
            traces = [r.trace for r in atlas.reports]
            labels = [int(cohort_y[i]) for i in range(len(traces))]
            tds = TraceDataset.from_traces(traces, true_labels=labels)
            # Attach fingerprints from atlas when available.
            if getattr(atlas, "fingerprints", None) is not None:
                tds.fingerprints = np.asarray(atlas.fingerprints)

            unique_labels = sorted(set(labels), key=str)
            class_a, class_b = unique_labels[:2]
            truth_positive = tds.true_labels == class_b
            predicted_positive = tds.predicted_labels == class_b
            confusion_counts = {
                "tp": int(np.sum(truth_positive & predicted_positive)),
                "fp": int(np.sum(~truth_positive & predicted_positive)),
                "fn": int(np.sum(truth_positive & ~predicted_positive)),
                "tn": int(np.sum(~truth_positive & ~predicted_positive)),
            }
            status["confusion_counts"] = confusion_counts

            cohort_runners = [
                ("prediction_depth", lambda: analyse_prediction_depth(tds)),
                ("unused_capacity", lambda: audit_unused_capacity(tds)),
                ("cost_evidence_tradeoff", lambda: analyse_cost_evidence_tradeoff(tds)),
                (
                    "class_conditional_internals",
                    lambda: compare_class_conditionals(tds, class_a=class_a, class_b=class_b),
                ),
                (
                    "class_conditional_unit_field",
                    lambda: map_class_conditional_unit_field(tds, class_a=class_a, class_b=class_b),
                ),
            ]
            if confusion_counts["fp"] and confusion_counts["tp"]:
                cohort_runners.append(
                    (
                        "failure_anatomy_fp_vs_tp",
                        lambda: analyse_failure_anatomy(tds, failure="fp", reference="tp"),
                    )
                )
            else:
                status["skipped"]["failure_anatomy_fp_vs_tp"] = (
                    "Not estimable: the cohort has no false positives or no true positives"
                )
            if confusion_counts["fn"] and confusion_counts["tn"]:
                cohort_runners.append(
                    (
                        "failure_anatomy_fn_vs_tn",
                        lambda: analyse_failure_anatomy(tds, failure="fn", reference="tn"),
                    )
                )
            else:
                status["skipped"]["failure_anatomy_fn_vs_tn"] = (
                    "Not estimable: the cohort has no false negatives or no true negatives"
                )

            for name, runner in cohort_runners:
                try:
                    result = runner()
                    result.save_json(dest / f"{name}.json")
                    with suppress(Exception):
                        result.figure(mode="interactive_dark").write_html(
                            dest / f"{name}_interactive_dark.html", include_plotlyjs=True
                        )
                    status["written"].append(name)
                except Exception as error:  # noqa: BLE001
                    status["skipped"][name] = str(error)

            if tds.fingerprints is not None and tds.fingerprints.shape[0] >= 8:
                # Deterministic within-class alternating split prevents class contrast
                # from masquerading as an internal anomaly signal.
                reference_indices = []
                query_indices = []
                for label in unique_labels:
                    indices = np.flatnonzero(tds.true_labels == label)
                    reference_indices.extend(indices[::2].tolist())
                    query_indices.extend(indices[1::2].tolist())
                confidence = np.asarray(
                    [
                        trace.prediction.confidence[0] if trace.prediction.confidence else np.nan
                        for trace in traces
                    ],
                    dtype=np.float64,
                )
                try:
                    ood = analyse_high_confidence_ood(
                        tds.fingerprints[reference_indices],
                        tds.fingerprints[query_indices],
                        query_confidence=confidence[query_indices],
                        confidence_threshold=0.90,
                    )
                    ood.metadata["split_protocol"] = (
                        "within-class deterministic alternating reference/query split"
                    )
                    ood.save_json(dest / "high_confidence_ood.json")
                    ood.figure(mode="interactive_dark").write_html(
                        dest / "high_confidence_ood_interactive_dark.html", include_plotlyjs=True
                    )
                    status["written"].append("high_confidence_ood")
                except Exception as error:  # noqa: BLE001
                    status["skipped"]["high_confidence_ood"] = str(error)

            # Paired numerical stability suite. The perturbation is deliberately
            # labelled as Gaussian sensitivity, not as a semantic or domain-valid attack.
            try:
                clean_x = cohort_x[: len(traces)].detach()
                generator = torch.Generator(device=clean_x.device).manual_seed(24032026)
                noise = 0.05 * torch.randn(
                    clean_x.shape,
                    generator=generator,
                    device=clean_x.device,
                    dtype=clean_x.dtype,
                )
                altered_x = clean_x + noise
                altered_items = [
                    vraven.DatasetItem(
                        inputs=altered_x[i : i + 1],
                        target=1,
                        label=int(cohort_y[i]),
                        sample_id=tds.sample_ids[i],
                    )
                    for i in range(len(traces))
                ]
                altered_atlas = vraven.explain_dataset(
                    model,
                    altered_items,
                    mode="gradient",
                    class_names=class_names,
                    max_samples=len(altered_items),
                )
                altered_tds = TraceDataset.from_traces(
                    [report.trace for report in altered_atlas.reports],
                    true_labels=labels,
                    sample_ids=tds.sample_ids,
                )
                flat_noise = noise.reshape(len(traces), -1).detach().cpu().numpy()
                paired_runners = (
                    ("stability_geometry", lambda: analyse_stability_geometry(tds, altered_tds)),
                    (
                        "adversarial_rerouting",
                        lambda: analyse_adversarial_rerouting(
                            tds,
                            altered_tds,
                            input_l0=np.count_nonzero(flat_noise, axis=1),
                            input_l2=np.linalg.norm(flat_noise, axis=1),
                        ),
                    ),
                    ("topological_shift", lambda: analyse_topological_shift(tds, altered_tds)),
                )
                for name, runner in paired_runners:
                    result = runner()
                    result.metadata["alteration_protocol"] = (
                        "paired deterministic Gaussian input perturbation, sigma=0.05, seed=24032026"
                    )
                    result.save_json(dest / f"{name}.json")
                    result.figure(mode="interactive_dark").write_html(
                        dest / f"{name}_interactive_dark.html", include_plotlyjs=True
                    )
                    status["written"].append(name)
            except Exception as error:  # noqa: BLE001
                status["skipped"]["paired_stability_suite"] = str(error)

            # Repeated explanations of the same decision under the declared local
            # Gaussian variation source, preserving the 1,024-pixel index surface.
            try:
                repeated = []
                for run in range(12):
                    run_generator = torch.Generator(device=focus.device).manual_seed(24032026 + run)
                    perturbed_focus = focus + 0.01 * torch.randn(
                        focus.shape,
                        generator=run_generator,
                        device=focus.device,
                        dtype=focus.dtype,
                    )
                    report = vraven.explain(model, perturbed_focus, target=1, mode="gradient")
                    repeated.append(report.trace.inputs.attribution_values)
                uncertainty = analyse_explanation_uncertainty(
                    repeated,
                    top_k=20,
                    bootstrap_repeats=500,
                    seed=24032026,
                )
                uncertainty.metadata["repetition_protocol"] = (
                    "same input under deterministic Gaussian perturbations, sigma=0.01, 12 runs"
                )
                uncertainty.save_json(dest / "explanation_uncertainty.json")
                uncertainty.figure(mode="interactive_dark").write_html(
                    dest / "explanation_uncertainty_interactive_dark.html", include_plotlyjs=True
                )
                status["written"].append("explanation_uncertainty")
            except Exception as error:  # noqa: BLE001
                status["skipped"]["explanation_uncertainty"] = str(error)

            # A true cross-method audit needs two methods on the same uncompressed
            # surface. CNN LRP is compressed at 256 values while input gradients use
            # all 1,024 pixels, so recording non-estimability is more honest than
            # comparing mismatched coordinates.
            status["skipped"]["cross_method_disagreement"] = (
                "Not estimable on a shared indexed surface: CNN LRP is compressed to "
                "256 values while input gradients retain 1,024 pixels"
            )

            if checkpoint_models and checkpoint_names:
                try:
                    first_model = checkpoint_models[0]
                    first_items = [
                        vraven.DatasetItem(
                            inputs=cohort_x[i : i + 1],
                            target=1,
                            label=int(cohort_y[i]),
                            sample_id=tds.sample_ids[i],
                        )
                        for i in range(len(traces))
                    ]
                    first_atlas = vraven.explain_dataset(
                        first_model,
                        first_items,
                        mode="gradient",
                        class_names=class_names,
                        max_samples=len(first_items),
                    )
                    first_tds = TraceDataset.from_traces(
                        [report.trace for report in first_atlas.reports],
                        true_labels=labels,
                        sample_ids=tds.sample_ids,
                    )
                    drift = analyse_representation_drift(first_tds, tds)
                    drift.metadata["reference_checkpoint"] = checkpoint_names[0]
                    drift.metadata["comparison_checkpoint"] = "final_model"
                    drift.save_json(dest / "representation_drift.json")
                    drift.figure(mode="interactive_dark").write_html(
                        dest / "representation_drift_interactive_dark.html",
                        include_plotlyjs=True,
                    )
                    status["written"].append("representation_drift")
                except Exception as error:  # noqa: BLE001
                    status["skipped"]["representation_drift"] = str(error)

                try:
                    effect_vectors = {}
                    paired_models = list(zip(checkpoint_names, checkpoint_models, strict=True))
                    if paired_models[-1][0] != "final_model":
                        paired_models.append(("final_model", model))
                    for checkpoint_name, checkpoint_model in paired_models:
                        causal = vraven.explain(
                            checkpoint_model, focus, target=1, mode="causal"
                        ).trace.causal_interventions
                        effect_vectors[checkpoint_name] = [
                            intervention.score_delta
                            for intervention in causal
                            if intervention.strategy == "zero"
                        ]
                    transfer = analyse_mechanism_transfer(effect_vectors, top_k=3)
                    transfer.metadata["component_alignment"] = (
                        "identical architecture layer order and zero-ablation intervention"
                    )
                    transfer.save_json(dest / "mechanism_transfer.json")
                    transfer.figure(mode="interactive_dark").write_html(
                        dest / "mechanism_transfer_interactive_dark.html", include_plotlyjs=True
                    )
                    status["written"].append("mechanism_transfer")
                except Exception as error:  # noqa: BLE001
                    status["skipped"]["mechanism_transfer"] = str(error)
        except Exception as error:  # noqa: BLE001
            status["skipped"]["cohort_darkside"] = str(error)

    _write_json(dest / "STATUS.json", status)
    return status


def _focus_from_batch(model, x: torch.Tensor, y: np.ndarray, positive: int = 1) -> tuple[torch.Tensor, int]:
    with torch.no_grad():
        probs = model(x).softmax(-1)[:, positive].cpu().numpy()
    mask = y == positive
    if not mask.any():
        idx = int(np.argmax(probs))
    else:
        scores = np.where(mask, probs, -1.0)
        idx = int(np.argmax(scores))
    return x[idx : idx + 1], idx


def _items(x: torch.Tensor, y: np.ndarray, class_names, prefix: str):
    return [
        vraven.DatasetItem(
            inputs=x[i : i + 1],
            target=1,
            label=class_names[int(y[i])] if isinstance(class_names, dict) else class_names[int(y[i])],
            sample_id=f"{prefix}_{i}",
        )
        for i in range(len(y))
    ]


def run_classifier_suite(
    *,
    name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: np.ndarray,
    feature_names: list[str],
    class_names: dict[int, str],
    input_groups: dict | None = None,
    checkpoint_loader: Callable[[], tuple[list[torch.nn.Module], list[str]]] | None = None,
    compare_peer: tuple[str, torch.nn.Module] | None = None,
    input_spec: InputDomainSpec | None = None,
    calibration_x: torch.Tensor | None = None,
    calibration_sample_ids: list[str] | None = None,
    calibration_y: torch.Tensor | np.ndarray | None = None,
    calibration_disjoint_from_evaluation: bool | None = None,
    evaluation_sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    print(f"\n=== {name}: full VRAVEN suite ===")
    out_root = OUT / name
    status: dict[str, Any] = {"family": name}
    focus, focus_idx = _focus_from_batch(model, x, y)
    layers = _linear_names(model)
    items = _items(x, y, class_names, name)

    # Capture
    try:
        status["capture"] = _save_capture(model, focus, out_root / "capture")
        print("  Capture OK")
    except Exception:
        status["capture_error"] = traceback.format_exc()
        print(status["capture_error"])

    # Reverse
    try:
        config = DecompileConfig()
        if input_groups:
            config.promote_association_labels = True
        result = VRavenDecompiler(config).reverse(
            model,
            (x,),
            labels=y,
            focus_args=(focus,),
            target=1,
            class_names=class_names,
            input_groups=input_groups,
        )
        status["reverse"] = _save_reverse(result, out_root / "reverse")
        print(f"  Reverse OK → {out_root / 'reverse'}")
    except Exception as error:
        status["reverse_error"] = traceback.format_exc()
        _write_json(
            out_root / "reverse" / "STATUS.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "note": (
                    "Capture / Proof / Atlas / Watch / DarkSide may still succeed. "
                    "This records a Reverse (compiler-decompiler) failure on this architecture."
                ),
            },
        )
        print(f"  Reverse FAILED: {type(error).__name__}: {error}")

    # Adversarial stress audit (input-space; independent of Reverse success)
    try:
        status["adversarial_audit"] = _save_adversarial_audit(
            model,
            focus,
            out_root / "reverse",
            model_name=name,
            feature_names=feature_names,
            target=1,
            batch_x=x,
            batch_y=y,
            autoattack_epsilon=0.1,
            robust_max_samples=12,
            input_spec=input_spec,
            calibration_x=calibration_x,
            calibration_sample_ids=calibration_sample_ids,
            calibration_y=calibration_y,
            calibration_disjoint_from_evaluation=(
                calibration_disjoint_from_evaluation
            ),
            evaluation_sample_ids=evaluation_sample_ids,
        )
        print(
            "  Adversarial audit OK · "
            f"min flip ε∞={status['adversarial_audit'].get('empirical_min_flip_eps_linf')} · "
            f"ensemble-lite no-flip={status['adversarial_audit'].get('autoattack_robust')} · "
            f"batch rob.acc={status['adversarial_audit'].get('robust_accuracy')} · "
            f"certified ε∞≥{status['adversarial_audit'].get('certified_radius_lower_bound')}"
        )
    except Exception:
        status["adversarial_audit_error"] = traceback.format_exc()
        print(status["adversarial_audit_error"])

    # Proof
    try:
        _save_proof(
            model,
            focus,
            target=1,
            class_names=class_names,
            feature_names=feature_names,
            dest=out_root / "proof",
            metadata={"model": name, "focus_index": focus_idx},
        )
        status["proof"] = str(out_root / "proof" / "vraven_proof.html")
        print("  Proof OK")
    except Exception:
        status["proof_error"] = traceback.format_exc()
        print(status["proof_error"])

    # Atlas
    try:
        _save_atlas(
            model,
            items,
            dest=out_root / "atlas",
            class_names=class_names,
            cohort_x=x,
        )
        status["atlas"] = str(out_root / "atlas" / "atlas.json")
        print("  Atlas OK")
    except Exception:
        status["atlas_error"] = traceback.format_exc()
        print(status["atlas_error"])

    # Watch
    watch_dir = out_root / "watch"
    ckpt_models: list[torch.nn.Module] = []
    ckpt_names: list[str] = []
    if checkpoint_loader is not None:
        try:
            ckpt_models, ckpt_names = checkpoint_loader()
            if len(ckpt_models) >= 2:
                status["watch"] = _save_watch(
                    ckpt_models,
                    focus,
                    dest=watch_dir,
                    names=ckpt_names,
                    class_names=class_names,
                    target=1,
                )
                print("  Watch OK")
            else:
                note = {"status": "blocked", "reason": "Fewer than 2 checkpoints"}
                _write_json(watch_dir / "STATUS.json", note)
                status["watch"] = note
        except Exception:
            status["watch_error"] = traceback.format_exc()
            print(status["watch_error"])
    else:
        note = {
            "status": "blocked",
            "reason": "No ordered training checkpoints packaged for this model.",
        }
        _write_json(watch_dir / "STATUS.json", note)
        status["watch"] = note

    # Compare peer
    if compare_peer is not None:
        peer_name, peer_model = compare_peer
        try:
            status["compare"] = _save_compare(
                model,
                peer_model,
                items,
                dest=out_root / "compare",
                left_name=name,
                right_name=peer_name,
                class_names=class_names,
            )
            print(f"  Compare OK vs {peer_name}")
        except Exception:
            status["compare_error"] = traceback.format_exc()
            print(status["compare_error"])

    # DarkSide
    try:
        status["darkside"] = _save_darkside(
            model,
            focus,
            dest=out_root / "darkside",
            layer_names=layers,
            feature_groups=input_groups,
            cohort_x=x,
            cohort_y=y,
            class_names=class_names,
            checkpoint_models=ckpt_models,
            checkpoint_names=ckpt_names,
        )
        print(f"  DarkSide written={status['darkside'].get('written')}")
    except Exception:
        status["darkside_error"] = traceback.format_exc()
        print(status["darkside_error"])

    return status


def load_probe(name: str) -> tuple[torch.nn.Module, torch.Tensor, np.ndarray, list[str], dict, dict, Callable]:
    model_dir = MODELS / name
    payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    arch = payload["architecture"]
    model = _build_from_arch(arch).eval()
    model.load_state_dict(payload["model_state"])

    ref = np.load(model_dir / "reference.npz", allow_pickle=True)
    if "images" in ref.files:
        x = torch.from_numpy(np.asarray(ref["images"], dtype=np.float32))
    else:
        x = torch.from_numpy(np.asarray(ref["features"], dtype=np.float32))
    y = np.asarray(ref["labels"], dtype=np.int64)
    feature_names = [str(v) for v in payload.get("feature_names", [])]
    if not feature_names:
        if x.ndim == 4:
            _, _, h, w = x.shape
            feature_names = [f"px_{i}_{j}" for i in range(h) for j in range(w)]
        else:
            feature_names = [f"f{i}" for i in range(int(x.shape[-1]))]
    class_list = payload["class_names"]
    class_names = {0: class_list[0], 1: class_list[1]}
    groups = {k: list(v) for k, v in payload.get("input_groups", {}).items()}

    def checkpoint_loader() -> tuple[list[torch.nn.Module], list[str]]:
        ckpts = sorted((model_dir / "checkpoints").glob("epoch_*.pt"))
        picks = [ckpts[0], ckpts[len(ckpts) // 2], ckpts[-1]] if len(ckpts) >= 3 else ckpts
        models = []
        names = []
        for path in picks:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            m = _build_from_arch(blob["architecture"]).eval()
            m.load_state_dict(blob["model_state"])
            models.append(m)
            names.append(path.stem)
        return models, names

    return model, x, y, feature_names, class_names, groups, checkpoint_loader


PROBE_NAMES_TABULAR = (
    "phishing_url_mlp",
    "phishing_url_cnn1d",
    "medical_triage_mlp",
    "network_ids_mlp",
)
PROBE_NAMES_NOVEL = (
    "image_pattern_cnn2d",
    "image_pattern_diffusion",
    "image_pattern_vit",
    "sensor_transformer",
)


def run_probes(*, only_novel: bool = False) -> dict[str, Any]:
    status: dict[str, Any] = {}
    phishing_mlp = None
    phishing_cnn = None
    image_cnn = None
    image_vit = None
    names = PROBE_NAMES_NOVEL if only_novel else PROBE_NAMES_TABULAR + PROBE_NAMES_NOVEL
    for name in names:
        if not (MODELS / name / "model.pt").exists():
            hint = (
                "run scripts/train_vraven_novel_probes.py"
                if name in PROBE_NAMES_NOVEL
                else "run scripts/train_vraven_probe_models.py"
            )
            status[name] = {"status": "missing", "hint": hint}
            continue
        model, x, y, feats, classes, groups, ckpt_loader = load_probe(name)
        if name == "phishing_url_mlp":
            phishing_mlp = model
        if name == "phishing_url_cnn1d":
            phishing_cnn = (name, model)
        if name == "image_pattern_cnn2d":
            image_cnn = (name, model, x, y, classes)
        if name == "image_pattern_vit":
            image_vit = (name, model)
        status[name] = run_classifier_suite(
            name=name,
            model=model,
            x=x,
            y=y,
            feature_names=feats,
            class_names=classes,
            input_groups=groups,
            checkpoint_loader=ckpt_loader,
            compare_peer=None,
        )

    if not only_novel and phishing_mlp is not None and phishing_cnn is not None:
        try:
            model, x, y, feats, classes, groups, _ = load_probe("phishing_url_mlp")
            peer_name, peer_model = phishing_cnn
            items = _items(x, y, classes, "phishing_compare")
            status["phishing_compare"] = _save_compare(
                model,
                peer_model,
                items,
                dest=OUT / "phishing_url_mlp" / "compare",
                left_name="phishing_url_mlp",
                right_name=peer_name,
                class_names=classes,
            )
            print("  Phishing Compare OK (mlp vs cnn1d)")
        except Exception:
            status["phishing_compare_error"] = traceback.format_exc()
            print(status["phishing_compare_error"])

    if image_cnn is not None and image_vit is not None:
        try:
            left_name, left_model, x, y, classes = image_cnn
            peer_name, peer_model = image_vit
            items = _items(x, y, classes, "image_compare")
            status["image_compare"] = _save_compare(
                left_model,
                peer_model,
                items,
                dest=OUT / "image_pattern_cnn2d" / "compare",
                left_name=left_name,
                right_name=peer_name,
                class_names=classes,
            )
            print("  Image Compare OK (cnn2d vs vit)")
        except Exception:
            status["image_compare_error"] = traceback.format_exc()
            print(status["image_compare_error"])
    return status


def run_kronodroid() -> dict[str, Any]:
    status: dict[str, Any] = {"family": "kronodroid", "models": {}}
    split = np.load(MODELS / "kronodroid" / "split.npz", allow_pickle=True)
    x_test = np.asarray(split["x_test"], dtype=np.float32)
    y_test = np.asarray(split["y_test"], dtype=np.int64)
    feature_names = [str(v) for v in split["feature_names"]]
    class_names = {0: "Benign", 1: "Malware"}
    benign = np.where(y_test == 0)[0][:32]
    malware = np.where(y_test == 1)[0][:32]
    reference_idx = np.concatenate([benign, malware])
    x = torch.from_numpy(x_test[reference_idx])
    y = y_test[reference_idx]
    loaded: dict[str, torch.nn.Module] = {}

    for arch, builder in MODEL_BUILDERS.items():
        ckpt = torch.load(MODELS / "kronodroid" / arch / "model.pt", map_location="cpu", weights_only=True)
        model = builder(int(ckpt["input_dim"])).eval()
        model.load_state_dict(ckpt["model_state"])
        loaded[arch] = model
        status["models"][arch] = run_classifier_suite(
            name=f"kronodroid/{arch}",
            model=model,
            x=x,
            y=y,
            feature_names=feature_names,
            class_names=class_names,
            input_groups=KRONO_GROUPS,
            checkpoint_loader=None,
            compare_peer=None,
        )

    # Family compare
    try:
        items = _items(x, y, class_names, "kronodroid")
        status["compare"] = _save_compare(
            loaded["cnn1d"],
            loaded["mlp"],
            items,
            dest=OUT / "kronodroid" / "compare",
            left_name="kronodroid_cnn1d",
            right_name="kronodroid_mlp",
            class_names=class_names,
        )
        print("\n=== Kronodroid Compare OK ===")
    except Exception:
        status["compare_error"] = traceback.format_exc()
        print(status["compare_error"])
    return status


def run_winlolbin() -> dict[str, Any]:
    model_dir = MODELS / "winlolbin_mlp"
    payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    arch = payload["architecture"]
    model = LolbinMLP(
        input_dim=int(arch["input_dim"]),
        hidden_sizes=tuple(arch["hidden_sizes"]),
        num_classes=int(arch["num_classes"]),
    ).eval()
    model.load_state_dict(payload["model_state"])
    sample = np.load(model_dir / "explain_sample.npz")
    features = np.asarray(sample["features_scaled"], dtype=np.float32)
    labels = np.asarray(sample["labels"], dtype=np.int64)
    feature_names = [str(v) for v in payload["feature_names"]]
    class_names = {0: "Benign", 1: "Malware"}
    idx0 = np.flatnonzero(labels == 0)[:32]
    idx1 = np.flatnonzero(labels == 1)[:32]
    keep = np.concatenate([idx0, idx1])
    x = torch.from_numpy(features[keep])
    y = labels[keep]
    return run_classifier_suite(
        name="winlolbin_mlp",
        model=model,
        x=x,
        y=y,
        feature_names=feature_names,
        class_names=class_names,
        input_groups={
            "Command structure": list(range(0, 16)),
            "LOLBIN flags": list(range(16, 38)),
            "Process / path / parent": list(range(38, 51)),
            "Signing / network / integrity": list(range(51, 55)),
        },
        checkpoint_loader=None,
    )


def run_ae(name: str, *, compare_peer_name: str | None = None) -> dict[str, Any]:
    """Run the full VRAVEN suite on a reconstruction AE via its MSE anomaly score."""

    dest = OUT / name
    if (dest / "STATUS.json").exists():
        (dest / "STATUS.json").unlink()

    bundle = load_ae_anomaly_bundle(MODELS / name)
    compare_peer = None
    if compare_peer_name is not None:
        peer = load_ae_anomaly_bundle(MODELS / compare_peer_name)
        compare_peer = (compare_peer_name, peer["model"])

    status = run_classifier_suite(
        name=name,
        model=bundle["model"],
        x=bundle["x"],
        y=bundle["y"],
        feature_names=bundle["feature_names"],
        class_names=bundle["class_names"],
        input_groups=bundle["input_groups"],
        checkpoint_loader=None,
        compare_peer=compare_peer,
    )
    status["ae"] = {
        "focus_binary": bundle["focus_binary"],
        "threshold": bundle["threshold"],
        "matrix_path": bundle["matrix_path"],
        "injection_matrix": bundle["injection_matrix"],
        "score": "reconstruction_mse_log1p_vs_threshold",
        "class_names": bundle["class_names"],
    }
    _write_json(dest / "AE_RUNTIME.json", status["ae"])
    return status


def write_index(status: dict[str, Any]) -> Path:
    _write_json(OUT / "RUN_STATUS.json", status)
    from build_vraven_gallery import main as build_gallery

    build_gallery()
    return OUT / "INDEX.html"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    only_novel = "--only-novel" in args
    only_ae = "--only-ae" in args
    refresh_all = "--refresh-all" in args

    if refresh_all and OUT.exists():
        import shutil

        for child in OUT.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    OUT.mkdir(parents=True, exist_ok=True)

    if only_ae:
        status = {
            "vraven_version": getattr(vraven, "__version__", "unknown"),
            "mode": "only_ae",
            "windows10_full_benign_ae": run_ae(
                "windows10_full_benign_ae",
                compare_peer_name="win2016_dc",
            ),
            "win2016_dc": run_ae("win2016_dc"),
        }
    elif only_novel:
        status = {
            "vraven_version": getattr(vraven, "__version__", "unknown"),
            "mode": "only_novel",
            "probes": run_probes(only_novel=True),
        }
    else:
        status = {
            "vraven_version": getattr(vraven, "__version__", "unknown"),
            "probes": run_probes(only_novel=False),
            "kronodroid": run_kronodroid(),
            "winlolbin_mlp": run_winlolbin(),
            "windows10_full_benign_ae": run_ae(
                "windows10_full_benign_ae",
                compare_peer_name="win2016_dc",
            ),
            "win2016_dc": run_ae("win2016_dc"),
        }
    index = write_index(status)
    print(f"\nMaster index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
