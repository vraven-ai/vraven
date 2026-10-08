"""Feature catalogue and community-specific selection guidance."""

from __future__ import annotations

from .contracts import FeatureRecommendation

FEATURES: dict[str, FeatureRecommendation] = {
    "decision_circuitry": FeatureRecommendation(
        "decision_circuitry", "high", "Find an approximate sparse causal core for one prediction.",
        ("live model", "sample", "target", "candidate layers"), "high", "intervention-based",
    ),
    "shortcut_detection": FeatureRecommendation(
        "shortcut_detection", "high", "Test whether narrow, brittle input groups dominate a prediction.",
        ("live model", "sample", "declared feature groups"), "medium", "mixed observational/intervention",
    ),
    "class_conditional_internals": FeatureRecommendation(
        "class_conditional_internals", "medium", "Find internal communities that flip between classes.",
        ("cohort traces", "labels"), "medium", "observational",
    ),
    "prediction_depth": FeatureRecommendation(
        "prediction_depth", "medium", "Locate the earliest stable class-decodable layer and potential early shortcut decisions.",
        ("labelled reference traces",), "low", "probe-based observational",
    ),
    "representation_drift": FeatureRecommendation(
        "representation_drift", "high", "Measure geometric internal change across checkpoints or domains.",
        ("aligned reference traces", "aligned comparison traces"), "medium", "observational",
    ),
    "failure_anatomy": FeatureRecommendation(
        "failure_anatomy", "high", "Localise internal differences between failures and correct predictions.",
        ("labelled cohort traces",), "medium", "observational",
    ),
    "stability_geometry": FeatureRecommendation(
        "stability_geometry", "high", "Locate the first internal path that collapses under perturbation.",
        ("paired clean traces", "paired perturbed traces"), "medium", "observational",
    ),
    "cross_method_disagreement": FeatureRecommendation(
        "cross_method_disagreement", "high", "Expose where explanation methods disagree in sign and rank.",
        ("aligned method vectors",), "low", "method uncertainty",
    ),
    "unused_capacity": FeatureRecommendation(
        "unused_capacity", "medium", "Find dead, dormant-sensitive and high-impact units.",
        ("dataset traces with unit summaries",), "low", "observational",
    ),
    "adversarial_rerouting": FeatureRecommendation(
        "adversarial_rerouting", "high", "Measure how evasion redirects internal evidence pathways.",
        ("paired clean traces", "paired altered traces"), "medium", "paired observational",
    ),
    "cost_evidence_tradeoff": FeatureRecommendation(
        "cost_evidence_tradeoff", "medium", "Find layers with high runtime and little measured evidence.",
        ("timed dataset traces",), "low", "performance diagnostic",
    ),
    "circuit_multiplicity": FeatureRecommendation(
        "circuit_multiplicity", "high", "Measure whether repeated valid circuit searches recover one stable core or several alternatives.",
        ("aligned binary circuit masks",), "low", "empirical mechanism multiplicity",
    ),
    "causal_synergy": FeatureRecommendation(
        "causal_synergy", "high", "Detect cooperative, redundant and compensatory internal groups.",
        ("live model", "sample", "candidate groups"), "very high", "intervention-based",
    ),
    "activation_mediation": FeatureRecommendation(
        "activation_mediation", "high", "Test where clean internal state repairs a corrupted run.",
        ("live model", "clean/corrupt pair", "layers"), "high", "intervention-based",
    ),
    "topological_shift": FeatureRecommendation(
        "topological_shift", "research", "Detect connectivity and loop changes in representation geometry.",
        ("aligned cohort traces",), "medium to high", "observational geometry",
    ),
    "explanation_uncertainty": FeatureRecommendation(
        "explanation_uncertainty", "high", "Quantify rank, sign and selection uncertainty over repeats.",
        ("repeated aligned explanations",), "medium", "empirical uncertainty",
    ),
    "mechanism_transfer": FeatureRecommendation(
        "mechanism_transfer", "research", "Measure circuit-effect overlap across models or checkpoints.",
        ("aligned intervention effect maps",), "medium", "comparative",
    ),
    "high_confidence_ood": FeatureRecommendation(
        "high_confidence_ood", "high", "Flag confident predictions with atypical internal fingerprints.",
        ("reference fingerprints", "query fingerprints"), "low", "anomaly screening",
    ),
    "internal_anatomy": FeatureRecommendation(
        "internal_anatomy", "high", "Print and plot live Linear-layer W×a anatomy for one sample.",
        ("live model", "sample", "target"), "low", "local measured anatomy",
    ),
    "circuit_wiring": FeatureRecommendation(
        "circuit_wiring", "high", "Wire retained ε-core groups with measured weight×activation edge mass.",
        ("live model", "sample", "decision_circuitry result"), "medium", "measured coupling of an intervention core",
    ),
    "class_conditional_unit_field": FeatureRecommendation(
        "class_conditional_unit_field", "medium", "Map a dense layer×unit class-contrast field.",
        ("cohort traces with unit influences", "labels"), "medium", "observational",
    ),
}


COMMUNITY_PROFILES: dict[str, tuple[str, ...]] = {
    "security": (
        "shortcut_detection",
        "adversarial_rerouting",
        "failure_anatomy",
        "high_confidence_ood",
        "decision_circuitry",
        "circuit_multiplicity",
        "activation_mediation",
    ),
    "malware": (
        "shortcut_detection",
        "adversarial_rerouting",
        "failure_anatomy",
        "representation_drift",
        "decision_circuitry",
        "cross_method_disagreement",
    ),
    "training": (
        "representation_drift",
        "prediction_depth",
        "unused_capacity",
        "class_conditional_internals",
        "failure_anatomy",
        "cost_evidence_tradeoff",
    ),
    "model_engineering": (
        "unused_capacity",
        "cost_evidence_tradeoff",
        "stability_geometry",
        "representation_drift",
        "mechanism_transfer",
    ),
    "research": (
        "decision_circuitry",
        "circuit_wiring",
        "internal_anatomy",
        "circuit_multiplicity",
        "causal_synergy",
        "activation_mediation",
        "topological_shift",
        "class_conditional_unit_field",
        "cross_method_disagreement",
        "explanation_uncertainty",
        "mechanism_transfer",
    ),
    "audit": (
        "failure_anatomy",
        "shortcut_detection",
        "explanation_uncertainty",
        "cross_method_disagreement",
        "high_confidence_ood",
        "internal_anatomy",
        "circuit_wiring",
    ),
}


def recommend_features(community: str) -> list[FeatureRecommendation]:
    """Return a focused set rather than running every expensive analysis."""

    key = community.strip().lower().replace(" ", "_")
    if key not in COMMUNITY_PROFILES:
        supported = ", ".join(sorted(COMMUNITY_PROFILES))
        raise KeyError(f"Unknown community {community!r}. Supported: {supported}")
    return [FEATURES[name] for name in COMMUNITY_PROFILES[key]]
