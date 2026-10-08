from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vraven.core.trace import CausalInterventionTrace, LayerTrace, NeuralTrace, PredictionTrace
from vraven.metrics.evidence_audit import (
    evidence_concentration,
    observational_causal_concordance,
    scientific_evidence_audit,
)
from vraven.report.report import VRavenReport


def _trace() -> NeuralTrace:
    signed = (3.0, 2.0, 1.0, -1.0)
    layers = [
        LayerTrace(
            name=f"layer_{index}",
            layer_type="Linear",
            execution_index=index,
            activation_gradient_signed=value,
            activation_gradient_influence=1.0,
        )
        for index, value in enumerate(signed)
    ]
    interventions = [
        CausalInterventionTrace(
            layer_name=f"layer_{index}",
            strategy="zero",
            baseline_target_score=4.0,
            intervened_target_score=4.0 - causal,
            score_delta=causal,
            relative_delta=causal / 4.0,
            prediction_before=1,
            prediction_after=1,
            target_index=1,
            successful=True,
        )
        for index, causal in enumerate((30.0, 20.0, 10.0))
    ]
    return NeuralTrace(
        layers=layers,
        prediction=PredictionTrace(
            output_kind="multiclass",
            logits_shape=(2, 2),
            predicted_classes=[1, 1],
            predictive_entropy=[0.2, 0.4],
            margin=[0.7, 0.5],
        ),
        mode="causal",
        causal_interventions=interventions,
    )


def _data_signature(figure) -> str:
    payload = json.loads(figure.to_json())
    return json.dumps(
        [
            {key: trace.get(key) for key in ("type", "x", "y", "customdata", "text") if key in trace}
            for trace in payload["data"]
        ],
        sort_keys=True,
    )


def test_evidence_concentration_has_known_uniform_solution() -> None:
    audit = evidence_concentration(_trace())
    assert np.allclose(audit.mass_share, np.full(4, 0.25))
    assert audit.normalised_entropy == 1.0
    assert audit.effective_layers == 4.0
    assert audit.participation_ratio == 4.0
    assert audit.herfindahl_concentration == 0.25
    assert audit.normalised_gini == 0.0
    assert audit.half_mass_layer == "layer_1"
    assert audit.eighty_mass_layer == "layer_3"


def test_observational_causal_concordance_is_scale_invariant() -> None:
    audit = observational_causal_concordance(_trace())
    assert audit.layer_names == ("layer_0", "layer_1", "layer_2")
    assert audit.spearman_rho == 1.0
    assert audit.sign_agreement == 1.0
    assert np.isclose(audit.cosine_similarity, 1.0)
    assert np.isclose(audit.normalised_l1_discrepancy, 0.0)
    assert np.isclose(audit.permutation_p_value, 1 / 3)
    assert audit.permutation_method == "exact (6 permutations)"


def test_scientific_audit_keeps_dimensions_separate() -> None:
    audit = scientific_evidence_audit(_trace())
    assert audit.evidence_tier == "targeted_interventional"
    assert audit.gradient_layer_count == 4
    assert audit.zero_ablation_layer_count == 3
    assert audit.causal_coverage == 0.75
    assert np.isclose(audit.median_predictive_entropy, 0.3)
    assert np.isclose(audit.median_prediction_margin, 0.6)
    assert "trust" not in audit.as_dict()


def test_mechanism_concordance_is_dual_mode_and_scientifically_labelled() -> None:
    report = VRavenReport(_trace())
    light = report.mechanism_concordance(mode="publication_light")
    dark = report.mechanism_concordance(mode="interactive_dark")
    assert light.layout.paper_bgcolor == "#FFFFFF"
    assert dark.layout.paper_bgcolor != "#FFFFFF"
    assert _data_signature(light) == _data_signature(dark)
    metadata = light.layout.meta["scientific"]
    assert metadata["visual_id"] == "vraven.mechanism_concordance"
    assert "not recovery of a unique causal mechanism" in " ".join(metadata["limitations"])


def test_html_report_contains_audit_ledger_and_can_disable_it(tmp_path: Path) -> None:
    report = VRavenReport(_trace())
    audited = tmp_path / "audited.html"
    report.export_html(
        audited,
        visuals=("mechanism_concordance",),
        include_scientific_audit=True,
    )
    html = audited.read_text(encoding="utf-8")
    assert "Scientific Evidence Audit" in html
    assert "Hypothesis and evidence ledger" in html
    assert "Direction agrees" in html
    assert "single explanation-confidence or trust score" in html

    compact = tmp_path / "compact.html"
    report.export_html(
        compact,
        visuals=("mechanism_concordance",),
        include_scientific_audit=False,
    )
    assert "Scientific Evidence Audit" not in compact.read_text(encoding="utf-8")
