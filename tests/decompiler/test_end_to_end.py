from __future__ import annotations

import json

from vraven.decompiler import DecompileConfig, VRavenDecompiler
from vraven.decompiler.feature_audit import build_reverse_feature_audits


def test_end_to_end_decision_proof_and_visuals(tmp_path, tiny_model, tiny_data):
    inputs, labels = tiny_data
    config = DecompileConfig.quick()
    config.causal.bootstrap_repeats = 8
    config.causal.max_samples = 24
    config.grounding_permutations = 8
    config.run_profiler = False
    config.counterfactual.maximum_verified_candidates = 4
    config.programme.interventional_weight = 16.0
    result = VRavenDecompiler(config).reverse(
        tiny_model,
        (inputs,),
        labels=labels,
        focus_args=(inputs[:1],),
        class_names={0: "negative", 1: "positive"},
        input_groups={
            "joint signal": (0, 1),
            "difference signal": (2, 3),
            "distractors": (4,),
            "inhibitor": (5,),
        },
    )
    assert result.captured.ir.coverage.numeric_equivalence_passed is True
    assert result.programme.rules
    assert result.programme.fidelity.validation_decision_agreement >= 0.7
    assert result.certificate.validity_envelope.unresolved_operations is not None
    audits = build_reverse_feature_audits(result)
    assert set(audits) == {
        "model_xray",
        "concept_genome",
        "causal_proof",
        "programme_proof",
        "contrastive_proof",
        "counterfactual_twin",
        "dark_matter",
    }
    assert 0.0 <= audits["programme_proof"].metrics["local_cancellation_index"] <= 1.0
    assert result.contrastive.metadata["decisive_depth_definition"].startswith(
        "earliest evidence-bearing depth"
    )

    output = result.save(tmp_path / "proof")
    assert (output / "mechanism_ir.json").exists()
    assert (output / "executable_programme.json").exists()
    assert (output / "mechanism_certificate.json").exists()

    visuals = (
        "model_xray",
        "concept_genome",
        "causal_proof",
        "programme_proof",
        "contrastive_proof",
        "counterfactual_twin",
        "dark_matter",
    )
    for visual in visuals:
        dark = result.figure(visual, mode="interactive_dark")
        light = result.figure(visual, mode="publication_light")
        assert dark.layout.paper_bgcolor != light.layout.paper_bgcolor
        assert light.layout.paper_bgcolor == "#FFFFFF"
        assert (
            dark.layout.meta["scientific"]["visual_id"]
            == light.layout.meta["scientific"]["visual_id"]
        )
        assert "feature_audit" in dark.layout.meta["scientific"]

    report = result.write_report(tmp_path / "decision_proof.html", include_plotlyjs=False)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Executable decision programme" in text
    assert "Raven Dark Matter" in text
    assert "Recovered scientific result" in text
    assert "Primary scientific result" in text
    assert "Scope, unresolved evidence and next validation targets" in text

    json_report = result.write_report(
        tmp_path / "decision_proof.json",
        format="json",
    )
    json_payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert json_payload["report_type"] == "vraven_decision_proof"
    assert json_payload["summary"]["model"] == type(tiny_model).__name__
    assert set(json_payload["feature_audits"]) == set(audits)

    feature_report = result.write_feature_report(
        "causal_proof",
        tmp_path / "causal_feature.html",
        include_plotlyjs=False,
    )
    feature_text = feature_report.read_text(encoding="utf-8")
    assert "Primary scientific result" in feature_text
    assert "Direct interventions isolate a distributed causal submechanism" in feature_text
    assert "Scope and limitations" in feature_text
    assert "interval exclusion rate" in feature_text

    feature_json = result.write_feature_report(
        "causal_proof",
        tmp_path / "causal_feature.json",
        format="json",
    )
    feature_payload = json.loads(feature_json.read_text(encoding="utf-8"))
    assert feature_payload["report_type"] == "vraven_feature_report"
    assert feature_payload["visual"] == "causal_proof"
