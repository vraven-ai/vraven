from __future__ import annotations

import json

import numpy as np
import torch

from vraven.cli import _demo_model, build_parser, main


def test_cli_demo_and_inspect(tmp_path, capsys) -> None:
    trace = tmp_path / "vraven_demo_report.json"
    assert main(
        [
            "demo",
            "--mode",
            "gradient",
            "--samples",
            "4",
            "--report-format",
            "json",
            "--output",
            str(trace),
        ]
    ) == 0
    assert trace.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["vraven_demo_report.json"]
    assert main(["inspect", str(trace)]) == 0
    output = capsys.readouterr().out
    assert "layers_captured" in output


def test_cli_demo_writes_one_html_report(tmp_path) -> None:
    destination = tmp_path / "vraven.html"
    assert main(
        [
            "demo",
            "--mode",
            "fast",
            "--samples",
            "4",
            "--report-format",
            "html",
            "--output",
            str(destination),
        ]
    ) == 0
    assert destination.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["vraven.html"]


def test_cli_explain_model_with_public_parameters(tmp_path) -> None:
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.npy"
    report_path = tmp_path / "explanation.json"
    torch.save(_demo_model().state_dict(), model_path)
    np.save(input_path, np.zeros((2, 12), dtype=np.float32))

    assert main(
        [
            "explain",
            str(model_path),
            "--model-factory",
            "vraven.cli:_demo_model",
            "--input",
            str(input_path),
            "--mode",
            "fast",
            "--target",
            "1",
            "--class-names",
            "class_0,class_1,class_2",
            "--module-policy",
            "leaf",
            "--max-layers",
            "8",
            "--max-units",
            "16",
            "--no-attention",
            "--report-format",
            "json",
            "--output",
            str(report_path),
        ]
    ) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "fast"
    assert payload["prediction"]["target_indices"] == [1, 1]
    assert {path.name for path in tmp_path.iterdir()} == {
        "model.pt",
        "input.npy",
        "explanation.json",
    }


def test_cli_features_and_doctor_are_machine_readable(capsys) -> None:
    assert main(["features"]) == 0
    features = json.loads(capsys.readouterr().out)
    assert "explain" in features["commands"]
    assert "adversarial" in features["commands"]
    assert "certify" in features["commands"]
    assert "dataset" in features["commands"]
    assert "compare" in features["commands"]
    assert "evolution" in features["commands"]
    assert "anatomy" in features["commands"]
    assert "raven" in features["commands"]
    assert features["terminal_ui"]["raven_styles"] == ["still", "blink", "dance"]
    assert features["report_formats"] == ["html", "pdf", "json"]
    assert features["explanation_modes"] == ["fast", "gradient", "causal"]

    assert main(["doctor"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["vraven_package"].endswith("vraven/__init__.py")
    assert doctor["python_executable"]
    assert doctor["vraven_invocation"]
    assert "explain" in doctor["commands"]


def test_cli_raven_has_non_animated_fallback(capsys) -> None:
    assert main(["raven", "--style", "dance"]) == 0
    output = capsys.readouterr().out
    assert "VRAVEN" in output
    assert "Ready to inspect" not in output


def test_cli_explain_parameter_contract() -> None:
    args = build_parser().parse_args(
        [
            "explain",
            "weights.pt",
            "--model-factory",
            "models:build",
            "--factory-kwargs",
            '{"width": 16}',
            "--state-dict-key",
            "state_dict",
            "--strip-prefix",
            "module.",
            "--device",
            "cpu",
            "--input",
            "batch.npz",
            "--target",
            "2",
            "--mode",
            "causal",
            "--module-policy",
            "semantic",
            "--include-module-types",
            "Linear,Conv2d",
            "--exclude-name-patterns",
            "dropout*,head.debug",
            "--max-layers",
            "40",
            "--max-units",
            "24",
            "--representation-size",
            "32",
            "--max-attention-tokens",
            "64",
            "--histogram-bins",
            "20",
            "--no-entropy",
            "--no-attention",
            "--no-timing",
            "--causal-top-k",
            "7",
            "--causal-strategy",
            "scale",
            "--causal-scale",
            "0.25",
            "--seed",
            "41",
            "--output-path",
            "prediction.logits",
            "--report-format",
            "pdf",
            "--output",
            "result.pdf",
        ]
    )
    assert args.mode == "causal"
    assert args.module_policy == "semantic"
    assert args.causal_top_k == 7
    assert args.causal_strategy == "scale"
    assert args.causal_scale == 0.25
    assert args.report_format == "pdf"
    assert args.output == "result.pdf"


def test_cli_analysis_commands_execute_real_algorithms(tmp_path) -> None:
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.npy"
    labels_path = tmp_path / "labels.npy"
    torch.save(_demo_model().state_dict(), model_path)
    np.save(input_path, np.zeros((2, 12), dtype=np.float32))
    np.save(labels_path, np.array([1, 1], dtype=np.int64))
    common = [
        str(model_path),
        "--model-factory",
        "vraven.cli:_demo_model",
        "--input",
        str(input_path),
    ]

    adversarial_path = tmp_path / "adversarial.json"
    assert main(
        [
            "adversarial",
            *common,
            "--attacks",
            "fgsm",
            "--epsilon-grid",
            "0.01",
            "--steps",
            "1",
            "--output",
            str(adversarial_path),
        ]
    ) == 0
    assert json.loads(adversarial_path.read_text())["attacks_tested"][0]["attack_family"] == "fgsm"

    batch_path = tmp_path / "robust_accuracy.json"
    assert main(
        [
            "adversarial",
            *common,
            "--scope",
            "batch",
            "--protocol",
            "pgd",
            "--epsilon",
            "0.01",
            "--max-samples",
            "2",
            "--steps",
            "1",
            "--output",
            str(batch_path),
        ]
    ) == 0
    batch = json.loads(batch_path.read_text())
    assert batch["protocol"] == "pgd_linf"
    assert batch["n_samples"] == 2

    curve_path = tmp_path / "robustness_curve.json"
    assert main(
        [
            "adversarial",
            *common,
            "--scope",
            "curve",
            "--labels",
            str(labels_path),
            "--protocol",
            "pgd",
            "--epsilon-grid",
            "0,0.01",
            "--max-samples",
            "2",
            "--steps",
            "1",
            "--output",
            str(curve_path),
        ]
    ) == 0
    curve = json.loads(curve_path.read_text())
    assert curve["protocol"] == "pgd_linf"
    assert len(curve["points"]) == 2

    counterfactual_path = tmp_path / "counterfactual.json"
    assert main(
        [
            "counterfactual",
            *common,
            "--target-class",
            "0",
            "--steps",
            "2",
            "--output",
            str(counterfactual_path),
        ]
    ) == 0
    assert json.loads(counterfactual_path.read_text())["steps_completed"] == 2

    stability_path = tmp_path / "stability.json"
    assert main(
        [
            "stability",
            *common,
            "--target",
            "1",
            "--levels",
            "0,0.1",
            "--kinds",
            "noise",
            "--repeats",
            "1",
            "--output",
            str(stability_path),
        ]
    ) == 0
    assert json.loads(stability_path.read_text())["curves"][0]["kind"] == "noise"

    lrp_path = tmp_path / "lrp.json"
    assert main(["lrp", *common, "--target", "1", "--output", str(lrp_path)]) == 0
    assert json.loads(lrp_path.read_text())["layers"]

    certificate_path = tmp_path / "certificate.json"
    assert main(
        [
            "certify",
            *common,
            "--target",
            "1",
            "--epsilon-grid",
            "0,0.01",
            "--radius-search-steps",
            "1",
            "--output",
            str(certificate_path),
        ]
    ) == 0
    certificate = json.loads(certificate_path.read_text())
    assert certificate["property"] == "local classification invariance"
    assert certificate["sample_points"]


def test_cli_source_backed_cohort_and_model_commands(tmp_path) -> None:
    model_a = tmp_path / "model_a.pt"
    model_b = tmp_path / "model_b.pt"
    input_path = tmp_path / "input.npy"
    torch.save(_demo_model().state_dict(), model_a)
    torch.save(_demo_model().state_dict(), model_b)
    np.save(input_path, np.zeros((2, 12), dtype=np.float32))

    loader = ["--model-factory", "vraven.cli:_demo_model"]

    dataset_path = tmp_path / "dataset.json"
    assert main(
        [
            "dataset",
            str(model_a),
            *loader,
            "--input",
            str(input_path),
            "--max-samples",
            "2",
            "--output",
            str(dataset_path),
        ]
    ) == 0
    dataset = json.loads(dataset_path.read_text())
    assert dataset["schema_version"] == "1.0"
    assert len(dataset["traces"]) == 2

    comparison_path = tmp_path / "comparison.json"
    assert main(
        [
            "compare",
            str(model_a),
            str(model_b),
            *loader,
            "--input",
            str(input_path),
            "--max-samples",
            "2",
            "--output",
            str(comparison_path),
        ]
    ) == 0
    comparison = json.loads(comparison_path.read_text())
    assert comparison["sample_count"] == 2
    assert comparison["prediction_agreement_rate"] == 1.0

    evolution_path = tmp_path / "evolution.json"
    assert main(
        [
            "evolution",
            str(model_a),
            str(model_b),
            *loader,
            "--checkpoint-names",
            "initial,final",
            "--input",
            str(input_path),
            "--output",
            str(evolution_path),
        ]
    ) == 0
    evolution = json.loads(evolution_path.read_text())
    assert evolution["checkpoint_count"] == 2
    assert evolution["prediction_changes"] == 0

    anatomy_path = tmp_path / "anatomy.json"
    assert main(
        [
            "anatomy",
            str(model_a),
            *loader,
            "--input",
            str(input_path),
            "--target",
            "1",
            "--top-units",
            "2",
            "--output",
            str(anatomy_path),
        ]
    ) == 0
    anatomy = json.loads(anatomy_path.read_text())
    assert anatomy["analysis"] == "internal_anatomy"
    assert anatomy["metrics"]["layers"] == 3


def test_cli_beginner_analyse_workflow_and_alias(tmp_path) -> None:
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.npy"
    explanation_path = tmp_path / "easy_explanation.json"
    anatomy_path = tmp_path / "easy_anatomy.json"
    torch.save(_demo_model().state_dict(), model_path)
    np.save(input_path, np.zeros((2, 12), dtype=np.float32))
    common = [
        str(model_path),
        "--model-factory",
        "vraven.cli:_demo_model",
        "--input",
        str(input_path),
    ]

    assert main(
        [
            "quick",
            *common,
            "--preset",
            "quick",
            "--format",
            "json",
            "--output",
            str(explanation_path),
        ]
    ) == 0
    assert json.loads(explanation_path.read_text())["mode"] == "fast"

    assert main(
        [
            "analyze",
            *common,
            "--task",
            "anatomy",
            "--preset",
            "quick",
            "--output",
            str(anatomy_path),
        ]
    ) == 0
    assert json.loads(anatomy_path.read_text())["analysis"] == "internal_anatomy"
