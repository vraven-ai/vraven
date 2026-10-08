"""VRAVEN command-line interface.

The legacy trace demo remains available.  The ``capture`` and ``reverse`` commands expose
the compiler-assisted causal decompiler.  Model deserialisation is explicit and safe by
default: a state dictionary requires an importable architecture factory, while Python
pickle loading requires the user to opt in.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import vraven
from vraven.adapters import load_pytorch_model
from vraven.attribution import explain_lrp
from vraven.bundle import ModelBundle, load_model_bundle
from vraven.causal import generate_counterfactual, run_perturbation_stability
from vraven.darkside import inspect_internal_anatomy
from vraven.decompiler import (
    AdversarialAuditor,
    CaptureConfig,
    DecompileConfig,
    IntervalBoundVerifier,
    ModelCaptureRouter,
    VRavenDecompiler,
)
from vraven.report.export import (
    REPORT_FORMATS,
    find_chromium_executable,
    resolve_report_destination,
)
from vraven.report.report import PLOT_ALIASES, VRavenReport
from vraven.terminal import TerminalUI, emit_result
from vraven.visualisation.export import PUBLICATION_PRESETS


def _demo_model() -> torch.nn.Module:
    torch.manual_seed(17)
    return torch.nn.Sequential(
        torch.nn.Linear(12, 24),
        torch.nn.GELU(),
        torch.nn.Linear(24, 12),
        torch.nn.ReLU(),
        torch.nn.Linear(12, 3),
    )


def command_demo(args: argparse.Namespace) -> int:
    torch.manual_seed(17)
    model = _demo_model()
    inputs = torch.randn(args.samples, 12)
    report = vraven.explain(
        model,
        inputs,
        target=args.target,
        mode=args.mode,
        class_names=["class_0", "class_1", "class_2"],
    )
    destination = _report_destination(args, "vraven_demo_report")
    written = report.write_report(
        destination,
        format=args.report_format,
        mode="interactive_dark",
    )
    print(json.dumps(report.summary(), indent=2))
    print(f"Wrote VRAVEN report to {written.resolve()}")
    return 0


def command_explain(args: argparse.Namespace) -> int:
    """Explain one saved PyTorch model invocation and write one report."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    inputs: Any
    if model_kwargs:
        inputs = model_kwargs
    elif len(model_args) == 1:
        inputs = model_args[0]
    else:
        inputs = model_args
    class_names = _comma_separated(args.class_names) or None
    report = vraven.explain(
        model,
        inputs,
        target=args.target,
        mode=args.mode,
        class_names=class_names,
        device=args.device,
        auto_move_inputs=not args.no_auto_move_inputs,
        module_policy=args.module_policy,
        include_module_types=_comma_separated(args.include_module_types),
        exclude_module_types=(
            _comma_separated(args.exclude_module_types)
            if args.exclude_module_types is not None
            else ("Dropout", "Identity")
        ),
        include_name_patterns=_comma_separated(args.include_name_patterns),
        exclude_name_patterns=_comma_separated(args.exclude_name_patterns),
        max_layers=args.max_layers,
        max_units=args.max_units,
        representation_size=args.representation_size,
        max_attention_tokens=args.max_attention_tokens,
        histogram_bins=args.histogram_bins,
        capture_entropy=not args.no_entropy,
        capture_representations=not args.no_representations,
        capture_unit_profiles=not args.no_unit_profiles,
        capture_attention=not args.no_attention,
        capture_input_attribution=not args.no_input_attribution,
        capture_full_tensors=args.capture_full_tensors,
        capture_timing=not args.no_timing,
        causal_top_k=args.causal_top_k,
        causal_strategy=args.causal_strategy,
        causal_scale=args.causal_scale,
        random_seed=args.seed,
        strict=not args.non_strict_capture,
        output_path=args.output_path,
    )
    destination = _report_destination(args, "vraven_explanation")
    written = report.write_report(destination, format=args.report_format)
    print(json.dumps(report.summary(), indent=2))
    print(f"Wrote VRAVEN explanation report to {written.resolve()}")
    return 0


def command_adversarial(args: argparse.Namespace) -> int:
    """Run a finite-budget empirical adversarial stress audit."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    inputs = _single_tensor_input(model_args, model_kwargs, command="adversarial")
    labels = _load_labels(args.labels) if args.labels else None
    auditor = AdversarialAuditor(
        model,
        model_name=args.model_name,
        output_path=args.output_path,
    )
    if args.scope == "sample":
        sample = _select_sample(inputs, args.sample_index)
        if labels is not None and args.sample_index >= len(labels):
            raise IndexError("sample index is out of range for the supplied labels")
        label = int(labels[args.sample_index]) if labels is not None else None
        report = auditor.audit_sample(
            sample,
            y=label,
            target=args.target,
            epsilon_grid=_comma_separated_floats(args.epsilon_grid),
            pgd_steps=args.steps,
            pgd_restarts=args.restarts,
            pgd_step_size=args.step_size,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            seed=args.seed,
            attacks=_attack_selection(args.attacks),
            deepfool_iters=args.deepfool_iters,
            square_queries=args.square_queries,
            fab_iters=args.fab_iters,
            fab_restarts=args.fab_restarts,
            n_target_classes=args.target_classes,
        )
        default_stem = "vraven_adversarial_audit"
    elif args.scope == "batch":
        if args.format != "json":
            raise ValueError("batch adversarial output currently supports --format json only")
        report = auditor.audit_batch(
            inputs,
            labels,
            epsilon=args.epsilon,
            protocol=args.protocol,
            max_samples=args.max_samples,
            sampling=args.sampling,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            seed=args.seed,
            steps=args.steps,
            pgd_restarts=args.restarts,
            square_queries=args.square_queries,
            fab_iters=args.fab_iters,
            fab_restarts=args.fab_restarts,
            n_target_classes=args.target_classes,
        )
        default_stem = "vraven_robust_accuracy"
    else:
        if labels is None:
            raise ValueError("--scope curve requires --labels")
        report = auditor.audit_robustness_curve(
            inputs,
            labels,
            epsilon_grid=_comma_separated_floats(args.epsilon_grid),
            protocol=args.protocol,
            max_samples=args.max_samples,
            sampling=args.sampling,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            seed=args.seed,
            steps=args.steps,
            pgd_restarts=args.restarts,
            square_queries=args.square_queries,
            fab_iters=args.fab_iters,
            fab_restarts=args.fab_restarts,
            n_target_classes=args.target_classes,
        )
        default_stem = "vraven_robustness_curve"
    destination, _ = resolve_report_destination(
        args.output or f"{default_stem}.{args.format}",
        args.format,
    )
    if args.format == "html":
        written = report.write_html(destination)
    elif hasattr(report, "save"):
        written = report.save(destination)
    else:
        written = _write_json_result(destination, report.to_dict())
    payload = report.to_dict()
    if args.scope == "sample":
        rows = {
            "Scope": "single sample",
            "Attacks executed": len(payload["attacks_tested"]),
            "Smallest L∞ flip": (
                payload["empirical_min_flip_eps_linf"]
                if payload["empirical_min_flip_eps_linf"] is not None
                else "not found"
            ),
        }
    elif args.scope == "batch":
        rows = {
            "Scope": "fixed-epsilon batch",
            "Samples": payload["n_samples"],
            "Clean accuracy": _terminal_percent(payload.get("clean_accuracy")),
            "Robust accuracy": _terminal_percent(payload.get("robust_accuracy")),
        }
    else:
        rows = {
            "Scope": "robustness curve",
            "Epsilon points": len(payload["points"]),
            "Robustness AUC": f"{float(payload['robust_accuracy_auc']):.4f}",
            "Endpoint accuracy": _terminal_percent(payload.get("endpoint_robust_accuracy")),
        }
    _emit_saved_result(payload, "ADVERSARIAL AUDIT", rows, written)
    return 0


def command_counterfactual(args: argparse.Namespace) -> int:
    """Search for one bounded continuous-input counterfactual."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    inputs = _single_tensor_input(model_args, model_kwargs, command="counterfactual")
    sample = _select_sample(inputs, args.sample_index)
    if (args.clamp_min is None) != (args.clamp_max is None):
        raise ValueError("--clamp-min and --clamp-max must be supplied together")
    clamp = (
        (float(args.clamp_min), float(args.clamp_max))
        if args.clamp_min is not None and args.clamp_max is not None
        else None
    )
    result = generate_counterfactual(
        model,
        sample,
        target_class=args.target_class,
        steps=args.steps,
        learning_rate=args.learning_rate,
        l2_weight=args.l2_weight,
        max_linf=args.max_linf,
        clamp=clamp,
        top_changes=args.top_changes,
    )
    destination = _write_json_result(args.output, asdict(result))
    payload = asdict(result)
    _emit_saved_result(
        payload,
        "COUNTERFACTUAL",
        {
            "Succeeded": payload["success"],
            "Steps": payload["steps_completed"],
            "Top-change L2": f"{float(np.linalg.norm(payload['delta_values'])):.4g}",
            "Top-change L∞": f"{max((abs(v) for v in payload['delta_values']), default=0):.4g}",
        },
        destination,
    )
    return 0


def command_stability(args: argparse.Namespace) -> int:
    """Measure prediction-score stability under controlled perturbations."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    inputs = _public_inputs(model_args, model_kwargs)
    result = run_perturbation_stability(
        model,
        inputs,
        target=args.target,
        levels=list(_comma_separated_floats(args.levels)),
        kinds=_comma_separated(args.kinds),
        repeats=args.repeats,
        random_seed=args.seed,
        auto_move_inputs=not args.no_auto_move_inputs,
    )
    payload = asdict(result)
    destination = _write_json_result(args.output, payload)
    _emit_saved_result(
        payload,
        "PERTURBATION STABILITY",
        {
            "Target": args.target if args.target is not None else "predicted",
            "Perturbation families": len(payload["curves"]),
            "Repeats": args.repeats,
            "Levels": len(_comma_separated_floats(args.levels)),
        },
        destination,
    )
    return 0


def command_lrp(args: argparse.Namespace) -> int:
    """Run epsilon-rule Layer-wise Relevance Propagation."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    result = explain_lrp(
        model,
        _public_inputs(model_args, model_kwargs),
        target=args.target,
        epsilon=args.epsilon,
        auto_move_inputs=not args.no_auto_move_inputs,
    )
    payload = asdict(result)
    destination = _write_json_result(args.output, payload)
    _emit_saved_result(
        payload,
        "LAYER-WISE RELEVANCE",
        {
            "Target": args.target if args.target is not None else "predicted",
            "Layers": len(payload["layers"]),
            "Conservation ratio": f"{float(payload['conservation_ratio']):.4f}",
            "Rule": "epsilon",
        },
        destination,
    )
    return 0


def command_certify(args: argparse.Namespace) -> int:
    """Run the supported-operator formal L-infinity verifier."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    inputs = _single_tensor_input(model_args, model_kwargs, command="certify")
    sample = _select_sample(inputs, args.sample_index)
    certificate = IntervalBoundVerifier(model, model_name=args.model_name).certify(
        sample,
        target=args.target,
        epsilon_grid=_comma_separated_floats(args.epsilon_grid),
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        radius_search_steps=args.radius_search_steps,
    )
    destination, _ = resolve_report_destination(
        args.output or f"vraven_formal_certificate.{args.format}",
        args.format,
    )
    written = (
        certificate.write_html(destination)
        if args.format == "html"
        else certificate.save(destination)
    )
    payload = certificate.to_dict()
    _emit_saved_result(
        payload,
        "FORMAL ROBUSTNESS",
        {
            "Status": payload["sample_points"][-1]["status"],
            "Property": payload["property"],
            "Certified radius": payload.get("certified_radius_lower_bound", "unavailable"),
            "Target": payload.get("target", args.target if args.target is not None else "predicted"),
        },
        written,
    )
    return 0


def command_raven(args: argparse.Namespace) -> int:
    """Display the optional terminal raven mascot."""

    TerminalUI().mascot(style=args.style, cycles=args.cycles, delay=args.delay)
    return 0


def command_analyse(args: argparse.Namespace) -> int:
    """Route a small, beginner-facing option set to the full analysis commands."""

    bundle = load_model_bundle(args.model) if Path(args.model).is_dir() else None
    _apply_bundle_to_analyse_args(args, bundle)
    if args.input is None:
        raise ValueError("--input is required when the model is not a VRAVEN bundle directory")
    task = args.task or "explain"
    preset = args.preset or "quick"
    defaults = {
        "explain": f"vraven_explanation.{args.format}",
        "adversarial": f"vraven_adversarial_audit.{args.format}",
        "counterfactual": "vraven_counterfactual.json",
        "stability": "vraven_stability.json",
        "lrp": "vraven_lrp.json",
        "certify": f"vraven_formal_certificate.{args.format}",
        "dataset": "vraven_dataset_atlas.json",
        "anatomy": "vraven_internal_anatomy.json",
    }
    output = args.output or defaults[task]
    command = [task, args.model, *_forward_model_loading_arguments(args), "--input", args.input]
    if args.target is not None and task in {
        "explain",
        "adversarial",
        "stability",
        "lrp",
        "certify",
        "anatomy",
    }:
        command.extend(["--target", str(args.target)])
    if args.labels and task in {"adversarial", "dataset"}:
        command.extend(["--labels", args.labels])
    if args.class_names and task in {"explain", "dataset"}:
        command.extend(["--class-names", args.class_names])

    if task == "explain":
        mode = {"quick": "fast", "standard": "gradient", "thorough": "causal"}[preset]
        command.extend(["--mode", mode, "--report-format", args.format, "--output", output])
    elif task == "adversarial":
        attack_options = {
            "quick": ("quick", "5", "0.01,0.05"),
            "standard": ("fgsm,bim,pgd_linf,mi_fgsm", "10", "0.01,0.05,0.1"),
            "thorough": ("full", "25", "0.005,0.01,0.05,0.1"),
        }
        attacks, steps, epsilon_grid = attack_options[preset]
        command.extend(
            [
                "--attacks",
                attacks,
                "--steps",
                steps,
                "--epsilon-grid",
                epsilon_grid,
                "--format",
                args.format,
                "--output",
                output,
            ]
        )
    elif task == "counterfactual":
        if args.target_class is None:
            raise ValueError("counterfactual analysis requires --target-class")
        steps = {"quick": 25, "standard": 100, "thorough": 250}[preset]
        command.extend(
            ["--target-class", str(args.target_class), "--steps", str(steps), "--output", output]
        )
    elif task == "stability":
        repeats = {"quick": 2, "standard": 5, "thorough": 10}[preset]
        levels = {"quick": "0,0.1", "standard": "0,0.1,0.25,0.5", "thorough": "0,0.1,0.2,0.4,0.6,0.8,1"}[preset]
        command.extend(["--repeats", str(repeats), "--levels", levels, "--output", output])
    elif task == "lrp":
        command.extend(["--output", output])
    elif task == "certify":
        radius_steps = {"quick": 8, "standard": 24, "thorough": 48}[preset]
        command.extend(
            [
                "--radius-search-steps",
                str(radius_steps),
                "--format",
                args.format,
                "--output",
                output,
            ]
        )
    elif task == "dataset":
        max_samples = {"quick": 16, "standard": 64, "thorough": 128}[preset]
        mode = {"quick": "fast", "standard": "gradient", "thorough": "gradient"}[preset]
        command.extend(
            ["--max-samples", str(max_samples), "--mode", mode, "--output", output]
        )
    else:
        top_units = {"quick": 5, "standard": 12, "thorough": 24}[preset]
        command.extend(["--top-units", str(top_units), "--output", output])

    parsed = build_parser().parse_args(command)
    bundle_import_root = str(bundle.root) if bundle is not None else None
    inserted_bundle_path = bool(bundle_import_root and bundle_import_root not in sys.path)
    if inserted_bundle_path and bundle_import_root is not None:
        sys.path.insert(0, bundle_import_root)
    try:
        return int(parsed.handler(parsed))
    finally:
        if inserted_bundle_path and bundle_import_root is not None:
            sys.path.remove(bundle_import_root)


def command_bundle(args: argparse.Namespace) -> int:
    """Validate and describe a self-contained VRAVEN model bundle."""

    bundle = load_model_bundle(args.directory)
    payload = bundle.summary()
    emit_result(
        payload,
        title="MODEL BUNDLE",
        rows={
            "Model": bundle.model.name,
            "Input": bundle.input.name,
            "Labels": bundle.labels.name if bundle.labels else "not supplied",
            "Factory": bundle.factory or "not supplied",
            "Default task": bundle.defaults.get("task", "explain"),
        },
        footer=f"Valid bundle: {bundle.root}",
    )
    return 0


def command_bundle_init(args: argparse.Namespace) -> int:
    """Create and validate a public VRAVEN model-bundle manifest."""

    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Bundle directory does not exist: {root}")
    manifest = root / "vraven.json"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest}")
    factory_kwargs = json.loads(args.factory_kwargs)
    if not isinstance(factory_kwargs, dict):
        raise ValueError("--factory-kwargs must decode to a JSON object")
    if args.model_format == "state_dict" and not args.model_factory:
        raise ValueError("state_dict bundles require --factory")
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "model": {
            "path": args.model,
            "format": args.model_format,
            "factory": args.model_factory,
            "factory_kwargs": factory_kwargs,
            "device": args.device,
        },
        "input": {"path": args.input},
        "defaults": {
            "task": args.task,
            "preset": args.preset,
            "format": args.format,
        },
    }
    if args.labels:
        payload["input"]["labels"] = args.labels
    if args.class_names:
        payload["class_names"] = list(_comma_separated(args.class_names))
    if args.target is not None:
        payload["defaults"]["target"] = args.target
    if args.target_class is not None:
        payload["defaults"]["target_class"] = args.target_class
    if args.output:
        payload["defaults"]["output"] = args.output
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        bundle = load_model_bundle(root)
    except Exception:
        manifest.unlink(missing_ok=True)
        raise
    emit_result(
        bundle.summary(),
        title="BUNDLE CREATED",
        rows={
            "Manifest": manifest.name,
            "Model": bundle.model.name,
            "Input": bundle.input.name,
            "Default task": args.task,
        },
        footer=f"Next: vraven quick {root}",
    )
    return 0


def command_dataset(args: argparse.Namespace) -> int:
    """Explain samples independently and compare their reasoning fingerprints."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    labels = _load_labels(args.labels) if args.labels else None
    report = vraven.explain_dataset(
        model,
        _public_inputs(model_args, model_kwargs),
        targets=labels,
        mode=args.mode,
        class_names=_comma_separated(args.class_names) or None,
        max_samples=args.max_samples,
        fingerprint_dimensions=args.fingerprint_dimensions,
        max_layers=args.max_layers,
        max_units=args.max_units,
    )
    destination = report.save(args.output)
    summary = report.summary()
    _emit_saved_result(
        report.to_dict(),
        "DATASET ATLAS",
        {
            "Samples": summary["sample_count"],
            "Mode": summary["mode"],
            "Mean confidence": _terminal_percent(summary["mean_confidence"]),
            "Most unusual sample": summary["most_unusual_sample"],
        },
        destination,
    )
    return 0


def command_compare(args: argparse.Namespace) -> int:
    """Compare two saved models on the same bounded sample cohort."""

    model_a = _load_model_path(
        args,
        args.model_a,
        factory_path=args.model_a_factory or args.model_factory,
        factory_kwargs_json=args.model_a_factory_kwargs or args.factory_kwargs,
    )
    model_b = _load_model_path(
        args,
        args.model_b,
        factory_path=args.model_b_factory or args.model_factory,
        factory_kwargs_json=args.model_b_factory_kwargs or args.factory_kwargs,
    )
    model_args, model_kwargs = _load_inputs(args.input)
    labels = _load_labels(args.labels) if args.labels else None
    comparison = vraven.compare_models(
        model_a,
        model_b,
        _public_inputs(model_args, model_kwargs),
        targets=labels,
        mode=args.mode,
        class_names=_comma_separated(args.class_names) or None,
        max_samples=args.max_samples,
        fingerprint_dimensions=args.fingerprint_dimensions,
        left_name=args.left_name,
        right_name=args.right_name,
        max_layers=args.max_layers,
        max_units=args.max_units,
    )
    payload = comparison.summary()
    destination = _write_json_result(args.output, payload)
    _emit_saved_result(
        payload,
        "MODEL COMPARISON",
        {
            "Models": f"{args.left_name} ↔ {args.right_name}",
            "Samples": payload["sample_count"],
            "Prediction agreement": _terminal_percent(payload["prediction_agreement_rate"]),
            "Mean fingerprint similarity": f"{payload['mean_fingerprint_similarity']:.4f}",
        },
        destination,
    )
    return 0


def command_evolution(args: argparse.Namespace) -> int:
    """Track one input across an ordered sequence of model checkpoints."""

    models = [_load_model_path(args, path) for path in args.models]
    model_args, model_kwargs = _load_inputs(args.input)
    names = _comma_separated(args.checkpoint_names) or tuple(
        Path(path).stem for path in args.models
    )
    result = vraven.compare_checkpoints(
        models,
        _public_inputs(model_args, model_kwargs),
        target=args.target,
        checkpoint_names=names,
        mode=args.mode,
        class_names=_comma_separated(args.class_names) or None,
        fingerprint_dimensions=args.fingerprint_dimensions,
        max_layers=args.max_layers,
        max_units=args.max_units,
    )
    payload = result.summary()
    destination = _write_json_result(args.output, payload)
    _emit_saved_result(
        payload,
        "CHECKPOINT EVOLUTION",
        {
            "Checkpoints": payload["checkpoint_count"],
            "Prediction changes": payload["prediction_changes"],
            "First → final similarity": (
                f"{payload['initial_to_final_fingerprint_similarity']:.4f}"
            ),
            "Final checkpoint": payload["final_checkpoint"],
        },
        destination,
    )
    return 0


def command_anatomy(args: argparse.Namespace) -> int:
    """Inspect measured local anatomy of Linear layers for one sample."""

    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    inputs = _single_tensor_input(model_args, model_kwargs, command="anatomy")
    sample = _select_sample(inputs, args.sample_index)
    result = inspect_internal_anatomy(
        model,
        sample,
        target=args.target,
        layer_names=_comma_separated(args.layers) or None,
        top_units=args.top_units,
    )
    destination = result.save_json(args.output)
    payload = result.to_dict()
    _emit_saved_result(
        payload,
        "INTERNAL ANATOMY",
        {
            "Linear layers": payload["metrics"]["layers"],
            "Target score": f"{payload['metrics']['target_score']:.4f}",
            "Top units per layer": payload["metrics"]["top_units"],
            "Claim type": payload["metadata"]["claim_type"],
        },
        destination,
    )
    return 0


def command_decompile_demo(args: argparse.Namespace) -> int:
    torch.manual_seed(41)
    features = torch.randn(args.samples, 10)
    labels = ((features[:, 0] + 0.8 * features[:, 1] - 0.55 * features[:, 4]) > 0).long()
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 18),
        torch.nn.GELU(),
        torch.nn.Linear(18, 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 2),
    ).eval()
    config = _decompile_profile(args.profile)
    result = VRavenDecompiler(config).reverse(
        model,
        (features,),
        labels=labels,
        focus_args=(features[args.focus_index : args.focus_index + 1],),
        class_names=["class_0", "class_1"],
    )
    destination = _report_destination(args, "vraven_decision_proof")
    written = result.write_report(destination, format=args.report_format)
    print(json.dumps(result.summary(), indent=2))
    print(f"Wrote VRAVEN Decision Proof to {written.resolve()}")
    return 0


def command_capture(args: argparse.Namespace) -> int:
    model = _load_cli_model(args)
    model_args, model_kwargs = _load_inputs(args.input)
    captured = ModelCaptureRouter(CaptureConfig()).capture(model, model_args, model_kwargs)
    destination = Path(args.output)
    captured.ir.save(destination)
    print(json.dumps(captured.summary(), indent=2))
    print(f"Wrote VRAVEN Mechanism IR to {destination.resolve()}")
    return 0


def command_reverse(args: argparse.Namespace) -> int:
    model = _load_cli_model(args)
    reference_args, reference_kwargs = _load_inputs(args.reference)
    labels = _load_labels(args.labels) if args.labels else None
    focus_args, focus_kwargs = _slice_loaded_inputs(
        reference_args,
        reference_kwargs,
        args.focus_index,
    )
    class_names = args.class_names.split(",") if args.class_names else None
    input_groups = _load_json_mapping(args.input_groups) if args.input_groups else None
    validated_input_groups = (
        tuple(item.strip() for item in args.validated_input_groups.split(",") if item.strip())
        if args.validated_input_groups
        else None
    )
    config = _decompile_profile(args.profile)
    config.run_counterfactual = not args.no_counterfactual
    config.run_profiler = not args.no_profiler
    if args.output_path:
        config.causal.output_path = args.output_path
        config.counterfactual.output_path = args.output_path
    result = VRavenDecompiler(config).reverse(
        model,
        reference_args,
        reference_kwargs=reference_kwargs,
        labels=labels,
        focus_args=focus_args,
        focus_kwargs=focus_kwargs,
        target=args.target,
        class_names=class_names,
        input_groups=input_groups,
        validated_input_groups=validated_input_groups,
    )
    destination = _report_destination(args, "vraven_decision_proof")
    written = result.write_report(destination, format=args.report_format)
    print(json.dumps(result.summary(), indent=2))
    print(f"Wrote VRAVEN Decision Proof to {written.resolve()}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    report = VRavenReport.load_trace(args.trace)
    payload = {"summary": report.summary(), "capabilities": report.capabilities()}
    print(json.dumps(payload, indent=2))
    return 0


def command_render(args: argparse.Namespace) -> int:
    report = VRavenReport.load_trace(args.trace)
    report.export(args.visual, args.output, mode=args.mode, preset=args.preset, dpi=args.dpi)
    print(f"Wrote {Path(args.output).resolve()}")
    return 0


def command_doctor(_args: argparse.Namespace) -> int:
    browser = find_chromium_executable()
    invocation = Path(sys.argv[0]).resolve() if Path(sys.argv[0]).exists() else sys.argv[0]
    command_on_path = shutil.which("vraven")
    payload = {
        "vraven": vraven.__version__,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "vraven_package": str(Path(vraven.__file__).resolve()),
        "vraven_invocation": str(invocation),
        "vraven_command_on_path": command_on_path,
        "vraven_command": command_on_path,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "torch_export": bool(hasattr(torch, "export") and hasattr(torch.export, "export")),
        "torch_func": bool(hasattr(torch, "func")),
        "pdf_browser": str(browser) if browser else None,
        "pdf_report_available": browser is not None,
        "report_formats": list(REPORT_FORMATS),
        "commands": _feature_payload()["commands"],
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_features(_args: argparse.Namespace) -> int:
    """Print a stable, machine-readable inventory of public CLI capabilities."""

    print(json.dumps(_feature_payload(), indent=2))
    return 0


def _feature_payload() -> dict[str, object]:
    return {
        "commands": [
            "analyse",
            "bundle",
            "bundle-init",
            "explain",
            "adversarial",
            "counterfactual",
            "stability",
            "lrp",
            "certify",
            "dataset",
            "compare",
            "evolution",
            "anatomy",
            "reverse",
            "capture",
            "demo",
            "decompile-demo",
            "inspect",
            "render",
            "raven",
            "features",
            "doctor",
        ],
        "explanation_modes": ["fast", "gradient", "causal"],
        "report_formats": list(REPORT_FORMATS),
        "render_modes": ["publication_light", "interactive_dark"],
        "figure_formats": ["html", "json", "svg", "pdf", "png", "jpg", "jpeg", "webp"],
        "publication_presets": sorted(PUBLICATION_PRESETS),
        "visuals": sorted(set(PLOT_ALIASES) | set(PLOT_ALIASES.values())),
        "model_formats": ["state_dict", "pickle"],
        "loader_only_formats": ["torchscript"],
        "analysis_outputs": {
            "adversarial": ["json", "html"],
            "counterfactual": ["json"],
            "stability": ["json"],
            "lrp": ["json"],
            "certify": ["json", "html"],
            "dataset": ["json"],
            "compare": ["json"],
            "evolution": ["json"],
            "anatomy": ["json"],
        },
        "terminal_ui": {
            "tty_result_panels": True,
            "machine_readable_when_piped": True,
            "raven_styles": ["still", "blink", "dance"],
        },
        "beginner_workflow": {
            "command": "analyse",
            "aliases": ["analyze", "quick"],
            "presets": ["quick", "standard", "thorough"],
            "tasks": [
                "explain",
                "adversarial",
                "counterfactual",
                "stability",
                "lrp",
                "certify",
                "dataset",
                "anatomy",
            ],
            "model_bundle_manifest": "vraven.json",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vraven",
        description="Evidence-aware explainability and causal model auditing for PyTorch",
    )
    parser.add_argument("--version", action="version", version=vraven.__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser(
        "analyse",
        aliases=("analyze", "quick"),
        help="Run a common analysis with beginner-friendly defaults",
    )
    analyse.add_argument("model", help="PyTorch model file")
    _add_model_loading_arguments(analyse, nullable=True)
    analyse.add_argument(
        "-i", "--input", help="Input tensor file; optional for a VRAVEN bundle directory"
    )
    analyse.add_argument(
        "-t",
        "--task",
        choices=(
            "explain",
            "adversarial",
            "counterfactual",
            "stability",
            "lrp",
            "certify",
            "dataset",
            "anatomy",
        ),
        default=None,
    )
    analyse.add_argument(
        "-p",
        "--preset",
        choices=("quick", "standard", "thorough"),
        default=None,
        help="Trade runtime for analysis depth",
    )
    analyse.add_argument("--target", type=int, help="Optional class target")
    analyse.add_argument("--target-class", type=int, help="Required for counterfactual search")
    analyse.add_argument("--labels", help="Optional label file for dataset/adversarial tasks")
    analyse.add_argument("--class-names", help="Optional comma-separated class names")
    analyse.add_argument("--format", choices=("json", "html"), default=None)
    analyse.add_argument(
        "-o", "--output", help="Output path; a task-specific name is used by default"
    )
    analyse.set_defaults(handler=command_analyse)

    bundle = subparsers.add_parser(
        "bundle",
        help="Validate and describe a self-contained VRAVEN model bundle",
    )
    bundle.add_argument("directory", help="Directory containing vraven.json")
    bundle.set_defaults(handler=command_bundle)

    bundle_init = subparsers.add_parser(
        "bundle-init",
        help="Create vraven.json for model and input files already in a directory",
    )
    bundle_init.add_argument("directory", help="Existing model-bundle directory")
    bundle_init.add_argument("--model", default="model.pt", help="Relative model filename")
    bundle_init.add_argument("--input", default="sample.npy", help="Relative input filename")
    bundle_init.add_argument("--labels", help="Optional relative label filename")
    bundle_init.add_argument(
        "--model-format",
        choices=("state_dict", "pickle"),
        default="state_dict",
    )
    bundle_init.add_argument("--model-factory", "--factory", dest="model_factory")
    bundle_init.add_argument("--factory-kwargs", default="{}")
    bundle_init.add_argument("--device", default="cpu")
    bundle_init.add_argument(
        "--task",
        choices=(
            "explain",
            "adversarial",
            "counterfactual",
            "stability",
            "lrp",
            "certify",
            "dataset",
            "anatomy",
        ),
        default="explain",
    )
    bundle_init.add_argument(
        "--preset", choices=("quick", "standard", "thorough"), default="quick"
    )
    bundle_init.add_argument("--format", choices=("json", "html"), default="html")
    bundle_init.add_argument("--output", help="Optional relative default output path")
    bundle_init.add_argument("--target", type=int, help="Optional default class target")
    bundle_init.add_argument(
        "--target-class", type=int, help="Default destination class for counterfactual search"
    )
    bundle_init.add_argument("--class-names", help="Optional comma-separated class names")
    bundle_init.set_defaults(handler=command_bundle_init)

    demo = subparsers.add_parser("demo", help="Run a deterministic local PyTorch demonstration")
    demo.add_argument("--mode", choices=("fast", "gradient", "causal"), default="gradient")
    demo.add_argument("--samples", type=int, default=48)
    demo.add_argument("--target", type=int, default=1)
    _add_report_arguments(demo, default_stem="vraven_demo_report")
    demo.set_defaults(handler=command_demo)

    explain_parser = subparsers.add_parser(
        "explain",
        help="Explain one saved PyTorch model invocation and write one report",
    )
    _add_model_arguments(explain_parser)
    explain_parser.add_argument(
        "--input",
        required=True,
        help="Model input as .npy, .npz or a trusted tensor-only .pt/.pth file",
    )
    explain_parser.add_argument("--target", type=int, help="Target class index")
    explain_parser.add_argument(
        "--mode", choices=("fast", "gradient", "causal"), default="gradient"
    )
    explain_parser.add_argument("--class-names", help="Comma-separated class labels")
    explain_parser.add_argument(
        "--module-policy", choices=("leaf", "semantic", "all"), default="leaf"
    )
    explain_parser.add_argument("--include-module-types", help="Comma-separated module types")
    explain_parser.add_argument("--exclude-module-types", help="Comma-separated module types")
    explain_parser.add_argument("--include-name-patterns", help="Comma-separated layer-name patterns")
    explain_parser.add_argument("--exclude-name-patterns", help="Comma-separated layer-name patterns")
    explain_parser.add_argument("--max-layers", type=int, default=256)
    explain_parser.add_argument("--max-units", type=int, default=128)
    explain_parser.add_argument("--representation-size", type=int, default=64)
    explain_parser.add_argument("--max-attention-tokens", type=int, default=128)
    explain_parser.add_argument("--histogram-bins", type=int, default=32)
    explain_parser.add_argument("--no-auto-move-inputs", action="store_true")
    explain_parser.add_argument("--no-entropy", action="store_true")
    explain_parser.add_argument("--no-representations", action="store_true")
    explain_parser.add_argument("--no-unit-profiles", action="store_true")
    explain_parser.add_argument("--no-attention", action="store_true")
    explain_parser.add_argument("--no-input-attribution", action="store_true")
    explain_parser.add_argument("--capture-full-tensors", action="store_true")
    explain_parser.add_argument("--no-timing", action="store_true")
    explain_parser.add_argument("--causal-top-k", type=int, default=3)
    explain_parser.add_argument(
        "--causal-strategy", choices=("zero", "mean", "scale"), default="zero"
    )
    explain_parser.add_argument("--causal-scale", type=float, default=0.0)
    explain_parser.add_argument("--seed", type=int, default=17)
    explain_parser.add_argument("--output-path", help="Dot path to logits in structured output")
    explain_parser.add_argument(
        "--non-strict-capture",
        action="store_true",
        help="Continue when an individual capture operation is unsupported",
    )
    _add_report_arguments(explain_parser, default_stem="vraven_explanation")
    explain_parser.set_defaults(handler=command_explain)

    adversarial = subparsers.add_parser(
        "adversarial",
        help="Run finite-budget empirical adversarial stress testing",
    )
    _add_model_arguments(adversarial)
    adversarial.add_argument("--input", required=True, help="Input tensor file")
    adversarial.add_argument("--labels", help="Optional labels for the input batch")
    adversarial.add_argument(
        "--scope",
        choices=("sample", "batch", "curve"),
        default="sample",
        help="Audit one sample, fixed-epsilon batch accuracy, or an epsilon robustness curve",
    )
    adversarial.add_argument("--sample-index", type=int, default=0)
    adversarial.add_argument("--target", type=int, help="Decision target; defaults to label or prediction")
    adversarial.add_argument("--model-name")
    adversarial.add_argument("--output-path", help="Dot path to logits in structured output")
    adversarial.add_argument(
        "--attacks",
        default="quick",
        help="quick, full, or comma-separated attacks such as fgsm,pgd_linf,deepfool",
    )
    adversarial.add_argument("--epsilon-grid", default="0.01,0.05,0.1")
    adversarial.add_argument("--epsilon", type=float, default=0.1)
    adversarial.add_argument(
        "--protocol",
        choices=("pgd", "pgd_linf", "vraven_ensemble_lite"),
        default="vraven_ensemble_lite",
    )
    adversarial.add_argument("--max-samples", type=int)
    adversarial.add_argument("--sampling", choices=("head", "stratified"), default="head")
    adversarial.add_argument("--steps", type=int, default=10)
    adversarial.add_argument("--restarts", type=int, default=1)
    adversarial.add_argument("--step-size", type=float)
    adversarial.add_argument("--clip-min", type=float)
    adversarial.add_argument("--clip-max", type=float)
    adversarial.add_argument("--seed", type=int, default=24032026)
    adversarial.add_argument("--deepfool-iters", type=int, default=40)
    adversarial.add_argument("--square-queries", type=int, default=80)
    adversarial.add_argument("--fab-iters", type=int, default=20)
    adversarial.add_argument("--fab-restarts", type=int, default=2)
    adversarial.add_argument("--target-classes", type=int, default=3)
    adversarial.add_argument("--format", choices=("json", "html"), default="json")
    adversarial.add_argument("--output", help="Output path; defaults to the selected format")
    adversarial.set_defaults(handler=command_adversarial)

    counterfactual = subparsers.add_parser(
        "counterfactual",
        help="Search for a bounded continuous-input counterfactual",
    )
    _add_model_arguments(counterfactual)
    counterfactual.add_argument("--input", required=True, help="Input tensor file")
    counterfactual.add_argument("--sample-index", type=int, default=0)
    counterfactual.add_argument("--target-class", type=int, required=True)
    counterfactual.add_argument("--steps", type=int, default=100)
    counterfactual.add_argument("--learning-rate", type=float, default=0.05)
    counterfactual.add_argument("--l2-weight", type=float, default=0.01)
    counterfactual.add_argument("--max-linf", type=float)
    counterfactual.add_argument("--clamp-min", type=float)
    counterfactual.add_argument("--clamp-max", type=float)
    counterfactual.add_argument("--top-changes", type=int, default=20)
    counterfactual.add_argument("--output", default="vraven_counterfactual.json")
    counterfactual.set_defaults(handler=command_counterfactual)

    stability = subparsers.add_parser(
        "stability",
        help="Measure score stability under controlled input perturbations",
    )
    _add_model_arguments(stability)
    stability.add_argument("--input", required=True, help="Input tensor file")
    stability.add_argument("--target", type=int)
    stability.add_argument("--levels", default="0,0.2,0.4,0.6,0.8,1")
    stability.add_argument("--kinds", default="noise,feature_mask,shuffle")
    stability.add_argument("--repeats", type=int, default=5)
    stability.add_argument("--seed", type=int, default=17)
    stability.add_argument("--no-auto-move-inputs", action="store_true")
    stability.add_argument("--output", default="vraven_stability.json")
    stability.set_defaults(handler=command_stability)

    lrp = subparsers.add_parser(
        "lrp",
        help="Run epsilon-rule Layer-wise Relevance Propagation",
    )
    _add_model_arguments(lrp)
    lrp.add_argument("--input", required=True, help="Input tensor file")
    lrp.add_argument("--target", type=int)
    lrp.add_argument("--epsilon", type=float, default=1e-6)
    lrp.add_argument("--no-auto-move-inputs", action="store_true")
    lrp.add_argument("--output", default="vraven_lrp.json")
    lrp.set_defaults(handler=command_lrp)

    certify = subparsers.add_parser(
        "certify",
        help="Run formal local L-infinity verification for supported operators",
    )
    _add_model_arguments(certify)
    certify.add_argument("--input", required=True, help="Input tensor file")
    certify.add_argument("--sample-index", type=int, default=0)
    certify.add_argument("--target", type=int)
    certify.add_argument("--model-name")
    certify.add_argument("--epsilon-grid", default="0,0.001,0.005,0.01,0.05,0.1")
    certify.add_argument("--clip-min", type=float)
    certify.add_argument("--clip-max", type=float)
    certify.add_argument("--radius-search-steps", type=int, default=24)
    certify.add_argument("--format", choices=("json", "html"), default="json")
    certify.add_argument("--output", help="Output path; defaults to the selected format")
    certify.set_defaults(handler=command_certify)

    dataset = subparsers.add_parser(
        "dataset",
        help="Explain a bounded sample cohort and build a reasoning-fingerprint atlas",
    )
    _add_model_arguments(dataset)
    dataset.add_argument("--input", required=True, help="Batched tensor input file")
    dataset.add_argument("--labels", help="Optional per-sample target labels")
    dataset.add_argument("--mode", choices=("fast", "gradient", "causal"), default="fast")
    dataset.add_argument("--class-names", help="Comma-separated class labels")
    dataset.add_argument("--max-samples", type=int, default=128)
    dataset.add_argument("--fingerprint-dimensions", type=int, default=128)
    dataset.add_argument("--max-layers", type=int, default=256)
    dataset.add_argument("--max-units", type=int, default=128)
    dataset.add_argument("--output", default="vraven_dataset_atlas.json")
    dataset.set_defaults(handler=command_dataset)

    compare = subparsers.add_parser(
        "compare",
        help="Compare two saved models on the same bounded sample cohort",
    )
    compare.add_argument("model_a", help="Left model file")
    compare.add_argument("model_b", help="Right model file")
    _add_model_loading_arguments(compare)
    compare.add_argument("--model-a-factory", help="Optional factory override for model A")
    compare.add_argument("--model-b-factory", help="Optional factory override for model B")
    compare.add_argument("--model-a-factory-kwargs", help="Optional JSON kwargs for model A")
    compare.add_argument("--model-b-factory-kwargs", help="Optional JSON kwargs for model B")
    compare.add_argument("--input", required=True, help="Batched tensor input file")
    compare.add_argument("--labels", help="Optional per-sample target labels")
    compare.add_argument("--mode", choices=("fast", "gradient", "causal"), default="fast")
    compare.add_argument("--class-names", help="Comma-separated class labels")
    compare.add_argument("--left-name", default="model_a")
    compare.add_argument("--right-name", default="model_b")
    compare.add_argument("--max-samples", type=int, default=128)
    compare.add_argument("--fingerprint-dimensions", type=int, default=128)
    compare.add_argument("--max-layers", type=int, default=256)
    compare.add_argument("--max-units", type=int, default=128)
    compare.add_argument("--output", default="vraven_model_comparison.json")
    compare.set_defaults(handler=command_compare)

    evolution = subparsers.add_parser(
        "evolution",
        help="Track one input through two or more ordered model checkpoints",
    )
    evolution.add_argument("models", nargs="+", help="Ordered checkpoint model files")
    _add_model_loading_arguments(evolution)
    evolution.add_argument("--input", required=True, help="Input tensor file")
    evolution.add_argument("--target", type=int)
    evolution.add_argument("--checkpoint-names", help="Comma-separated checkpoint names")
    evolution.add_argument("--mode", choices=("fast", "gradient", "causal"), default="fast")
    evolution.add_argument("--class-names", help="Comma-separated class labels")
    evolution.add_argument("--fingerprint-dimensions", type=int, default=128)
    evolution.add_argument("--max-layers", type=int, default=256)
    evolution.add_argument("--max-units", type=int, default=128)
    evolution.add_argument("--output", default="vraven_checkpoint_evolution.json")
    evolution.set_defaults(handler=command_evolution)

    anatomy = subparsers.add_parser(
        "anatomy",
        help="Inspect measured local anatomy of Linear layers for one sample",
    )
    _add_model_arguments(anatomy)
    anatomy.add_argument("--input", required=True, help="Input tensor file")
    anatomy.add_argument("--sample-index", type=int, default=0)
    anatomy.add_argument("--target", type=int)
    anatomy.add_argument("--layers", help="Optional comma-separated Linear layer names")
    anatomy.add_argument("--top-units", type=int, default=12)
    anatomy.add_argument("--output", default="vraven_internal_anatomy.json")
    anatomy.set_defaults(handler=command_anatomy)

    decompile_demo = subparsers.add_parser(
        "decompile-demo",
        help="Run a deterministic compiler-assisted causal decompilation demonstration",
    )
    decompile_demo.add_argument("--samples", type=int, default=48)
    decompile_demo.add_argument("--focus-index", type=int, default=0)
    decompile_demo.add_argument("--profile", choices=("quick", "standard", "publication"), default="quick")
    _add_report_arguments(decompile_demo, default_stem="vraven_decision_proof")
    decompile_demo.set_defaults(handler=command_decompile_demo)

    capture = subparsers.add_parser(
        "capture",
        help="Compile a PyTorch model into the architecture-neutral VRAVEN Mechanism IR",
    )
    _add_model_arguments(capture)
    capture.add_argument("--input", required=True, help="Example .npy, .npz or trusted tensor .pt input")
    capture.add_argument("--output", default="mechanism_ir.json")
    capture.set_defaults(handler=command_capture)

    reverse_parser = subparsers.add_parser(
        "reverse",
        help="Recover concepts, causal mechanism, executable programme and Decision Proof",
    )
    _add_model_arguments(reverse_parser)
    reverse_parser.add_argument("--reference", required=True, help="Reference .npy, .npz or trusted tensor .pt data")
    reverse_parser.add_argument("--labels", help="Optional .npy, .npz, .csv or tensor .pt labels")
    reverse_parser.add_argument("--focus-index", type=int, default=0)
    reverse_parser.add_argument("--target", type=int)
    reverse_parser.add_argument("--class-names", help="Comma-separated class labels")
    reverse_parser.add_argument("--input-groups", help="JSON file mapping human group names to feature indices")
    reverse_parser.add_argument(
        "--validated-input-groups",
        help=(
            "Comma-separated group names whose semantic identity is independently "
            "established. Use only for planted mechanisms or externally reviewed domain "
            "ontologies; ordinary post-hoc associations must not be upgraded."
        ),
    )
    reverse_parser.add_argument("--output-path", help="Dot path to logits in a structured model output")
    reverse_parser.add_argument("--profile", choices=("quick", "standard", "publication"), default="standard")
    reverse_parser.add_argument("--no-counterfactual", action="store_true")
    reverse_parser.add_argument("--no-profiler", action="store_true")
    _add_report_arguments(reverse_parser, default_stem="vraven_decision_proof")
    reverse_parser.set_defaults(handler=command_reverse)

    inspect = subparsers.add_parser("inspect", help="Inspect a saved VRAVEN trace")
    inspect.add_argument("trace")
    inspect.set_defaults(handler=command_inspect)

    render = subparsers.add_parser("render", help="Render one standalone figure from a trace")
    render.add_argument("trace")
    render.add_argument("--visual", default="reasoning_dna")
    render.add_argument("--mode", choices=("publication_light", "interactive_dark"), default="publication_light")
    render.add_argument("--output", required=True)
    render.add_argument("--preset", choices=("single_column", "double_column", "square", "wide"), default="double_column")
    render.add_argument("--dpi", type=int, default=300)
    render.set_defaults(handler=command_render)

    raven = subparsers.add_parser(
        "raven",
        help="Display the optional blinking or dancing terminal raven",
    )
    raven.add_argument("--style", choices=("still", "blink", "dance"), default="blink")
    raven.add_argument("--cycles", type=int, default=2)
    raven.add_argument("--delay", type=float, default=0.18)
    raven.set_defaults(handler=command_raven)

    doctor = subparsers.add_parser("doctor", help="Report the local VRAVEN and PyTorch capabilities")
    doctor.set_defaults(handler=command_doctor)

    features = subparsers.add_parser(
        "features", help="List supported commands, modes, formats and visualisations"
    )
    features.set_defaults(handler=command_features)
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="PyTorch model file")
    _add_model_loading_arguments(parser)


def _add_model_loading_arguments(
    parser: argparse.ArgumentParser,
    *,
    nullable: bool = False,
) -> None:
    parser.add_argument(
        "--model-format",
        choices=("state_dict", "torchscript", "pickle"),
        default=None if nullable else "state_dict",
        help=(
            "Model storage format. VRAVEN instrumentation supports state_dict and trusted "
            "pickle models; TorchScript can be loaded but cannot expose PyTorch hooks."
        ),
    )
    parser.add_argument(
        "--model-factory",
        "--factory",
        dest="model_factory",
        default=None,
        help="Import path such as my_project.models:Classifier",
    )
    parser.add_argument(
        "--factory-kwargs",
        default=None if nullable else "{}",
        help="JSON object passed to the model factory",
    )
    parser.add_argument("--state-dict-key")
    parser.add_argument("--strip-prefix")
    parser.add_argument("--device", default=None if nullable else "cpu")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--trusted-model", action="store_true", help="Allow Python-pickle model loading")


def _add_report_arguments(parser: argparse.ArgumentParser, *, default_stem: str) -> None:
    parser.add_argument(
        "--report-format",
        choices=REPORT_FORMATS,
        default="html",
        help="Write exactly one report in the selected format (default: html)",
    )
    destinations = parser.add_mutually_exclusive_group()
    destinations.add_argument(
        "--output",
        help="Single report destination; its extension must match --report-format",
    )
    destinations.add_argument(
        "--output-dir",
        help="Compatibility option: place the single report in this directory",
    )
    parser.set_defaults(report_stem=default_stem)


def _report_destination(args: argparse.Namespace, default_stem: str) -> Path:
    stem = getattr(args, "report_stem", default_stem)
    report_format = str(args.report_format)
    if args.output:
        destination = Path(args.output)
    elif args.output_dir:
        destination = Path(args.output_dir) / f"{stem}.{report_format}"
    else:
        destination = Path(f"{stem}.{report_format}")
    return destination


def _comma_separated(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_separated_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in _comma_separated(value))
    if not values:
        raise ValueError("At least one numeric value is required")
    return values


def _public_inputs(model_args: tuple[Any, ...], model_kwargs: dict[str, Any]) -> Any:
    if model_kwargs:
        return model_kwargs
    if len(model_args) == 1:
        return model_args[0]
    return model_args


def _single_tensor_input(
    model_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
    *,
    command: str,
) -> torch.Tensor:
    if model_kwargs or len(model_args) != 1 or not isinstance(model_args[0], torch.Tensor):
        raise ValueError(f"vraven {command} requires one positional tensor input")
    if not model_args[0].is_floating_point():
        raise ValueError(f"vraven {command} requires a floating-point tensor input")
    return model_args[0]


def _select_sample(inputs: torch.Tensor, index: int) -> torch.Tensor:
    if inputs.ndim == 0:
        if index != 0:
            raise IndexError("sample index is out of range for a scalar input")
        return inputs.reshape(1)
    if index < 0 or index >= inputs.shape[0]:
        raise IndexError(f"sample index {index} is out of range for batch size {inputs.shape[0]}")
    return inputs[index : index + 1]


def _attack_selection(value: str) -> tuple[str, ...] | None:
    normalised = value.strip().lower()
    if normalised == "quick":
        return ("fgsm", "pgd_linf")
    if normalised == "full":
        return None
    attacks = _comma_separated(normalised)
    if not attacks:
        raise ValueError("--attacks must be quick, full or a comma-separated attack list")
    return attacks


def _write_json_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("JSON analysis output must use a .json extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _terminal_percent(value: Any) -> str:
    if value is None:
        return "not applicable"
    return f"{100.0 * float(value):.2f}%"


def _apply_bundle_to_analyse_args(
    args: argparse.Namespace,
    bundle: ModelBundle | None,
) -> None:
    """Fill omitted beginner options from a validated bundle or safe defaults."""

    defaults = bundle.defaults if bundle is not None else {}
    if bundle is not None:
        args.model = str(bundle.model)
        args.input = args.input or str(bundle.input)
        args.labels = args.labels or (str(bundle.labels) if bundle.labels else None)
        args.model_format = args.model_format or bundle.model_format
        args.model_factory = args.model_factory or bundle.factory
        args.factory_kwargs = args.factory_kwargs or json.dumps(bundle.factory_kwargs)
        args.state_dict_key = args.state_dict_key or bundle.state_dict_key
        args.strip_prefix = args.strip_prefix or bundle.strip_prefix
        args.device = args.device or bundle.device
        args.class_names = args.class_names or (
            ",".join(bundle.class_names) if bundle.class_names else None
        )
    else:
        args.model_format = args.model_format or "state_dict"
        args.factory_kwargs = args.factory_kwargs or "{}"
        args.device = args.device or "cpu"

    args.task = args.task or defaults.get("task") or "explain"
    args.preset = args.preset or defaults.get("preset") or "quick"
    args.format = args.format or defaults.get("format") or "html"
    if args.target is None and defaults.get("target") is not None:
        args.target = int(defaults["target"])
    if args.target_class is None and defaults.get("target_class") is not None:
        args.target_class = int(defaults["target_class"])
    if args.output is None and defaults.get("output") is not None:
        output = Path(str(defaults["output"]))
        if bundle is not None and not output.is_absolute():
            output = (bundle.root / output).resolve()
            if not output.is_relative_to(bundle.root):
                raise ValueError("defaults.output must remain inside the bundle directory")
        args.output = str(output)

    allowed_tasks = {
        "explain",
        "adversarial",
        "counterfactual",
        "stability",
        "lrp",
        "certify",
        "dataset",
        "anatomy",
    }
    if args.task not in allowed_tasks:
        raise ValueError(f"Bundle default task is unsupported: {args.task!r}")
    if args.preset not in {"quick", "standard", "thorough"}:
        raise ValueError(f"Bundle default preset is unsupported: {args.preset!r}")
    if args.format not in {"json", "html"}:
        raise ValueError(f"Bundle default format is unsupported: {args.format!r}")


def _forward_model_loading_arguments(args: argparse.Namespace) -> list[str]:
    """Serialise model-loading options for the beginner workflow router."""

    values = ["--model-format", args.model_format, "--device", args.device]
    optional = (
        ("--model-factory", args.model_factory),
        ("--factory-kwargs", args.factory_kwargs if args.factory_kwargs != "{}" else None),
        ("--state-dict-key", args.state_dict_key),
        ("--strip-prefix", args.strip_prefix),
    )
    for flag, value in optional:
        if value is not None:
            values.extend([flag, str(value)])
    if args.non_strict:
        values.append("--non-strict")
    if args.trusted_model:
        values.append("--trusted-model")
    return values


def _emit_saved_result(
    payload: Mapping[str, Any],
    title: str,
    rows: Mapping[str, object],
    destination: str | Path,
) -> None:
    resolved = Path(destination).resolve()
    interactive = TerminalUI().interactive
    emit_result(payload, title=title, rows=rows, footer=f"Saved: {resolved}")
    if not interactive:
        print(f"Wrote VRAVEN result to {resolved}", file=sys.stderr)


def _load_cli_model(args: argparse.Namespace) -> torch.nn.Module:
    return _load_model_path(args, args.model)


def _load_model_path(
    args: argparse.Namespace,
    model_path: str | Path,
    *,
    factory_path: str | None = None,
    factory_kwargs_json: str | None = None,
) -> torch.nn.Module:
    if args.model_format == "torchscript":
        raise ValueError(
            "TorchScript modules do not expose the forward hooks required by VRAVEN. "
            "Use a state_dict with --model-factory (recommended), or a trusted pickle "
            "with --trusted-model."
        )
    factory = None
    selected_factory = factory_path or args.model_factory
    if selected_factory:
        callable_object = _import_object(selected_factory)
        kwargs = json.loads(factory_kwargs_json or args.factory_kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("--factory-kwargs must decode to a JSON object")

        def instantiate_model():
            return callable_object(**kwargs)

        factory = instantiate_model
    return load_pytorch_model(
        model_path,
        model_factory=factory,
        format=args.model_format,
        state_dict_key=args.state_dict_key,
        strip_prefix=args.strip_prefix,
        map_location=args.device,
        strict=not args.non_strict,
        allow_unsafe_pickle=args.trusted_model,
    )


def _import_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError("Import paths must use module:object syntax")
    module_name, object_path = path.split(":", 1)
    current = importlib.import_module(module_name)
    for part in object_path.split("."):
        current = getattr(current, part)
    if not callable(current):
        raise TypeError(f"Imported object {path!r} is not callable")
    return current


def _load_inputs(path: str | Path) -> tuple[tuple[Any, ...], dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return (torch.as_tensor(np.load(source, allow_pickle=False)),), {}
    if suffix == ".npz":
        archive = np.load(source, allow_pickle=False)
        keys = list(archive.files)
        if len(keys) == 1:
            return (torch.as_tensor(archive[keys[0]]),), {}
        return (), {key: torch.as_tensor(archive[key]) for key in keys}
    if suffix in {".pt", ".pth"}:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(payload, torch.Tensor):
            return (payload,), {}
        if isinstance(payload, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in payload):
            return tuple(payload), {}
        if isinstance(payload, Mapping) and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return (), dict(payload)
        raise ValueError("Trusted input .pt files must contain a tensor, tensor sequence or tensor mapping")
    raise ValueError("Input files must be .npy, .npz, .pt or .pth")


def _load_labels(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False)).reshape(-1)
    if suffix == ".npz":
        archive = np.load(source, allow_pickle=False)
        if len(archive.files) != 1:
            raise ValueError("A label NPZ must contain exactly one array")
        return np.asarray(archive[archive.files[0]]).reshape(-1)
    if suffix in {".pt", ".pth"}:
        value = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError("A label PT file must contain a tensor")
        return value.detach().cpu().numpy().reshape(-1)
    if suffix in {".csv", ".txt"}:
        return np.asarray(np.loadtxt(source, delimiter="," if suffix == ".csv" else None)).reshape(-1)
    raise ValueError("Label files must be .npy, .npz, .csv, .txt, .pt or .pth")


def _slice_loaded_inputs(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], index: int
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def select(item: Any) -> Any:
        if isinstance(item, torch.Tensor) and item.ndim > 0:
            return item[index : index + 1]
        return item
    return tuple(select(item) for item in args), {key: select(value) for key, value in kwargs.items()}


def _load_json_mapping(path: str | Path) -> dict[str, list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input groups must be a JSON object")
    return {str(key): [int(item) for item in value] for key, value in payload.items()}


def _decompile_profile(name: str) -> DecompileConfig:
    if name == "quick":
        return DecompileConfig.quick()
    if name == "publication":
        return DecompileConfig.publication()
    return DecompileConfig()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        parser.exit(2, f"vraven: {type(error).__name__}: {error}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
