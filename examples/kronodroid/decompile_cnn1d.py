#!/usr/bin/env python3
"""Generate a compiler-assisted VRAVEN Decision Proof for the trained Kronodroid CNN1D.

This example uses a balanced reference cohort and a single malware focus sample.  The
feature-group labels are association evidence only.  They are deliberately prefixed with
"Associated with" and do not count as validated semantic grounding in the certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vraven.decompiler import DecompileConfig, VRavenDecompiler

try:
    from .models import MalwareCNN1D
except ImportError:  # Direct script execution.
    from models import MalwareCNN1D


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="VRAVEN repository root")
    parser.add_argument("--reference-per-class", type=int, default=32)
    parser.add_argument("--focus-malware-ordinal", type=int, default=27)
    parser.add_argument("--profile", choices=("quick", "standard", "publication"), default="quick")
    parser.add_argument(
        "--output",
        default="vraven_outputs/compiler_decompiler/kronodroid_cnn1d_decision_proof",
    )
    parser.add_argument("--no-counterfactual", action="store_true")
    parser.add_argument("--no-profiler", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    split_path = root / "Models/kronodroid/split.npz"
    model_path = root / "Models/kronodroid/cnn1d/model.pt"
    split = np.load(split_path, allow_pickle=True)
    x_test = np.asarray(split["x_test"], dtype=np.float32)
    y_test = np.asarray(split["y_test"], dtype=np.int64)
    feature_names = [str(value) for value in split["feature_names"]]

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = MalwareCNN1D(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=int(checkpoint.get("num_classes", 2)),
    ).eval()
    model.load_state_dict(checkpoint["model_state"])

    benign = np.where(y_test == 0)[0][: args.reference_per_class]
    malware = np.where(y_test == 1)[0][: args.reference_per_class]
    reference_indices = np.concatenate([benign, malware])
    focus_index = int(malware[args.focus_malware_ordinal])
    reference = torch.from_numpy(x_test[reference_indices])
    focus = torch.from_numpy(x_test[focus_index : focus_index + 1])

    if args.profile == "publication":
        config = DecompileConfig.publication()
    elif args.profile == "standard":
        config = DecompileConfig()
    else:
        config = DecompileConfig.quick()
    config.promote_association_labels = True
    config.programme.interventional_weight = 16.0
    config.run_counterfactual = not args.no_counterfactual
    config.run_profiler = not args.no_profiler
    if args.profile == "quick":
        config.grounding_permutations = 100
        config.causal.bootstrap_repeats = 40
        config.counterfactual.maximum_verified_candidates = 700

    input_groups = {
        "Linux system-call behaviour": tuple(range(0, 289)),
        "Android permissions": tuple(range(289, 462)),
        "APK packaging and submission metadata": tuple(range(462, 466)),
        "Network contact behaviour": (466,),
        "Android component structure": tuple(range(467, 476)),
    }

    result = VRavenDecompiler(config).reverse(
        model,
        (reference,),
        labels=y_test[reference_indices],
        sample_ids=tuple(f"kronodroid_test_{int(index)}" for index in reference_indices),
        focus_args=(focus,),
        target=1,
        class_names={0: "Benign", 1: "Malware"},
        input_groups=input_groups,
    )

    output = root / args.output
    result.save(output)
    result.write_report(output / "vraven_decision_proof.html", include_plotlyjs=True)
    visual_names = (
        "model_xray",
        "concept_genome",
        "causal_proof",
        "programme_proof",
        "contrastive_proof",
        "counterfactual_twin",
        "dark_matter",
    )
    for visual in visual_names:
        for mode in ("interactive_dark", "publication_light"):
            figure = result.figure(visual, mode=mode)
            figure.write_html(
                output / f"{visual}_{mode}.html",
                include_plotlyjs="directory",
                full_html=True,
            )

    metadata = {
        "focus_dataset_index": focus_index,
        "focus_true_label": int(y_test[focus_index]),
        "reference_indices": reference_indices.tolist(),
        "feature_groups": {key: list(value) for key, value in input_groups.items()},
        "feature_names": feature_names,
        "summary": result.summary(),
    }
    (output / "example_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result.summary(), indent=2))
    print(f"Wrote Decision Proof to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
