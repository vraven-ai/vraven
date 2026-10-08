"""Public VRAVEN walkthrough using torchvision MobileNetV2 and one CC BY-SA image.

The example intentionally keeps acquisition, preprocessing, explanation and export
visible so the documentation result can be reproduced without private artefacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import vraven

IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/"
    "Labrador_Retriever_Adulto.JPG/1280px-Labrador_Retriever_Adulto.JPG"
)
IMAGE_PAGE = "https://commons.wikimedia.org/wiki/File:Labrador_Retriever_Adulto.JPG"
IMAGE_AUTHOR = "Miguel Ángel Martínez González"
IMAGE_LICENSE = "CC BY-SA 3.0"


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "labrador.jpg"
    if not image_path.exists():
        request = Request(IMAGE_URL, headers={"User-Agent": "VRAVEN-documentation/1.0"})
        with urlopen(request, timeout=60) as response:
            image_path.write_bytes(response.read())

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights).eval()
    image = Image.open(image_path).convert("RGB")
    inputs = weights.transforms()(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = model(inputs).softmax(dim=1)[0]
    top_probability, top_index = probabilities.max(dim=0)
    top_probabilities, top_indices = probabilities.topk(5)
    target = int(top_index)

    report = vraven.explain(
        model,
        inputs,
        target=target,
        mode="causal",
        class_names=weights.meta["categories"],
        max_layers=96,
        max_units=256,
        causal_top_k=3,
        metadata={
            "walkthrough": "mobilenet_v2_imagenet",
            "image_source": IMAGE_PAGE,
            "image_author": IMAGE_AUTHOR,
            "image_license": IMAGE_LICENSE,
        },
    )

    report.write_report(
        output / "vraven_report.html",
        visuals=(
            "input_attribution",
            "decision_flow",
            "activation_energy",
            "evidence_trajectory",
            "causal_pathway",
            "mechanism_concordance",
        ),
        mode="interactive_dark",
    )
    for visual in (
        "input_attribution",
        "decision_flow",
        "activation_energy",
        "causal_pathway",
    ):
        report.export(visual, output / f"{visual}.html", mode="interactive_dark")

    causal = sorted(
        report.trace.causal_interventions,
        key=lambda item: abs(item.score_delta),
        reverse=True,
    )
    summary = {
        "model": "torchvision MobileNetV2",
        "weights": str(weights),
        "predicted_class_index": target,
        "predicted_class": weights.meta["categories"][target],
        "confidence": float(top_probability),
        "top_five": [
            {
                "class": weights.meta["categories"][int(index)],
                "probability": float(probability),
            }
            for probability, index in zip(top_probabilities, top_indices, strict=True)
        ],
        "layers_captured": len(report.trace.layers),
        "causal_interventions": len(causal),
        "largest_intervention": (
            {
                "layer": causal[0].layer_name,
                "baseline_target_score": causal[0].baseline_target_score,
                "intervened_target_score": causal[0].intervened_target_score,
                "score_delta": causal[0].score_delta,
                "relative_delta": causal[0].relative_delta,
            }
            if causal
            else None
        ),
        "image": {
            "source": IMAGE_PAGE,
            "author": IMAGE_AUTHOR,
            "license": IMAGE_LICENSE,
        },
        "evidence": {
            "input_attribution": "observational input-times-gradient",
            "causal_interventions": "zero-ablation of the three selected layers",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vraven_walkthrough_output/mobilenet_v2"),
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
