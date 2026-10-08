"""Dataset similarity, matched model comparison and checkpoint evolution."""

from pathlib import Path

import torch

import vraven


def build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(6, 18),
        torch.nn.GELU(),
        torch.nn.Linear(18, 3),
    )


def main() -> None:
    torch.manual_seed(31)
    samples = torch.randn(12, 6)
    targets = torch.arange(12) % 3
    baseline = build_model(1)
    candidate = build_model(2)
    output = Path("vraven_dataset_demo")
    output.mkdir(exist_ok=True)

    dataset_report = vraven.explain_dataset(
        baseline,
        samples,
        targets=targets,
        mode="gradient",
        max_samples=12,
        fingerprint_dimensions=64,
    )
    dataset_report.save(output / "dataset_report.json")
    dataset_report.reasoning_similarity(mode="publication_light").write_html(
        output / "reasoning_similarity.html",
        include_plotlyjs=True,
    )
    dataset_report.reasoning_landscape(mode="interactive_dark").write_html(
        output / "reasoning_landscape.html",
        include_plotlyjs=True,
    )

    comparison = vraven.compare_models(
        baseline,
        candidate,
        samples,
        targets=targets,
        max_samples=12,
        left_name="baseline",
        right_name="candidate",
    )
    comparison.alignment(mode="publication_light").write_html(
        output / "model_alignment.html",
        include_plotlyjs=True,
    )

    evolution = vraven.compare_checkpoints(
        [build_model(3), build_model(4), build_model(5)],
        samples[:1],
        target=1,
        checkpoint_names=["checkpoint 1", "checkpoint 2", "checkpoint 3"],
        mode="gradient",
    )
    evolution.figure(mode="interactive_dark").write_html(
        output / "model_evolution.html",
        include_plotlyjs=True,
    )

    print(dataset_report.summary())
    print(comparison.summary())
    print(evolution.summary())


if __name__ == "__main__":
    main()
