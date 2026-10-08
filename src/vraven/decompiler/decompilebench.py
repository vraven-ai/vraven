"""Small PyTorch-native known-mechanism benchmark for VRAVEN development.

This is not presented as a complete community benchmark.  It provides reproducible models
with planted concepts, distractors, redundant pathways and known counterfactual controls so
that core decompiler behaviour can be tested before analysing naturally trained models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .decompiler import DecompilationResult, DecompileConfig, VRavenDecompiler


@dataclass(slots=True)
class PlantedMechanism:
    name: str
    input_concepts: dict[str, tuple[int, ...]]
    causal_concepts: tuple[str, ...]
    distractor_concepts: tuple[str, ...]
    expected_signs: dict[str, int]
    description: str


@dataclass(slots=True)
class DecompileBenchResult:
    model_name: str
    samples: int
    capture_equivalence: bool | None
    operator_coverage: float
    programme_validation_r2: float
    programme_decision_agreement: float
    interventional_sign_agreement: float | None
    counterfactual_model_flip: bool
    causal_semantic_coverage: float
    passed: bool
    thresholds: dict[str, float]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlantedConceptMLP(nn.Module):
    """MLP with two causal concepts, one redundant route and two distractors."""

    def __init__(self) -> None:
        super().__init__()
        self.concepts = nn.Linear(8, 6, bias=False)
        self.readout = nn.Linear(6, 2, bias=False)
        with torch.no_grad():
            self.concepts.weight.zero_()
            # C0: x0 + x1, C1: x2 - x3, C2: redundant C0, C3/C4 distractors, C5 inhibition.
            self.concepts.weight[0, 0] = 1.0
            self.concepts.weight[0, 1] = 1.0
            self.concepts.weight[1, 2] = 1.0
            self.concepts.weight[1, 3] = -1.0
            self.concepts.weight[2, 0] = 0.5
            self.concepts.weight[2, 1] = 0.5
            self.concepts.weight[3, 4] = 1.0
            self.concepts.weight[4, 5] = 1.0
            self.concepts.weight[5, 6] = 1.0
            self.readout.weight.zero_()
            margin_weights = torch.tensor([2.0, 1.6, 0.8, 0.0, 0.0, -1.1])
            self.readout.weight[1].copy_(margin_weights / 2.0)
            self.readout.weight[0].copy_(-margin_weights / 2.0)

    def forward(self, value: Tensor) -> Tensor:
        concepts = torch.relu(self.concepts(value))
        return self.readout(concepts)


class PlantedResidualMLP(nn.Module):
    """Residual model with a causal branch and a compensating alternative branch."""

    def __init__(self) -> None:
        super().__init__()
        self.primary = nn.Linear(8, 6, bias=False)
        self.alternative = nn.Linear(8, 6, bias=False)
        self.out = nn.Linear(6, 2, bias=False)
        with torch.no_grad():
            self.primary.weight.zero_()
            self.alternative.weight.zero_()
            for index in range(6):
                self.primary.weight[index, index % 4] = 0.9
                self.alternative.weight[index, (index + 1) % 4] = 0.45
            self.out.weight.zero_()
            weights = torch.tensor([1.6, 1.1, -0.8, 0.7, 0.0, 0.0])
            self.out.weight[1].copy_(weights / 2.0)
            self.out.weight[0].copy_(-weights / 2.0)

    def forward(self, value: Tensor) -> Tensor:
        primary = torch.relu(self.primary(value))
        alternative = torch.relu(self.alternative(value))
        merged = primary + alternative
        return self.out(merged)


def planted_dataset(samples: int = 96, seed: int = 17) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(samples, 8, generator=generator)
    margin = 2.0 * torch.relu(value[:, 0] + value[:, 1])
    margin += 1.6 * torch.relu(value[:, 2] - value[:, 3])
    margin += 0.8 * torch.relu(0.5 * value[:, 0] + 0.5 * value[:, 1])
    margin -= 1.1 * torch.relu(value[:, 6])
    labels = (margin > 0).long()
    return value, labels


def planted_specification(name: str = "planted_concept_mlp") -> PlantedMechanism:
    if name == "planted_residual_mlp":
        return PlantedMechanism(
            name=name,
            input_concepts={"primary_signal": (0, 1), "alternative_signal": (1, 2)},
            causal_concepts=("primary_signal", "alternative_signal"),
            distractor_concepts=("feature_6", "feature_7"),
            expected_signs={"primary_signal": 1, "alternative_signal": 1},
            description="Known redundant residual routes with planted compensating structure.",
        )
    return PlantedMechanism(
        name="planted_concept_mlp",
        input_concepts={
            "joint_signal": (0, 1),
            "difference_signal": (2, 3),
            "negative_control": (6,),
            "distractors": (4, 5, 7),
        },
        causal_concepts=("joint_signal", "difference_signal", "negative_control"),
        distractor_concepts=("distractors",),
        expected_signs={"joint_signal": 1, "difference_signal": 1, "negative_control": -1},
        description="Two causal signals, one redundant route, one inhibitory signal and unused distractors.",
    )


def run_decompile_bench(
    *,
    model_name: str = "planted_concept_mlp",
    samples: int = 96,
    profile: str = "quick",
    seed: int = 17,
) -> tuple[DecompileBenchResult, DecompilationResult]:
    if model_name == "planted_residual_mlp":
        model: nn.Module = PlantedResidualMLP().eval()
    else:
        model = PlantedConceptMLP().eval()
    inputs, labels = planted_dataset(samples, seed)
    config = DecompileConfig.quick() if profile == "quick" else DecompileConfig()
    config.promote_association_labels = True
    config.programme.interventional_weight = 64.0
    config.grounding_permutations = 80 if profile == "quick" else 300
    config.run_profiler = False
    result = VRavenDecompiler(config).reverse(
        model,
        (inputs,),
        labels=labels,
        focus_args=(inputs[:1],),
        class_names=["negative", "positive"],
        input_groups=planted_specification(model_name).input_concepts,
        validated_input_groups=tuple(
            planted_specification(model_name).input_concepts
        ),
    )
    thresholds = {
        "operator_coverage": 0.80,
        "programme_validation_r2": 0.60,
        "programme_decision_agreement": 0.80,
        "interventional_sign_agreement": 0.50,
    }
    passed = bool(
        result.captured.ir.coverage.numeric_equivalence_passed
        and result.certificate.coverage.operator_coverage >= thresholds["operator_coverage"]
        and result.programme.fidelity.validation_r2 >= thresholds["programme_validation_r2"]
        and result.programme.fidelity.validation_decision_agreement
        >= thresholds["programme_decision_agreement"]
        and (result.programme.fidelity.interventional_sign_agreement or 0.0)
        >= thresholds["interventional_sign_agreement"]
    )
    bench = DecompileBenchResult(
        model_name=model_name,
        samples=samples,
        capture_equivalence=result.captured.ir.coverage.numeric_equivalence_passed,
        operator_coverage=result.certificate.coverage.operator_coverage,
        programme_validation_r2=result.programme.fidelity.validation_r2,
        programme_decision_agreement=result.programme.fidelity.validation_decision_agreement,
        interventional_sign_agreement=result.programme.fidelity.interventional_sign_agreement,
        counterfactual_model_flip=(
            result.counterfactual.behaviourally_verified if result.counterfactual else False
        ),
        causal_semantic_coverage=result.certificate.coverage.causal_semantic_coverage,
        passed=passed,
        thresholds=thresholds,
        summary=result.summary(),
    )
    return bench, result
