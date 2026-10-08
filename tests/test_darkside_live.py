from __future__ import annotations

import torch

from vraven.darkside import (
    DarkSideSession,
    InputFeatureGroup,
    UnitGroup,
    analyse_activation_mediation,
    analyse_causal_synergy,
    detect_shortcut_reliance,
    discover_decision_circuitry,
)


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(4, 4, bias=False)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(4))
            self.output.weight.copy_(
                torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.1, 0.1]])
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.relu(self.hidden(inputs)))


def sample() -> torch.Tensor:
    return torch.tensor([[2.0, 1.0, 0.2, 0.1]])


def test_sparse_circuit_retains_dominant_units() -> None:
    model = ToyModel().eval()
    result = discover_decision_circuitry(
        model,
        sample(),
        target=1,
        layer_names=("hidden",),
        group_size=1,
        relative_score_tolerance=0.15,
    )
    retained = [row for row in result.tables["groups"] if row["retained"]]
    assert result.metrics["retained_groups"] < result.metrics["candidate_groups"]
    assert any(row["indices"] == [0] for row in retained)
    assert result.metrics["original_prediction"] == result.metrics["final_prediction"]


def test_shortcut_audit_finds_the_dominant_coordinate() -> None:
    model = ToyModel().eval()
    result = detect_shortcut_reliance(
        model,
        sample(),
        target=1,
        feature_groups=(
            InputFeatureGroup((0,), label="dominant"),
            InputFeatureGroup((2, 3), label="weak"),
        ),
    )
    assert result.metrics["strongest_group"] == "dominant"


def test_pairwise_synergy_and_activation_mediation_run() -> None:
    model = ToyModel().eval()
    groups = (
        UnitGroup("hidden", (0,), axis=-1, label="strong"),
        UnitGroup("hidden", (1,), axis=-1, label="secondary"),
    )
    synergy = analyse_causal_synergy(model, sample(), target=1, groups=groups)
    assert synergy.metrics["pairs"] == 1
    mediation = analyse_activation_mediation(
        model,
        sample(),
        torch.zeros_like(sample()),
        target=1,
        layer_names=("hidden",),
    )
    assert mediation.metrics["strongest_recovery"] > 0


def test_session_wraps_live_analyses() -> None:
    session = DarkSideSession(ToyModel().eval(), sample(), target=1)
    result = session.decision_circuitry(
        layer_names=("hidden",), group_size=2, relative_score_tolerance=0.25
    )
    assert result.analysis == "decision_circuitry"
