from __future__ import annotations

import numpy as np
import torch

from vraven.darkside import (
    TraceDataset,
    build_circuit_wiring,
    discover_decision_circuitry,
    inspect_internal_anatomy,
    map_class_conditional_unit_field,
    print_internal_anatomy,
)


class DeepToy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.nn.Linear(4, 4, bias=False)
        self.relu = torch.nn.ReLU()
        self.mid = torch.nn.Linear(4, 4, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.output = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(4))
            self.mid.weight.copy_(torch.eye(4))
            self.output.weight.copy_(
                torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.1, 0.1]])
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.relu2(self.mid(self.relu(self.hidden(inputs)))))


def test_internal_anatomy_prints_and_figures() -> None:
    model = DeepToy().eval()
    sample = torch.tensor([[2.0, 1.0, 0.2, 0.1]])
    result = inspect_internal_anatomy(model, sample, target=1, top_units=4)
    text = print_internal_anatomy(result)
    assert "LAYER hidden" in text
    assert result.metrics["layers"] == 3
    figure = result.figure(mode="publication_light")
    assert len(figure.data) >= 1


def test_circuit_wiring_uses_measured_edge_mass() -> None:
    model = DeepToy().eval()
    sample = torch.tensor([[2.0, 1.0, 0.2, 0.1]])
    circuit = discover_decision_circuitry(
        model,
        sample,
        target=1,
        layer_names=("hidden", "mid", "output"),
        group_size=1,
        relative_score_tolerance=0.25,
    )
    wiring = build_circuit_wiring(model, sample, circuit)
    assert wiring.analysis == "circuit_wiring"
    assert wiring.metrics["edges"] >= 1
    assert wiring.metrics["total_edge_mass"] > 0
    figure = wiring.figure(mode="interactive_dark")
    assert len(figure.data) >= 1


def test_class_conditional_unit_field_heatmap() -> None:
    rng = np.random.default_rng(11)
    labels = np.array([0] * 12 + [1] * 12, dtype=object)
    influences = {
        "layer1#0": np.vstack(
            [rng.normal(0.0, 1.0, (12, 6)), rng.normal(1.5, 1.0, (12, 6))]
        ),
        "layer2#0": np.vstack(
            [rng.normal(-0.5, 1.0, (12, 6)), rng.normal(0.8, 1.0, (12, 6))]
        ),
    }
    dataset = TraceDataset(
        sample_ids=tuple(f"s{i}" for i in range(24)),
        true_labels=labels,
        predicted_labels=labels.copy(),
        target_scores=rng.normal(size=24),
        representations={name: values for name, values in influences.items()},
        unit_influences=influences,
        unit_energies={name: np.abs(values) for name, values in influences.items()},
        activation_rms={name: np.ones(24) for name in influences},
        timing_ms={name: np.ones(24) for name in influences},
        fingerprints=rng.normal(size=(24, 8)),
    )
    field = map_class_conditional_unit_field(dataset, class_a=0, class_b=1)
    assert field.metrics["units_mapped"] == 12
    figure = field.figure(mode="publication_light")
    assert figure.data[0].type == "heatmap"
