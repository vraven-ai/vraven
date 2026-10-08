"""Minimal live-model circuit, shortcut and synergy audit."""

from __future__ import annotations

import torch

from vraven.darkside import DarkSideSession, InputFeatureGroup, UnitGroup


class SmallClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(12, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 8),
            torch.nn.GELU(),
        )
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


model = SmallClassifier().eval()
sample = torch.randn(1, 12)
session = DarkSideSession(model, sample, target=1)

circuit = session.decision_circuitry(
    layer_names=("encoder.0", "encoder.2"),
    group_size=4,
    relative_score_tolerance=0.10,
)
print(circuit.summary())
circuit.figure(mode="interactive_dark").write_html("darkside_circuit.html")

shortcut = session.shortcut_detection(
    (
        InputFeatureGroup((0, 1), label="candidate shortcut A"),
        InputFeatureGroup((8, 9, 10, 11), label="candidate shortcut B"),
    )
)
shortcut.figure(mode="publication_light").write_html("darkside_shortcut.html")

synergy = session.causal_synergy(
    (
        UnitGroup("encoder.2", (0, 1), label="community A"),
        UnitGroup("encoder.2", (2, 3), label="community B"),
        UnitGroup("encoder.2", (4, 5), label="community C"),
    )
)
synergy.figure(mode="interactive_dark").write_html("darkside_synergy.html")
