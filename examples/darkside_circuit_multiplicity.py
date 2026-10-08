"""Inspect whether repeated circuit searches recover one stable mechanism."""

from __future__ import annotations

from vraven.darkside import analyse_circuit_multiplicity

result = analyse_circuit_multiplicity(
    {
        "seed_1": [1, 1, 0, 0, 1, 0],
        "seed_2": [1, 1, 0, 1, 0, 0],
        "seed_3": [1, 1, 1, 0, 0, 0],
        "mean_baseline": [1, 1, 0, 0, 1, 0],
        "perturbed_input": [1, 1, 0, 1, 0, 0],
        "checkpoint_b": [1, 1, 1, 0, 0, 0],
        "seed_4": [1, 1, 0, 0, 1, 0],
        "seed_5": [1, 1, 0, 1, 0, 0],
        "seed_6": [1, 1, 1, 0, 0, 0],
        "seed_7": [1, 1, 0, 0, 1, 0],
        "seed_8": [1, 1, 0, 1, 0, 0],
        "seed_9": [1, 1, 1, 0, 0, 0],
    },
    component_labels=(
        "encoder.core_A",
        "encoder.core_B",
        "route_X",
        "route_Y",
        "route_Z",
        "unused_candidate",
    ),
    confidence=0.95,
    stable_threshold=0.75,
)

print(result.summary())
result.figure(mode="interactive_dark").write_html(
    "darkside_circuit_multiplicity.html",
    include_plotlyjs=True,
)
