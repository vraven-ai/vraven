from __future__ import annotations

import numpy as np
import pytest

from vraven.darkside import analyse_circuit_multiplicity


def test_circuit_multiplicity_finds_stable_core_and_alternatives() -> None:
    masks = {
        "seed_1": [1, 1, 0, 0, 1],
        "seed_2": [1, 1, 0, 1, 0],
        "seed_3": [1, 1, 1, 0, 0],
        "seed_4": [1, 1, 0, 0, 1],
        "seed_5": [1, 1, 0, 1, 0],
        "seed_6": [1, 1, 1, 0, 0],
        "seed_7": [1, 1, 0, 0, 1],
        "seed_8": [1, 1, 0, 1, 0],
        "seed_9": [1, 1, 1, 0, 0],
        "seed_10": [1, 1, 0, 0, 1],
        "seed_11": [1, 1, 0, 1, 0],
        "seed_12": [1, 1, 1, 0, 0],
        "seed_13": [1, 1, 0, 0, 1],
        "seed_14": [1, 1, 0, 1, 0],
        "seed_15": [1, 1, 1, 0, 0],
        "seed_16": [1, 1, 0, 0, 1],
        "seed_17": [1, 1, 0, 1, 0],
        "seed_18": [1, 1, 1, 0, 0],
        "seed_19": [1, 1, 0, 0, 1],
        "seed_20": [1, 1, 0, 1, 0],
    }
    result = analyse_circuit_multiplicity(
        masks,
        component_labels=["core_a", "core_b", "route_x", "route_y", "route_z"],
        stable_threshold=0.80,
    )

    assert result.metrics["unique_circuits"] == 3
    assert result.metrics["stable_core_size"] == 2
    assert result.metrics["multiplicity_index"] > 0
    assert result.tables["stable_core_indices"] == [0, 1]


def test_circuit_multiplicity_validates_alignment_and_binary_values() -> None:
    with pytest.raises(ValueError):
        analyse_circuit_multiplicity({"a": [1, 0], "b": [1, 0, 1]})
    with pytest.raises(ValueError):
        analyse_circuit_multiplicity(np.array([[1, 0], [1, 2]]))


def test_circuit_multiplicity_figure_has_both_modes() -> None:
    result = analyse_circuit_multiplicity(
        np.asarray([[1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]]),
        confidence=0.80,
        stable_threshold=0.50,
    )
    dark = result.figure(mode="interactive_dark")
    light = result.figure(mode="publication_light")
    assert dark.layout.paper_bgcolor == "#030A12"
    assert light.layout.paper_bgcolor == "#FFFFFF"
    assert len(dark.data) == len(light.data)
