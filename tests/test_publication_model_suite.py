from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_publication_model_suite as suite  # noqa: E402
from publication_models import build_publication_model  # noqa: E402
from train_publication_model_suite import (  # noqa: E402
    ARCHITECTURES,
    DATASET_ORDER,
    ROBUST_DATASETS,
    architecture_payload,
    build_model,
    expected_run_count,
    load_tabular,
    regimes_for,
)


@pytest.fixture
def synthetic_wdbc_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a deterministic WDBC-shaped fixture without requiring local datasets."""

    dataset_root = tmp_path / "DATASETS"
    destination = dataset_root / "wdbc" / "training_ready" / "wdbc.csv"
    destination.parent.mkdir(parents=True)
    rows = 80
    labels = np.asarray(["B", "M"] * (rows // 2))
    payload: dict[str, np.ndarray] = {
        "id": np.arange(10_000, 10_000 + rows),
        "diagnosis": labels,
    }
    base = np.linspace(-2.0, 2.0, rows)
    for index in range(30):
        payload[f"feature_{index:02d}"] = base + index * 0.1 + (labels == "M") * 0.25
    pd.DataFrame(payload).to_csv(destination, index=False)
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "DATASETS", dataset_root)


@pytest.mark.parametrize(
    ("kind", "kwargs", "shape"),
    [
        ("publication_residual_mlp", {"input_dim": 17, "num_classes": 3}, (4, 17)),
        ("publication_feature_transformer", {"input_dim": 17, "num_classes": 3}, (4, 17)),
        ("publication_sequence_cnn", {"input_channels": 9, "num_classes": 6}, (4, 128, 9)),
        (
            "publication_sequence_transformer",
            {"seq_len": 128, "input_channels": 9, "num_classes": 6},
            (4, 128, 9),
        ),
    ],
)
def test_publication_models_reconstruct_exactly(
    kind: str,
    kwargs: dict,
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(7)
    inputs = torch.randn(shape)
    model = build_publication_model(kind, **kwargs).eval()
    expected = model(inputs)
    rebuilt = build_publication_model(kind, **kwargs).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    actual = rebuilt(inputs)
    assert actual.shape == (shape[0], kwargs["num_classes"])
    torch.testing.assert_close(actual, expected)


def test_suite_has_26_configurations_and_130_seeded_checkpoints() -> None:
    configurations = sum(
        len(ARCHITECTURES[dataset]) * len(regimes_for(dataset))
        for dataset in DATASET_ORDER
    )
    assert configurations == 26
    assert expected_run_count(list(DATASET_ORDER), [11, 29, 47, 71, 101]) == 130
    assert {"kronodroid", "fashion_mnist", "uci_har"} == ROBUST_DATASETS


def test_wdbc_preprocessing_is_train_fitted_and_split_disjoint(
    synthetic_wdbc_dataset: None,
) -> None:
    data = load_tabular("wdbc", seed=11)
    assert data.x_train.shape[1] == 30
    assert data.x_train.dtype == np.float32
    assert set(data.train_ids).isdisjoint(data.val_ids)
    assert set(data.train_ids).isdisjoint(data.test_ids)
    assert set(data.val_ids).isdisjoint(data.test_ids)
    assert data.preprocessing["kind"] == "train_only_numeric_standardisation_and_one_hot"
    assert data.preprocessing["excluded_columns"] == ["id"]


def test_architecture_payload_builds_for_vector_data(
    synthetic_wdbc_dataset: None,
) -> None:
    data = load_tabular("wdbc", seed=29)
    for alias in ARCHITECTURES["wdbc"]:
        architecture = architecture_payload(alias, data, seed=29)
        model = build_model(architecture).eval()
        output = model(torch.from_numpy(data.x_val[:3]))
        assert output.shape == (3, 2)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is unavailable",
)
def test_publication_sequence_model_mps_smoke() -> None:
    device = torch.device("mps")
    model = build_publication_model(
        "publication_sequence_cnn",
        input_channels=9,
        num_classes=6,
    ).to(device)
    output = model(torch.randn(2, 128, 9, device=device))
    torch.mps.synchronize()
    assert output.shape == (2, 6)
    assert torch.isfinite(output).all()
