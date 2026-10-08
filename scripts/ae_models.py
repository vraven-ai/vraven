"""Binary-conditioned AE + reconstruction-MSE anomaly readout for VRAVEN.

The AE's deployed detector objective is reconstruction MSE vs a threshold
(see Research_Paper process-injection autoencoders). VRAVEN needs a scalar /
class-logit target; this module exposes that same MSE as binary logits
(Benign vs Anomaly). It is an explicit anomaly-score selector, not a trained
decision head.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

VRAVEN_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path(
    os.environ.get(
        "VRAVEN_AE_DATA_ROOT",
        str(VRAVEN_ROOT / "DATASETS" / "process_injection_autoencoders"),
    )
).expanduser()
CHARMAP_CSV = PAPER_ROOT / "data" / "processed" / "charmap-ae-numeric.csv"
FOCUS_BINARY = "charmap_x64"


@dataclass(frozen=True)
class AutoencoderArchitecture:
    feature_dim: int
    binary_count: int
    embedding_dim: int
    encoder_hidden_sizes: tuple[int, ...]
    latent_dim: int
    decoder_hidden_sizes: tuple[int, ...]


def _mlp(input_size: int, hidden_sizes: Sequence[int], output_size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_size
    for size in hidden_sizes:
        layers.extend((nn.Linear(current, size), nn.ReLU()))
        current = size
    layers.append(nn.Linear(current, output_size))
    return nn.Sequential(*layers)


class BinaryConditionedAutoencoder(nn.Module):
    """Reconstruct features while conditioning encoder/decoder on binary ID."""

    def __init__(self, architecture: AutoencoderArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.binary_embedding = nn.Embedding(
            architecture.binary_count, architecture.embedding_dim
        )
        self.encoder = _mlp(
            architecture.feature_dim + architecture.embedding_dim,
            architecture.encoder_hidden_sizes,
            architecture.latent_dim,
        )
        self.decoder = _mlp(
            architecture.latent_dim + architecture.embedding_dim,
            architecture.decoder_hidden_sizes,
            architecture.feature_dim,
        )

    def forward(self, features: Tensor, binary_ids: Tensor) -> Tensor:
        condition = self.binary_embedding(binary_ids)
        latent = self.encoder(torch.cat((features, condition), dim=1))
        return self.decoder(torch.cat((latent, condition), dim=1))


class ReconstructionAnomalyDetector(nn.Module):
    """Expose AE reconstruction MSE as Benign/Anomaly logits for VRAVEN."""

    def __init__(
        self,
        autoencoder: BinaryConditionedAutoencoder,
        *,
        threshold: float,
        binary_id: int,
    ) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.autoencoder = autoencoder
        self.threshold = float(threshold)
        self._log_threshold = math.log1p(self.threshold)
        self.register_buffer("_binary_id", torch.tensor(int(binary_id), dtype=torch.long))

    def reconstruction_mse(self, features: Tensor) -> Tensor:
        ids = self._binary_id.expand(features.shape[0])
        recon = self.autoencoder(features, ids)
        return torch.mean((features - recon) ** 2, dim=-1)

    def forward(self, features: Tensor) -> Tensor:
        score = torch.log1p(self.reconstruction_mse(features))
        pivot = features.new_tensor(self._log_threshold)
        return torch.stack((pivot - score, score - pivot), dim=-1)


def _impute_scale(
    x: np.ndarray, col_median: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    inds = np.where(np.isnan(out))
    out[inds] = np.take(col_median, inds[1])
    scale_safe = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return ((out - center) / scale_safe).astype(np.float32)


def _read_feature_rows(
    path: Path, feature_names: list[str]
) -> tuple[np.ndarray, list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            value = row.get(name, "")
            x[i, j] = np.nan if value in ("", None) else float(value)
    return x, rows


def _feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "Memory footprint": [],
        "Executable / RWX regions": [],
        "Threads / modules / PEB": [],
        "Network endpoints": [],
        "Capability flags": [],
    }
    for idx, name in enumerate(feature_names):
        if name.startswith("cap_"):
            groups["Capability flags"].append(idx)
        elif any(
            token in name
            for token in (
                "tcp",
                "udp",
                "rdp",
                "smb",
                "network",
            )
        ):
            groups["Network endpoints"].append(idx)
        elif any(
            token in name
            for token in (
                "thread",
                "module",
                "peb",
                "image_path",
            )
        ):
            groups["Threads / modules / PEB"].append(idx)
        elif any(
            token in name
            for token in (
                "executable",
                "rwx",
                "wx_",
                "rx_private",
                "region",
                "mapped",
                "committed",
                "priv_exec",
            )
        ):
            groups["Executable / RWX regions"].append(idx)
        else:
            groups["Memory footprint"].append(idx)
    return {key: value for key, value in groups.items() if value}


def load_ae_anomaly_bundle(
    model_dir: Path,
    *,
    max_per_class: int = 32,
    focus_binary: str = FOCUS_BINARY,
) -> dict[str, Any]:
    """Load AE package + charmap injection rows into a VRAVEN-ready detector."""

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    scaler = np.load(model_dir / "scaler.npz")

    arch_raw = dict(payload["architecture"])
    for key in ("encoder_hidden_sizes", "decoder_hidden_sizes"):
        arch_raw[key] = tuple(arch_raw[key])
    architecture = AutoencoderArchitecture(**arch_raw)
    autoencoder = BinaryConditionedAutoencoder(architecture)
    autoencoder.load_state_dict(payload["state_dict"])
    autoencoder.eval()

    feature_names = [str(v) for v in meta["feature_names"]]
    vocab = [str(v) for v in meta["binary_vocab"]]
    if focus_binary not in vocab:
        raise KeyError(f"{focus_binary!r} missing from {model_dir.name} binary vocab")
    binary_index = vocab.index(focus_binary)
    threshold = float(meta.get("thresholds", {}).get(focus_binary, meta["global_threshold"]))

    matrix_path = Path(meta["matrix"])
    if not matrix_path.is_file():
        raise FileNotFoundError(f"AE matrix missing: {matrix_path}")
    if not CHARMAP_CSV.is_file():
        raise FileNotFoundError(f"Charmap injection matrix missing: {CHARMAP_CSV}")

    center = np.asarray(scaler["center_"], dtype=np.float64)
    scale = np.asarray(scaler["scale_"], dtype=np.float64)
    col_median = np.asarray(scaler["col_median"], dtype=np.float64)

    benign_raw, benign_rows = _read_feature_rows(matrix_path, feature_names)
    benign_mask = np.array(
        [str(row.get("binary_id", "")) == focus_binary for row in benign_rows],
        dtype=bool,
    )
    if not benign_mask.any():
        raise RuntimeError(f"No benign rows for {focus_binary} in {matrix_path}")
    benign_scaled = _impute_scale(benign_raw[benign_mask], col_median, center, scale)

    inject_raw, inject_rows = _read_feature_rows(CHARMAP_CSV, feature_names)
    inject_labels = np.array(
        [int(float(row.get("label_malware", "0") or 0)) for row in inject_rows],
        dtype=np.int64,
    )
    anomaly_mask = inject_labels == 1
    if not anomaly_mask.any():
        raise RuntimeError(f"No anomaly rows in {CHARMAP_CSV}")
    anomaly_scaled = _impute_scale(inject_raw[anomaly_mask], col_median, center, scale)

    rng = np.random.default_rng(24032026)
    benign_idx = rng.choice(
        benign_scaled.shape[0],
        size=min(max_per_class, benign_scaled.shape[0]),
        replace=False,
    )
    anomaly_idx = rng.choice(
        anomaly_scaled.shape[0],
        size=min(max_per_class, anomaly_scaled.shape[0]),
        replace=False,
    )
    x = np.concatenate([benign_scaled[benign_idx], anomaly_scaled[anomaly_idx]], axis=0)
    y = np.concatenate(
        [
            np.zeros(len(benign_idx), dtype=np.int64),
            np.ones(len(anomaly_idx), dtype=np.int64),
        ]
    )

    detector = ReconstructionAnomalyDetector(
        autoencoder,
        threshold=threshold,
        binary_id=binary_index,
    ).eval()

    return {
        "model": detector,
        "x": torch.from_numpy(x),
        "y": y,
        "feature_names": feature_names,
        "class_names": {0: "Benign", 1: "Anomaly"},
        "input_groups": _feature_groups(feature_names),
        "meta": meta,
        "focus_binary": focus_binary,
        "threshold": threshold,
        "matrix_path": str(matrix_path),
        "injection_matrix": str(CHARMAP_CSV),
    }
