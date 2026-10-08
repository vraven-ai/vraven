#!/usr/bin/env python3
"""Train synthetic probes with architectures not used in the first probe set.

New algorithms / modalities:
  image_pattern_cnn2d      classic 2D CNN on planted visual patterns
  image_pattern_diffusion  noise-conditioned diffusion-style UNet classifier
  image_pattern_vit        tiny Vision Transformer
  sensor_transformer       token Transformer on synthetic sensor sequences

All emit class logits + reference.npz + epoch checkpoints for VRAVEN Watch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_models import build_probe  # noqa: E402

MODELS = ROOT / "Models"
SEED = 41
IMAGE_SIZE = 32


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_image_patterns(n: int, seed: int, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Class 0: vertical bars; class 1: centered blob — plus distractor noise."""

    g = _rng(seed)
    images = g.normal(0.0, 0.15, size=(n, 1, size, size)).astype(np.float32)
    y = g.integers(0, 2, size=n).astype(np.int64)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size // 2
    for i in range(n):
        if y[i] == 0:
            # Vertical bar cue (planted).
            col = int(g.integers(size // 4, 3 * size // 4))
            images[i, 0, :, col - 1 : col + 2] += 1.4
            # Horizontal distractor.
            row = int(g.integers(2, size - 2))
            images[i, 0, row, :] += 0.25
        else:
            # Blob cue (planted).
            radius = float(g.uniform(4.0, 8.0))
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
            images[i, 0][mask] += 1.5
            # Corner distractor.
            images[i, 0, 0:3, 0:3] += 0.35
    return images, y


def make_sensor_sequence(n: int, seed: int, seq_len: int = 48) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Planted multi-sensor anomaly sequence for Transformer probe."""

    g = _rng(seed)
    names = [f"t{t:02d}" for t in range(seq_len)]
    x = g.normal(0.0, 1.0, size=(n, seq_len)).astype(np.float32)
    # Causal windows: early spike + late oscillation.
    score = (
        1.3 * x[:, 8:12].mean(axis=1)
        + 1.1 * x[:, 30:36].mean(axis=1)
        - 0.8 * x[:, 20:24].mean(axis=1)
        + 0.2 * g.normal(size=n)
    )
    y = (score > 0).astype(np.int64)
    return x, y, names


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, noisy: bool = False) -> float:
    model.eval()
    with torch.no_grad():
        if noisy and hasattr(model, "noise_conditioned") and model.noise_conditioned:
            t = torch.rand(x.shape[0], 1) * 0.4
            noise = torch.randn_like(x) * t.view(-1, 1, 1, 1)
            logits = model(x + noise, t)
        else:
            logits = model(x)
        pred = logits.argmax(-1)
    return float((pred == y).float().mean())


def _save_package(
    *,
    name: str,
    model: nn.Module,
    architecture: dict,
    class_names: list[str],
    history: list[dict],
    reference: dict[str, np.ndarray],
    input_groups: dict | None,
    feature_names: list[str] | None,
    extra_meta: dict | None = None,
) -> Path:
    out = MODELS / name
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Move existing epoch files written during train — caller saves checkpoints.
    payload = {
        "model_state": model.state_dict(),
        "architecture": architecture,
        "class_names": class_names,
        "feature_names": feature_names or [],
        "input_groups": input_groups or {},
        "history": history,
        "task": architecture.get("task", name),
        "seed": architecture.get("seed", SEED),
    }
    torch.save(payload, out / "model.pt")
    np.savez_compressed(out / "reference.npz", **reference)
    meta = {
        "name": name,
        "task": payload["task"],
        "architecture": architecture,
        "class_names": class_names,
        "feature_names": feature_names or [],
        "input_groups": input_groups or {},
        "history": history,
        "final_test_acc": history[-1]["test_acc"] if history else None,
        "checkpoints": sorted(p.name for p in ckpt_dir.glob("epoch_*.pt")),
        "note": "Synthetic planted-mechanism probe for measuring VRAVEN (novel architecture set).",
        **(extra_meta or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def train_image_family() -> None:
    images, labels = make_image_patterns(3600, SEED)
    g = _rng(SEED + 7)
    perm = g.permutation(len(labels))
    split = int(0.8 * len(labels))
    train_idx, test_idx = perm[:split], perm[split:]
    x_train, y_train = images[train_idx], labels[train_idx]
    x_test, y_test = images[test_idx], labels[test_idx]

    # Pixel quadrant groups for DarkSide shortcuts (flattened NCHW).
    h = w = IMAGE_SIZE
    coords = [(i, j) for i in range(h) for j in range(w)]
    flat_index = {(i, j): i * w + j for i, j in coords}
    groups = {
        "top_left": [flat_index[(i, j)] for i in range(h // 2) for j in range(w // 2)],
        "top_right": [flat_index[(i, j)] for i in range(h // 2) for j in range(w // 2, w)],
        "bottom_left": [flat_index[(i, j)] for i in range(h // 2, h) for j in range(w // 2)],
        "bottom_right": [flat_index[(i, j)] for i in range(h // 2, h) for j in range(w // 2, w)],
    }
    feature_names = [f"px_{i}_{j}" for i in range(h) for j in range(w)]

    specs = [
        (
            "image_pattern_cnn2d",
            {
                "kind": "cnn2d",
                "in_channels": 1,
                "image_size": IMAGE_SIZE,
                "channels": [16, 32, 64],
                "num_classes": 2,
                "task": "synthetic_image_pattern_cnn2d",
                "seed": SEED,
            },
            False,
        ),
        (
            "image_pattern_diffusion",
            {
                "kind": "diffusion_unet",
                "in_channels": 1,
                "image_size": IMAGE_SIZE,
                "base_channels": 32,
                "num_classes": 2,
                "noise_conditioned": True,
                "task": "synthetic_image_diffusion_unet_classifier",
                "seed": SEED + 1,
            },
            True,
        ),
        (
            "image_pattern_vit",
            {
                "kind": "vit_tiny",
                "image_size": IMAGE_SIZE,
                "patch_size": 8,
                "in_channels": 1,
                "d_model": 64,
                "nhead": 4,
                "layers": 2,
                "num_classes": 2,
                "task": "synthetic_image_vit_tiny",
                "seed": SEED + 2,
            },
            False,
        ),
    ]

    class_names = ["vertical_bar", "centered_blob"]
    for name, arch, noisy in specs:
        print(f"\n=== Training {name} ===")
        torch.manual_seed(int(arch["seed"]))
        build_kwargs = {k: v for k, v in arch.items() if k not in {"kind", "task", "seed"}}
        # tuple-ify channel lists where needed
        if "channels" in build_kwargs:
            build_kwargs["channels"] = tuple(build_kwargs["channels"])
        model = build_probe(arch["kind"], **build_kwargs)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=64,
            shuffle=True,
        )
        out = MODELS / name
        ckpt_dir = out / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
        history = []
        x_test_t = torch.from_numpy(x_test)
        y_test_t = torch.from_numpy(y_test)
        epochs = 8
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                if noisy:
                    t = torch.rand(xb.shape[0], 1) * 0.5
                    noise = torch.randn_like(xb) * t.view(-1, 1, 1, 1)
                    logits = model(xb + noise, t)
                else:
                    logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(xb)
            train_acc = _accuracy(
                model, torch.from_numpy(x_train[:512]), torch.from_numpy(y_train[:512]), noisy=noisy
            )
            test_acc = _accuracy(model, x_test_t, y_test_t, noisy=noisy)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / len(x_train),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
            )
            torch.save(
                {"model_state": model.state_dict(), "architecture": arch, "epoch": epoch},
                ckpt_dir / f"epoch_{epoch:02d}.pt",
            )
            print(f"  epoch {epoch}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

        # Balanced reference cohort.
        ref_pos = np.flatnonzero(y_test == 1)[:32]
        ref_neg = np.flatnonzero(y_test == 0)[:32]
        ref_idx = np.concatenate([ref_neg, ref_pos])
        reference = {
            "images": x_test[ref_idx],
            "labels": y_test[ref_idx],
            "all_test_images": x_test,
            "all_test_labels": y_test,
            "feature_names": np.asarray(feature_names),
        }
        path = _save_package(
            name=name,
            model=model,
            architecture=arch,
            class_names=class_names,
            history=history,
            reference=reference,
            input_groups=groups,
            feature_names=feature_names,
            extra_meta={
                "modality": "image",
                "image_size": IMAGE_SIZE,
                "input_domain": {
                    "lower": None,
                    "upper": None,
                    "units": "synthetic intensity units",
                    "preprocessing": "none",
                    "semantic_domain": (
                        "real-valued 32x32 single-channel arrays generated by Gaussian "
                        "noise plus planted additive visual patterns"
                    ),
                    "source": "scripts/train_vraven_novel_probes.py:make_image_patterns",
                    "declared": True,
                },
            },
        )
        print(f"  saved → {path}")


def train_sensor_transformer() -> None:
    name = "sensor_transformer"
    print(f"\n=== Training {name} ===")
    x, y, feature_names = make_sensor_sequence(4000, SEED + 11, seq_len=48)
    mean = x.mean(0, keepdims=True)
    std = x.std(0, keepdims=True) + 1e-6
    x = ((x - mean) / std).astype(np.float32)
    g = _rng(SEED + 12)
    perm = g.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = perm[:split], perm[split:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    groups = {
        "early_window": list(range(0, 16)),
        "mid_window": list(range(16, 32)),
        "late_window": list(range(32, 48)),
    }
    arch = {
        "kind": "transformer",
        "seq_len": 48,
        "d_model": 48,
        "nhead": 4,
        "layers": 2,
        "num_classes": 2,
        "task": "synthetic_sensor_transformer",
        "seed": SEED + 11,
    }
    torch.manual_seed(arch["seed"])
    model = build_probe(
        "transformer",
        seq_len=48,
        d_model=48,
        nhead=4,
        layers=2,
        num_classes=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=128,
        shuffle=True,
    )
    out = MODELS / name
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    history = []
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)
    for epoch in range(1, 9):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
        train_acc = _accuracy(model, torch.from_numpy(x_train), torch.from_numpy(y_train))
        test_acc = _accuracy(model, x_test_t, y_test_t)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(x_train),
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        torch.save(
            {"model_state": model.state_dict(), "architecture": arch, "epoch": epoch},
            ckpt_dir / f"epoch_{epoch:02d}.pt",
        )
        print(f"  epoch {epoch}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

    ref_pos = np.flatnonzero(y_test == 1)[:48]
    ref_neg = np.flatnonzero(y_test == 0)[:48]
    ref_idx = np.concatenate([ref_neg, ref_pos])
    reference = {
        "features": x_test[ref_idx],
        "labels": y_test[ref_idx],
        "all_test_features": x_test,
        "all_test_labels": y_test,
        "feature_names": np.asarray(feature_names),
        "mean": mean.reshape(-1),
        "std": std.reshape(-1),
    }
    path = _save_package(
        name=name,
        model=model,
        architecture=arch,
        class_names=["normal", "anomaly"],
        history=history,
        reference=reference,
        input_groups=groups,
        feature_names=feature_names,
        extra_meta={"modality": "sequence"},
    )
    print(f"  saved → {path}")


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    train_image_family()
    train_sensor_transformer()
    print("\nDone novel probes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
