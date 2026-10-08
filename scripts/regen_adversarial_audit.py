#!/usr/bin/env python3
"""Regenerate Reverse adversarial_audit.json/html for existing model outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "examples" / "kronodroid"))
sys.path.insert(0, str(ROOT / "examples" / "windows_malware_mlp"))

from ae_models import load_ae_anomaly_bundle
from model import MalwareMLP as LolbinMLP
from run_vraven_on_models import (
    MODELS,
    OUT,
    _build_from_arch,
    _focus_from_batch,
    _save_adversarial_audit,
)

from models import MODEL_BUILDERS

AE_NAMES = ("windows10_full_benign_ae", "win2016_dc")


def _load_model_bundle(rel: str):
    if rel in AE_NAMES:
        bundle = load_ae_anomaly_bundle(MODELS / rel, max_per_class=24)
        return bundle["model"], bundle["x"], bundle["y"], list(bundle.get("feature_names") or [])
    if rel == "winlolbin_mlp":
        model_dir = MODELS / rel
        payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        arch = payload["architecture"]
        model = LolbinMLP(
            input_dim=int(arch["input_dim"]),
            hidden_sizes=tuple(arch["hidden_sizes"]),
            num_classes=int(arch["num_classes"]),
        ).eval()
        model.load_state_dict(payload["model_state"])
        sample = np.load(model_dir / "explain_sample.npz")
        features = np.asarray(sample["features_scaled"], dtype=np.float32)
        labels = np.asarray(sample["labels"], dtype=np.int64)
        idx0 = np.flatnonzero(labels == 0)[:32]
        idx1 = np.flatnonzero(labels == 1)[:32]
        keep = np.concatenate([idx0, idx1])
        x = torch.from_numpy(features[keep])
        y = labels[keep]
        names = list(payload.get("feature_names") or [])
        return model, x, y, names
    if rel.startswith("kronodroid/"):
        arch = rel.split("/", 1)[1]
        ckpt = torch.load(
            MODELS / "kronodroid" / arch / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        builder = MODEL_BUILDERS[arch]
        model = builder(int(ckpt["input_dim"])).eval()
        model.load_state_dict(ckpt["model_state"])
        split = np.load(MODELS / "kronodroid" / "split.npz", allow_pickle=True)
        x_test = np.asarray(split["x_test"], dtype=np.float32)
        y_test = np.asarray(split["y_test"], dtype=np.int64)
        benign = np.where(y_test == 0)[0][:32]
        malware = np.where(y_test == 1)[0][:32]
        keep = np.concatenate([benign, malware])
        x = torch.from_numpy(x_test[keep])
        y = y_test[keep]
        return model, x, y, []
    model_dir = MODELS / rel
    payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    arch = dict(payload["architecture"])
    if arch.get("kind") == "cnn1d" and "hidden" in arch and "channels" not in arch:
        arch["channels"] = arch.pop("hidden")
    model = _build_from_arch(arch).eval()
    model.load_state_dict(payload["model_state"])
    ref = np.load(model_dir / "reference.npz", allow_pickle=True)
    if "images" in ref.files:
        x = torch.from_numpy(np.asarray(ref["images"], dtype=np.float32))
    else:
        x = torch.from_numpy(np.asarray(ref["features"], dtype=np.float32))
    y = np.asarray(ref["labels"], dtype=np.int64)
    names = list(payload.get("feature_names") or [])
    return model, x, y, names


def regen_one(rel: str) -> None:
    dest = OUT / rel / "reverse"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {rel} ===")
    model, x, y, feature_names = _load_model_bundle(rel)
    try:
        focus, _ = _focus_from_batch(model, x, y)
    except Exception:
        pos = np.flatnonzero(y == 1)
        idx = int(pos[0]) if len(pos) else 0
        focus = x[idx : idx + 1]
    status = _save_adversarial_audit(
        model,
        focus,
        dest,
        model_name=rel,
        feature_names=feature_names,
        target=1,
        batch_x=x,
        batch_y=y,
        autoattack_epsilon=0.1,
        robust_max_samples=12,
    )
    print(
        f"  attacks={status['attacks']} "
        f"min_eps_linf={status['empirical_min_flip_eps_linf']} "
        f"AA_robust={status.get('autoattack_robust')} "
        f"rob_acc={status.get('robust_accuracy')} "
        f"vulnerable={status['most_vulnerable_concept']}"
    )


def main() -> int:
    # reverse/ lives at <model>/reverse — take the model-relative parent.
    targets = sorted(
        {
            path.parent.relative_to(OUT).as_posix()
            for path in OUT.rglob("reverse")
            if path.is_dir() and path.name == "reverse"
        }
    )
    if not targets:
        print("No reverse/ folders found under vraven_outputs")
        return 1
    runnable = []
    for rel in targets:
        if rel in AE_NAMES or rel == "winlolbin_mlp" or rel.startswith("kronodroid/") or (MODELS / rel / "model.pt").is_file():
            runnable.append(rel)
    if not runnable:
        print(f"Found reverse folders {targets} but no loadable model packages")
        return 1
    for rel in runnable:
        try:
            regen_one(rel)
        except Exception as error:  # noqa: BLE001
            print(f"  FAILED {rel}: {type(error).__name__}: {error}")
    print("\nRebuilding gallery…")
    from build_vraven_gallery import main as build_gallery

    build_gallery()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
