#!/usr/bin/env python3
"""Rebuild DarkSide circuit wiring + CONNECTION edge cards for existing outputs."""

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
from run_vraven_on_models import MODELS, OUT, _build_from_arch, _focus_from_batch, _linear_names

from models import MODEL_BUILDERS
from vraven.darkside import AnalysisResult, DarkSideSession
from vraven.darkside.serialisation import load_result

AE_NAMES = ("windows10_full_benign_ae", "win2016_dc")
KRONO = ("mlp", "cnn1d", "resnet1d", "bigru")


def _load_model_bundle(rel: str):
    if rel in AE_NAMES:
        bundle = load_ae_anomaly_bundle(MODELS / rel, max_per_class=24)
        return (
            bundle["model"],
            bundle["x"],
            bundle["y"],
            bundle["class_names"],
            _linear_names(bundle["model"]),
        )
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
        return model, x, y, {0: "Benign", 1: "Malware"}, _linear_names(model)
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
        return model, x, y, {0: "Benign", 1: "Malware"}, _linear_names(model)
    # probe
    model_dir = MODELS / rel
    payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    arch = dict(payload["architecture"])
    # Older cnn1d packages stored channels under "hidden".
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
    class_list = payload["class_names"]
    class_names = {0: class_list[0], 1: class_list[1]}
    return model, x, y, class_names, _linear_names(model)


def regen_one(rel: str) -> None:
    dark = OUT / rel / "darkside"
    circuit_path = dark / "decision_circuitry.json"
    if not circuit_path.is_file():
        print(f"skip {rel}: no decision_circuitry.json")
        return
    print(f"\n=== {rel} ===")
    model, x, y, class_names, layers = _load_model_bundle(rel)
    try:
        focus, _ = _focus_from_batch(model, x, y)
    except Exception:
        # Odd output layouts (some transformers): fall back to first positive label.
        pos = np.flatnonzero(y == 1)
        idx = int(pos[0]) if len(pos) else 0
        focus = x[idx : idx + 1]
    circuit = load_result(circuit_path)
    if not isinstance(circuit, AnalysisResult):
        circuit = AnalysisResult(**circuit) if isinstance(circuit, dict) else circuit
    session = DarkSideSession(model, focus, target=1)
    ref = x[: min(64, len(x))]
    wiring = session.circuit_wiring(
        circuit,
        enrich=True,
        reference_batch=ref,
        class_names=class_names,
        top_k_intervene=16,
        bootstrap_samples=min(32, int(ref.shape[0])),
    )
    wiring.save_json(dark / "circuit_wiring.json")
    wiring.figure(mode="interactive_dark").write_html(
        dark / "circuit_wiring_interactive_dark.html"
    )
    wiring.figure(mode="publication_light").write_html(
        dark / "circuit_wiring_publication_light.html"
    )
    print(
        f"  edges={wiring.metrics.get('edges')} cards={wiring.metrics.get('edge_cards')} "
        f"intervened={wiring.metrics.get('intervened_edges')}"
    )


def main() -> int:
    targets = []
    for path in sorted(OUT.rglob("decision_circuitry.json")):
        rel = str(path.parent.parent.relative_to(OUT))
        targets.append(rel)
    if not targets:
        print("No decision_circuitry.json found")
        return 1
    for rel in targets:
        try:
            regen_one(rel)
        except Exception as error:  # noqa: BLE001
            print(f"  FAILED {rel}: {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
