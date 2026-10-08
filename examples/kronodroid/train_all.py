#!/usr/bin/env python3
"""Train MLP / CNN1D / ResNet1D / BiGRU on Kronodroid with a stratified 70/30 split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import MODEL_BUILDERS

ROOT = Path(__file__).resolve().parent
VRAVEN_ROOT = ROOT.parents[1]
DEFAULT_CSV = VRAVEN_ROOT / "KR" / "kronodroid_combined.csv"
DEFAULT_OUT = VRAVEN_ROOT / "Models" / "kronodroid"

# Metadata / leakage-prone columns (not used as model inputs)
DROP_COLUMNS = {
    "Package",
    "sha256",
    "EarliestModDate",
    "HighestModDate",
    "MalFamily",
    "Malware",
    "Detection_Ratio",
    "Scanners",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_BUILDERS),
        choices=sorted(MODEL_BUILDERS),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_xy(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    print(f"Loading {csv_path} …", flush=True)
    frame = pd.read_csv(csv_path, low_memory=False)
    if "Malware" not in frame.columns:
        raise KeyError("Expected binary label column 'Malware'")
    y = frame["Malware"].astype(np.int64).to_numpy()
    feature_cols = [c for c in frame.columns if c not in DROP_COLUMNS]
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_arr = x.to_numpy(dtype=np.float32)
    print(
        f"  samples={len(y):,}  features={len(feature_cols)}  "
        f"malware={int((y == 1).sum()):,}  benign={int((y == 0).sum()):,}",
        flush=True,
    )
    return x_arr, y, feature_cols


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for xb, yb in loader:
        logits = model(xb.to(device))
        logits_all.append(logits.cpu().numpy())
        y_all.append(yb.numpy())
    logits_np = np.concatenate(logits_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    probs = torch.softmax(torch.tensor(logits_np), dim=-1).numpy()[:, 1]
    y_pred = logits_np.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["benign", "malware"], digits=4, output_dict=True
        ),
    }
    return metrics


def train_one(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, dict, list[dict]]:
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_state = None
    best_auc = -1.0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
        scheduler.step()
        train_loss = running / max(n, 1)
        test_metrics = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        }
        history.append(row)
        print(
            f"[{name}] epoch {epoch:02d}/{epochs}  "
            f"loss={train_loss:.4f}  acc={test_metrics['accuracy']:.4f}  "
            f"f1={test_metrics['f1']:.4f}  auc={test_metrics['roc_auc']:.4f}",
            flush=True,
        )
        if test_metrics["roc_auc"] > best_auc:
            best_auc = test_metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, test_loader, device)
    final["train_seconds"] = time.perf_counter() - started
    final["best_test_roc_auc"] = best_auc
    return model, final, history


def main() -> None:
    args = parse_args()
    device = pick_device(args.cpu)
    print(f"device={device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, feature_names = load_xy(args.csv)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "split.npz",
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.asarray(feature_names),
    )
    # scaler params
    np.savez(
        out / "scaler.npz",
        mean_=scaler.mean_.astype(np.float64),
        scale_=scaler.scale_.astype(np.float64),
        n_features_in_=np.asarray([scaler.n_features_in_]),
    )

    summary: dict = {
        "dataset": str(args.csv),
        "device": str(device),
        "split": {"train": 1.0 - args.test_size, "test": args.test_size, "seed": args.seed},
        "samples": {
            "total": int(len(y)),
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "train_malware": int((y_train == 1).sum()),
            "test_malware": int((y_test == 1).sum()),
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "dropped_columns": sorted(DROP_COLUMNS),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "models": {},
    }

    for name in args.models:
        print(f"\n=== Training {name} ===", flush=True)
        model = MODEL_BUILDERS[name](x_train.shape[1])
        trained, metrics, history = train_one(
            name,
            model,
            train_loader,
            test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
        model_dir = out / name
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": trained.state_dict(),
                "architecture": name,
                "input_dim": int(x_train.shape[1]),
                "num_classes": 2,
            },
            model_dir / "model.pt",
        )
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        meta = {
            "architecture": name,
            "input_dim": int(x_train.shape[1]),
            "labels": {"benign": 0, "malware": 1},
            "feature_names": feature_names,
            "test_metrics": {
                k: metrics[k]
                for k in ("accuracy", "f1", "roc_auc", "average_precision", "confusion_matrix")
            },
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary["models"][name] = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "confusion_matrix": metrics["confusion_matrix"],
            "train_seconds": metrics["train_seconds"],
            "path": str(model_dir / "model.pt"),
        }
        print(
            f"[{name}] DONE  acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}  "
            f"time={metrics['train_seconds']:.1f}s",
            flush=True,
        )

    (out / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Summary ===", flush=True)
    for name, row in summary["models"].items():
        print(
            f"  {name:10s}  acc={row['accuracy']:.4f}  f1={row['f1']:.4f}  "
            f"auc={row['roc_auc']:.4f}",
            flush=True,
        )
    print(f"Saved under {out}", flush=True)


if __name__ == "__main__":
    main()
