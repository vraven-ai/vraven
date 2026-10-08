#!/usr/bin/env python3
"""Train diverse synthetic probe classifiers into Models/ for VRAVEN measurement.

Each package contains:
  model.pt          final weights + architecture metadata
  meta.json         task / metrics / feature groups
  reference.npz     held-out reference batch + labels
  checkpoints/      ordered epoch checkpoints for VRAVEN Watch

Domains: phishing URL, medical triage, network IDS, plus a CNN phishing twin
for Compare. No AE wrappers — these are native logit classifiers.
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
from probe_models import ProbeCNN1D, ProbeMLP  # noqa: E402

MODELS = ROOT / "Models"
SEED = 17


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_phishing(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Planted URL/email phishing score with distractors."""

    g = _rng(seed)
    names = [
        "url_length",
        "dot_count",
        "hyphen_count",
        "at_symbol",
        "https_token",
        "ip_in_host",
        "suspicious_tld",
        "brand_lookalike",
        "subdomain_depth",
        "query_entropy",
        "redirect_count",
        "punycode",
        "shortener",
        "login_keyword",
        "verify_keyword",
        "attachment_flag",
        "sender_domain_mismatch",
        "reply_to_mismatch",
        "urgency_lexicon",
        "money_lexicon",
        "html_form_count",
        "external_script",
        "homoglyph_score",
        "age_days_log",
    ]
    x = g.normal(0.0, 1.0, size=(n, len(names))).astype(np.float32)
    # Causal planted directions.
    score = (
        1.4 * x[:, names.index("ip_in_host")]
        + 1.2 * x[:, names.index("brand_lookalike")]
        + 1.1 * x[:, names.index("login_keyword")]
        + 0.9 * x[:, names.index("urgency_lexicon")]
        + 0.8 * x[:, names.index("sender_domain_mismatch")]
        - 1.0 * x[:, names.index("https_token")]
        - 0.7 * x[:, names.index("age_days_log")]
        + 0.15 * g.normal(size=n)
    )
    y = (score > 0).astype(np.int64)
    groups = {
        "Host / URL structure": list(range(0, 12)),
        "Content / lure language": list(range(12, 20)),
        "Page / delivery artefacts": list(range(20, 24)),
    }
    return x, y, names, groups


def make_medical(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Planted clinical deterioration risk with distractors."""

    g = _rng(seed)
    names = [
        "age",
        "heart_rate",
        "resp_rate",
        "systolic_bp",
        "spo2",
        "temperature",
        "wbc",
        "lactate",
        "creatinine",
        "glucose",
        "bmi",
        "comorbidity_count",
        "prior_admissions",
        "oxygen_support",
        "antibiotics_flag",
        "lab_missingness",
        "night_admit",
        "ward_transfer_count",
        "news2_proxy",
        "clinician_override_noise",
    ]
    x = g.normal(0.0, 1.0, size=(n, len(names))).astype(np.float32)
    score = (
        1.3 * x[:, names.index("lactate")]
        + 1.1 * x[:, names.index("resp_rate")]
        + 1.0 * x[:, names.index("news2_proxy")]
        + 0.9 * x[:, names.index("oxygen_support")]
        - 1.2 * x[:, names.index("spo2")]
        - 0.8 * x[:, names.index("systolic_bp")]
        + 0.5 * x[:, names.index("comorbidity_count")]
        + 0.2 * g.normal(size=n)
    )
    y = (score > 0).astype(np.int64)
    groups = {
        "Vitals": [0, 1, 2, 3, 4, 5],
        "Labs": [6, 7, 8, 9],
        "Context / process": list(range(10, 20)),
    }
    return x, y, names, groups


def make_network_ids(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Planted network intrusion score with distractors."""

    g = _rng(seed)
    names = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "packet_rate",
        "byte_ratio",
        "syn_count",
        "fin_count",
        "rst_count",
        "failed_logins",
        "same_srv_rate",
        "diff_srv_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "port_scan_score",
        "rare_port",
        "payload_entropy",
        "dns_nxdomain_rate",
        "http_error_rate",
        "tls_ja3_novelty",
        "beacon_periodicity",
        "offhours_fraction",
        "geo_anomaly",
        "asn_reputation",
        "process_parent_rare",
        "unsigned_binary",
        "lateral_hop_hint",
        "smb_enum_score",
        "rdp_brute_score",
        "exfil_burst",
        "dns_tunnel_hint",
        "idle_ratio",
        "protocol_mix_entropy",
    ]
    x = g.normal(0.0, 1.0, size=(n, len(names))).astype(np.float32)
    score = (
        1.4 * x[:, names.index("port_scan_score")]
        + 1.2 * x[:, names.index("beacon_periodicity")]
        + 1.1 * x[:, names.index("exfil_burst")]
        + 1.0 * x[:, names.index("dns_tunnel_hint")]
        + 0.9 * x[:, names.index("failed_logins")]
        + 0.8 * x[:, names.index("asn_reputation")]
        - 0.6 * x[:, names.index("idle_ratio")]
        + 0.2 * g.normal(size=n)
    )
    y = (score > 0).astype(np.int64)
    groups = {
        "Flow volume": list(range(0, 5)),
        "Protocol / host behaviour": list(range(5, 13)),
        "Threat indicators": list(range(13, 30)),
        "Residual": list(range(30, 32)),
    }
    return x, y, names, groups


TASKS = {
    "phishing_url_mlp": {
        "factory": make_phishing,
        "class_names": {0: "Legitimate", 1: "Phishing"},
        "arch": "mlp",
        "hidden": (64, 32),
        "task": "synthetic_phishing_url_email",
    },
    "phishing_url_cnn1d": {
        "factory": make_phishing,
        "class_names": {0: "Legitimate", 1: "Phishing"},
        "arch": "cnn1d",
        "hidden": (16, 32),
        "task": "synthetic_phishing_url_email_cnn",
        "seed_offset": 3,
    },
    "medical_triage_mlp": {
        "factory": make_medical,
        "class_names": {0: "Stable", 1: "Deterioration"},
        "arch": "mlp",
        "hidden": (48, 24),
        "task": "synthetic_medical_triage",
    },
    "network_ids_mlp": {
        "factory": make_network_ids,
        "class_names": {0: "Benign", 1: "Attack"},
        "arch": "mlp",
        "hidden": (64, 32),
        "task": "synthetic_network_ids",
    },
}


def _build(arch: str, input_dim: int, hidden: tuple[int, ...]) -> nn.Module:
    if arch == "mlp":
        return ProbeMLP(input_dim, hidden=hidden)
    if arch == "cnn1d":
        return ProbeCNN1D(input_dim, channels=hidden)  # type: ignore[arg-type]
    raise ValueError(arch)


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(-1)
    return float((pred == y).float().mean())


def train_one(name: str, spec: dict, *, n: int = 4000, epochs: int = 8) -> Path:
    out = MODELS / name
    ckpt_dir = out / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    seed = SEED + int(spec.get("seed_offset", 0))
    x, y, feature_names, groups = spec["factory"](n, seed)
    # Standardise.
    mean = x.mean(0, keepdims=True)
    std = x.std(0, keepdims=True) + 1e-6
    x = ((x - mean) / std).astype(np.float32)

    g = _rng(seed + 99)
    perm = g.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    torch.manual_seed(seed)
    model = _build(spec["arch"], len(feature_names), tuple(spec["hidden"]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=128,
        shuffle=True,
    )

    history = []
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        train_acc = _accuracy(model, torch.from_numpy(x_train), torch.from_numpy(y_train))
        test_acc = _accuracy(model, x_test_t, y_test_t)
        row = {
            "epoch": epoch,
            "train_loss": total / len(x_train),
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        history.append(row)
        torch.save(
            {
                "model_state": model.state_dict(),
                "architecture": {
                    "kind": spec["arch"],
                    "input_dim": len(feature_names),
                    "hidden": list(spec["hidden"]),
                    "num_classes": 2,
                },
                "epoch": epoch,
            },
            ckpt_dir / f"epoch_{epoch:02d}.pt",
        )
        print(f"  {name} epoch {epoch}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

    # Balanced reference cohort for VRAVEN.
    ref_pos = np.flatnonzero(y_test == 1)[:48]
    ref_neg = np.flatnonzero(y_test == 0)[:48]
    ref_idx = np.concatenate([ref_neg, ref_pos])
    np.savez_compressed(
        out / "reference.npz",
        features=x_test[ref_idx],
        labels=y_test[ref_idx],
        feature_names=np.asarray(feature_names),
        mean=mean.reshape(-1),
        std=std.reshape(-1),
        all_test_features=x_test,
        all_test_labels=y_test,
    )

    payload = {
        "model_state": model.state_dict(),
        "architecture": {
            "kind": spec["arch"],
            "input_dim": len(feature_names),
            "hidden": list(spec["hidden"]),
            "num_classes": 2,
        },
        "feature_names": feature_names,
        "class_names": [spec["class_names"][0], spec["class_names"][1]],
        "input_groups": groups,
        "scaler": {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()},
        "task": spec["task"],
        "history": history,
        "seed": seed,
    }
    torch.save(payload, out / "model.pt")
    meta = {
        "name": name,
        "task": spec["task"],
        "architecture": payload["architecture"],
        "class_names": payload["class_names"],
        "feature_names": feature_names,
        "input_groups": groups,
        "history": history,
        "final_test_acc": history[-1]["test_acc"],
        "checkpoints": sorted(p.name for p in ckpt_dir.glob("epoch_*.pt")),
        "artifacts": {
            "model": str(out / "model.pt"),
            "reference": str(out / "reference.npz"),
            "checkpoints": str(ckpt_dir),
        },
        "note": "Synthetic planted-mechanism probe for measuring VRAVEN, not a production detector.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, spec in TASKS.items():
        print(f"\n=== Training {name} ===")
        path = train_one(name, spec)
        print(f"  saved → {path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
