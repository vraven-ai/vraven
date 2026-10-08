#!/usr/bin/env python3
"""Train the 26-configuration, five-seed VRAVEN publication model suite.

The default run produces 130 independently trained final artefacts. Training is
validation-selected, capped at five epochs, resumable, and uses Apple MPS when
available. Test data are evaluated once after restoring the best validation epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "DATASETS"
DEFAULT_OUTPUT = ROOT / "Models" / "publication_suite"
sys.path.insert(0, str(ROOT / "scripts"))

from probe_models import build_probe  # noqa: E402
from publication_models import build_publication_model  # noqa: E402

TRAINING_SEEDS = (11, 29, 47, 71, 101)
DATASET_ORDER = (
    "kronodroid",
    "fashion_mnist",
    "uci_har",
    "sms_spam",
    "wdbc",
    "eeg_eye_state",
    "ai4i_predictive_maintenance",
    "dry_bean",
    "bank_marketing",
    "wine_quality",
)
ARCHITECTURES = {
    "kronodroid": ("residual_mlp", "feature_transformer"),
    "fashion_mnist": ("image_cnn", "vit_tiny"),
    "uci_har": ("sequence_cnn", "sequence_transformer"),
    "sms_spam": ("tfidf_cnn1d", "feature_transformer"),
    "wdbc": ("residual_mlp", "feature_transformer"),
    "eeg_eye_state": ("sequence_cnn", "sequence_transformer"),
    "ai4i_predictive_maintenance": ("residual_mlp", "feature_transformer"),
    "dry_bean": ("residual_mlp", "feature_transformer"),
    "bank_marketing": ("residual_mlp", "feature_transformer"),
    "wine_quality": ("residual_mlp", "feature_transformer"),
}
ROBUST_DATASETS = frozenset(("kronodroid", "fashion_mnist", "uci_har"))
QUALITY_TARGETS = {
    "kronodroid": 0.88,
    "fashion_mnist": 0.82,
    "uci_har": 0.84,
    "sms_spam": 0.88,
    "wdbc": 0.88,
    "eeg_eye_state": 0.62,
    "ai4i_predictive_maintenance": 0.55,
    "dry_bean": 0.85,
    "bank_marketing": 0.55,
    "wine_quality": 0.30,
}


@dataclass
class SuiteData:
    dataset: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    input_groups: dict[str, list[int]]
    preprocessing: dict[str, Any]
    input_domain: dict[str, Any]
    modality: str
    source_files: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--target-validation-accuracy",
        type=float,
        default=None,
        help="Stop a run once validation accuracy reaches this value (for example, 0.94).",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronise(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_val, test = train_test_split(
        indices,
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=seed + 1,
        stratify=labels[train_val],
    )
    return np.sort(train), np.sort(val), np.sort(test)


def encode_targets(values: pd.Series | np.ndarray) -> tuple[np.ndarray, list[str]]:
    array = np.asarray(values)
    classes = sorted(np.unique(array).tolist(), key=lambda value: str(value))
    mapping = {value: index for index, value in enumerate(classes)}
    encoded = np.asarray([mapping[value] for value in array], dtype=np.int64)
    return encoded, [str(value) for value in classes]


def vector_groups(size: int) -> dict[str, list[int]]:
    boundaries = np.linspace(0, size, 5, dtype=int)
    return {
        f"feature_quartile_{index + 1}": list(range(boundaries[index], boundaries[index + 1]))
        for index in range(4)
        if boundaries[index + 1] > boundaries[index]
    }


def image_groups(height: int, width: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for row_name, rows in (("top", range(0, height // 2)), ("bottom", range(height // 2, height))):
        for col_name, columns in (
            ("left", range(0, width // 2)),
            ("right", range(width // 2, width)),
        ):
            groups[f"{row_name}_{col_name}"] = [row * width + column for row in rows for column in columns]
    return groups


def sequence_groups(seq_len: int, channels: int, names: list[str] | None = None) -> dict[str, list[int]]:
    channel_names = names or [f"channel_{index}" for index in range(channels)]
    return {
        str(channel_names[channel]): [time_index * channels + channel for time_index in range(seq_len)]
        for channel in range(channels)
    }


def fit_tabular_transform(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    train_frame = frame.iloc[train_indices]

    numeric_median: dict[str, float] = {}
    numeric_mean: dict[str, float] = {}
    numeric_scale: dict[str, float] = {}
    for column in numeric_columns:
        train_numeric = pd.to_numeric(train_frame[column], errors="coerce")
        median = float(train_numeric.median())
        filled = train_numeric.fillna(median).to_numpy(dtype=np.float64)
        mean = float(filled.mean())
        scale = float(filled.std())
        numeric_median[column] = median
        numeric_mean[column] = mean
        numeric_scale[column] = scale if scale > 1e-8 else 1.0

    categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = train_frame[column].fillna("<MISSING>").astype(str)
        categories[column] = sorted(values.unique().tolist())

    feature_names = list(numeric_columns)
    for column in categorical_columns:
        feature_names.extend(f"{column}={category}" for category in categories[column])

    def transform(indices: np.ndarray) -> np.ndarray:
        subset = frame.iloc[indices]
        blocks: list[np.ndarray] = []
        if numeric_columns:
            numeric = np.column_stack(
                [
                    (
                        pd.to_numeric(subset[column], errors="coerce")
                        .fillna(numeric_median[column])
                        .to_numpy(dtype=np.float64)
                        - numeric_mean[column]
                    )
                    / numeric_scale[column]
                    for column in numeric_columns
                ]
            ).astype(np.float32)
            blocks.append(numeric)
        for column in categorical_columns:
            values = subset[column].fillna("<MISSING>").astype(str).to_numpy()
            block = np.zeros((len(indices), len(categories[column])), dtype=np.float32)
            positions = {value: index for index, value in enumerate(categories[column])}
            for row, value in enumerate(values):
                if value in positions:
                    block[row, positions[value]] = 1.0
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    preprocessing = {
        "kind": "train_only_numeric_standardisation_and_one_hot",
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_median": numeric_median,
        "numeric_mean": numeric_mean,
        "numeric_scale": numeric_scale,
        "categories": categories,
        "unknown_category_policy": "all-zero block",
    }
    return (
        transform(train_indices),
        transform(val_indices),
        transform(test_indices),
        feature_names,
        preprocessing,
    )


TABULAR_SCHEMAS = {
    "kronodroid": {
        "path": "kronodroid/training_ready/kronodroid_combined.csv",
        "target": "Malware",
        "drop": [
            "Package",
            "sha256",
            "EarliestModDate",
            "HighestModDate",
            "MalFamily",
            "Detection_Ratio",
            "Scanners",
        ],
    },
    "wdbc": {
        "path": "wdbc/training_ready/wdbc.csv",
        "target": "diagnosis",
        "drop": ["id"],
    },
    "ai4i_predictive_maintenance": {
        "path": "ai4i_predictive_maintenance/training_ready/ai4i2020.csv",
        "target": "Machine failure",
        "drop": ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
    },
    "dry_bean": {
        "path": "dry_bean/training_ready/dry_bean.csv",
        "target": "Class",
        "drop": [],
    },
    "bank_marketing": {
        "path": "bank_marketing/training_ready/bank_marketing.csv",
        "target": "y",
        "drop": ["duration"],
    },
    "wine_quality": {
        "path": "wine_quality/training_ready/wine_quality.csv",
        "target": "quality",
        "drop": [],
    },
}


def load_tabular(dataset: str, seed: int) -> SuiteData:
    schema = TABULAR_SCHEMAS[dataset]
    path = DATASETS / schema["path"]
    frame = pd.read_csv(path, low_memory=False)
    labels, class_names = encode_targets(frame[schema["target"]])
    train, val, test = stratified_indices(labels, seed)
    excluded = {schema["target"], *schema["drop"]}
    features = frame[[column for column in frame.columns if column not in excluded]].copy()
    x_train, x_val, x_test, feature_names, preprocessing = fit_tabular_transform(
        features,
        train,
        val,
        test,
    )
    preprocessing.update(
        {
            "target": schema["target"],
            "excluded_columns": sorted(excluded - {schema["target"]}),
            "split": "paired stratified 70/15/15",
            "split_seed": seed,
        }
    )
    return SuiteData(
        dataset=dataset,
        x_train=x_train,
        y_train=labels[train],
        x_val=x_val,
        y_val=labels[val],
        x_test=x_test,
        y_test=labels[test],
        train_ids=train.astype(str),
        val_ids=val.astype(str),
        test_ids=test.astype(str),
        class_names=class_names,
        feature_names=feature_names,
        input_groups=vector_groups(len(feature_names)),
        preprocessing=preprocessing,
        input_domain={
            "lower": None,
            "upper": None,
            "units": "training-fold standard deviations plus one-hot indicators",
            "preprocessing": "train-only imputation, standardisation and one-hot encoding",
            "semantic_domain": "mixed tabular domain; Lp stress is not a domain-valid mutation guarantee",
            "source": str(path.relative_to(ROOT)),
            "declared": True,
        },
        modality="vector",
        source_files=[str(path.relative_to(ROOT))],
    )


def load_fashion_mnist(seed: int) -> SuiteData:
    path = DATASETS / "fashion_mnist" / "training_ready" / "fashion_mnist.npz"
    source = np.load(path)
    full_train_x = source["x_train"].astype(np.float32)[:, None] / 255.0
    full_train_y = source["y_train"].astype(np.int64)
    x_test = source["x_test"].astype(np.float32)[:, None] / 255.0
    y_test = source["y_test"].astype(np.int64)
    indices = np.arange(len(full_train_y))
    train, val = train_test_split(
        indices,
        test_size=0.15,
        random_state=seed,
        stratify=full_train_y,
    )
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    feature_names = [f"pixel_{row}_{column}" for row in range(28) for column in range(28)]
    return SuiteData(
        dataset="fashion_mnist",
        x_train=full_train_x[train],
        y_train=full_train_y[train],
        x_val=full_train_x[val],
        y_val=full_train_y[val],
        x_test=x_test,
        y_test=y_test,
        train_ids=np.asarray([f"train:{index}" for index in train]),
        val_ids=np.asarray([f"train:{index}" for index in val]),
        test_ids=np.asarray([f"test:{index}" for index in range(len(y_test))]),
        class_names=class_names,
        feature_names=feature_names,
        input_groups=image_groups(28, 28),
        preprocessing={
            "kind": "fixed_uint8_to_unit_interval",
            "formula": "x_float = x_uint8 / 255",
            "validation_split": "stratified 15% of official training split",
            "split_seed": seed,
        },
        input_domain={
            "lower": 0.0,
            "upper": 1.0,
            "units": "normalised greyscale intensity",
            "preprocessing": "uint8 / 255",
            "semantic_domain": "28x28 single-channel Fashion-MNIST images",
            "source": str(path.relative_to(ROOT)),
            "declared": True,
        },
        modality="image",
        source_files=[str(path.relative_to(ROOT))],
    )


def load_uci_har(seed: int) -> SuiteData:
    path = DATASETS / "uci_har" / "training_ready" / "uci_har_inertial_signals.npz"
    source = np.load(path)
    all_train_x = source["x_train"].astype(np.float32)
    all_train_y = source["y_train"].astype(np.int64)
    subjects = source["subject_train"].astype(np.int64)
    unique_subjects = np.unique(subjects)
    generator = np.random.default_rng(seed)
    validation_subjects = generator.choice(
        unique_subjects,
        size=max(1, math.ceil(0.2 * len(unique_subjects))),
        replace=False,
    )
    val_mask = np.isin(subjects, validation_subjects)
    train_mask = ~val_mask
    x_train = all_train_x[train_mask]
    x_val = all_train_x[val_mask]
    x_test = source["x_test"].astype(np.float32)
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    scale = x_train.std(axis=(0, 1), keepdims=True)
    scale[scale < 1e-6] = 1.0
    x_train = ((x_train - mean) / scale).astype(np.float32)
    x_val = ((x_val - mean) / scale).astype(np.float32)
    x_test = ((x_test - mean) / scale).astype(np.float32)
    channel_names = [str(value) for value in source["channel_names"].tolist()]
    class_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    feature_names = [f"time_{time_index}:{channel}" for time_index in range(128) for channel in channel_names]
    return SuiteData(
        dataset="uci_har",
        x_train=x_train,
        y_train=all_train_y[train_mask],
        x_val=x_val,
        y_val=all_train_y[val_mask],
        x_test=x_test,
        y_test=source["y_test"].astype(np.int64),
        train_ids=np.asarray([f"official_train:{index}" for index in np.flatnonzero(train_mask)]),
        val_ids=np.asarray([f"official_train:{index}" for index in np.flatnonzero(val_mask)]),
        test_ids=np.asarray([f"official_test:{index}" for index in range(len(x_test))]),
        class_names=class_names,
        feature_names=feature_names,
        input_groups=sequence_groups(128, 9, channel_names),
        preprocessing={
            "kind": "training-subject_channel_standardisation",
            "channel_names": channel_names,
            "mean": mean.reshape(-1).tolist(),
            "scale": scale.reshape(-1).tolist(),
            "validation_subjects": validation_subjects.tolist(),
            "test_split": "official subject-disjoint test split",
        },
        input_domain={
            "lower": None,
            "upper": None,
            "units": "training-fold standard deviations",
            "preprocessing": "per-channel train-only standardisation",
            "semantic_domain": "128-step, 9-channel inertial windows",
            "source": str(path.relative_to(ROOT)),
            "declared": True,
        },
        modality="sequence",
        source_files=[str(path.relative_to(ROOT))],
    )


def load_sms_spam(seed: int) -> SuiteData:
    path = DATASETS / "sms_spam" / "training_ready" / "sms_spam.csv"
    frame = pd.read_csv(path)
    labels, class_names = encode_targets(frame["label"])
    train, val, test = stratified_indices(labels, seed)
    vectoriser = TfidfVectorizer(
        max_features=512,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectoriser.fit_transform(frame.iloc[train]["text"].fillna("")).toarray().astype(np.float32)
    x_val = vectoriser.transform(frame.iloc[val]["text"].fillna("")).toarray().astype(np.float32)
    x_test = vectoriser.transform(frame.iloc[test]["text"].fillna("")).toarray().astype(np.float32)
    feature_names = vectoriser.get_feature_names_out().tolist()
    return SuiteData(
        dataset="sms_spam",
        x_train=x_train,
        y_train=labels[train],
        x_val=x_val,
        y_val=labels[val],
        x_test=x_test,
        y_test=labels[test],
        train_ids=train.astype(str),
        val_ids=val.astype(str),
        test_ids=test.astype(str),
        class_names=class_names,
        feature_names=feature_names,
        input_groups=vector_groups(len(feature_names)),
        preprocessing={
            "kind": "train_only_tfidf",
            "max_features": 512,
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
            "vocabulary": vectoriser.vocabulary_,
            "idf": vectoriser.idf_.tolist(),
            "split": "paired stratified 70/15/15",
            "split_seed": seed,
        },
        input_domain={
            "lower": 0.0,
            "upper": 1.0,
            "units": "TF-IDF weight",
            "preprocessing": "train-only unigram/bigram TF-IDF",
            "semantic_domain": "fixed TF-IDF vector; Lp changes are not guaranteed valid text edits",
            "source": str(path.relative_to(ROOT)),
            "declared": True,
        },
        modality="vector",
        source_files=[str(path.relative_to(ROOT))],
    )


def make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    length: int = 32,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, len(features) - length + 1, stride)
    x = np.stack([features[start : start + length] for start in starts]).astype(np.float32)
    y = np.asarray([labels[start + length - 1] for start in starts], dtype=np.int64)
    window_ids = np.asarray([f"{ids[start]}..{ids[start + length - 1]}" for start in starts])
    return x, y, window_ids


def load_eeg_eye_state(seed: int) -> SuiteData:
    del seed
    path = DATASETS / "eeg_eye_state" / "training_ready" / "eeg_eye_state.csv"
    frame = pd.read_csv(path)
    labels = frame.pop("eyeDetection").to_numpy(dtype=np.int64)
    features = frame.to_numpy(dtype=np.float32)
    train_end = int(0.60 * len(frame))
    val_end = int(0.80 * len(frame))
    mean = features[:train_end].mean(axis=0, keepdims=True)
    scale = features[:train_end].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    features = ((features - mean) / scale).astype(np.float32)
    all_ids = np.arange(len(frame)).astype(str)
    x_train, y_train, train_ids = make_windows(features[:train_end], labels[:train_end], all_ids[:train_end])
    x_val, y_val, val_ids = make_windows(features[train_end:val_end], labels[train_end:val_end], all_ids[train_end:val_end])
    x_test, y_test, test_ids = make_windows(features[val_end:], labels[val_end:], all_ids[val_end:])
    channel_names = frame.columns.tolist()
    return SuiteData(
        dataset="eeg_eye_state",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        class_names=["eyes_open", "eyes_closed"],
        feature_names=[f"time_{time_index}:{channel}" for time_index in range(32) for channel in channel_names],
        input_groups=sequence_groups(32, len(channel_names), channel_names),
        preprocessing={
            "kind": "blocked_chronological_windows",
            "train_val_test": [0.60, 0.20, 0.20],
            "window_length": 32,
            "stride": 4,
            "label": "last point in window",
            "mean": mean.reshape(-1).tolist(),
            "scale": scale.reshape(-1).tolist(),
        },
        input_domain={
            "lower": None,
            "upper": None,
            "units": "training-block standard deviations",
            "preprocessing": "blocked split and train-only channel standardisation",
            "semantic_domain": "32-step, 14-channel EEG windows",
            "source": str(path.relative_to(ROOT)),
            "declared": True,
        },
        modality="sequence",
        source_files=[str(path.relative_to(ROOT))],
    )


DATA_LOADERS: dict[str, Callable[[int], SuiteData]] = {
    "kronodroid": lambda seed: load_tabular("kronodroid", seed),
    "fashion_mnist": load_fashion_mnist,
    "uci_har": load_uci_har,
    "sms_spam": load_sms_spam,
    "wdbc": lambda seed: load_tabular("wdbc", seed),
    "eeg_eye_state": load_eeg_eye_state,
    "ai4i_predictive_maintenance": lambda seed: load_tabular("ai4i_predictive_maintenance", seed),
    "dry_bean": lambda seed: load_tabular("dry_bean", seed),
    "bank_marketing": lambda seed: load_tabular("bank_marketing", seed),
    "wine_quality": lambda seed: load_tabular("wine_quality", seed),
}


def architecture_payload(alias: str, data: SuiteData, seed: int) -> dict[str, Any]:
    if alias == "residual_mlp":
        return {
            "kind": "publication_residual_mlp",
            "input_dim": int(data.x_train.shape[1]),
            "hidden_dim": 96,
            "blocks": 2,
            "dropout": 0.05,
            "num_classes": data.num_classes,
            "task": f"{data.dataset}_classification",
            "seed": seed,
        }
    if alias == "feature_transformer":
        return {
            "kind": "publication_feature_transformer",
            "input_dim": int(data.x_train.shape[1]),
            "chunk_size": 8,
            "d_model": 32,
            "nhead": 4,
            "layers": 2,
            "dropout": 0.05,
            "num_classes": data.num_classes,
            "task": f"{data.dataset}_classification",
            "seed": seed,
        }
    if alias == "tfidf_cnn1d":
        return {
            "kind": "cnn1d",
            "input_dim": int(data.x_train.shape[1]),
            "channels": [16, 32],
            "num_classes": data.num_classes,
            "task": f"{data.dataset}_tfidf_classification",
            "seed": seed,
        }
    if alias == "image_cnn":
        return {
            "kind": "cnn2d",
            "in_channels": int(data.x_train.shape[1]),
            "image_size": int(data.x_train.shape[2]),
            "channels": [16, 32, 64],
            "num_classes": data.num_classes,
            "task": f"{data.dataset}_image_classification",
            "seed": seed,
        }
    if alias == "vit_tiny":
        return {
            "kind": "vit_tiny",
            "image_size": int(data.x_train.shape[2]),
            "patch_size": 7,
            "in_channels": int(data.x_train.shape[1]),
            "d_model": 48,
            "nhead": 4,
            "layers": 2,
            "num_classes": data.num_classes,
            "task": f"{data.dataset}_image_classification",
            "seed": seed,
        }
    if alias == "sequence_cnn":
        return {
            "kind": "publication_sequence_cnn",
            "input_channels": int(data.x_train.shape[2]),
            "channels": [32, 64],
            "num_classes": data.num_classes,
            "dropout": 0.05,
            "task": f"{data.dataset}_sequence_classification",
            "seed": seed,
        }
    if alias == "sequence_transformer":
        return {
            "kind": "publication_sequence_transformer",
            "seq_len": int(data.x_train.shape[1]),
            "input_channels": int(data.x_train.shape[2]),
            "d_model": 48,
            "nhead": 4,
            "layers": 2,
            "dropout": 0.05,
            "num_classes": data.num_classes,
            "task": f"{data.dataset}_sequence_classification",
            "seed": seed,
        }
    raise ValueError(f"unknown architecture alias {alias!r}")


def build_model(architecture: dict[str, Any]) -> nn.Module:
    kind = architecture["kind"]
    kwargs = {key: value for key, value in architecture.items() if key not in {"kind", "task", "seed"}}
    if "channels" in kwargs and isinstance(kwargs["channels"], list):
        kwargs["channels"] = tuple(kwargs["channels"])
    if kind.startswith("publication_"):
        return build_publication_model(kind, **kwargs)
    return build_probe(kind, **kwargs)


def class_weights(labels: np.ndarray, num_classes: int) -> Tensor | None:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    positive = counts[counts > 0]
    if not len(positive) or positive.max() / positive.min() < 2.0:
        return None
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def batch_size(alias: str, regime: str) -> int:
    if alias in {"vit_tiny", "sequence_transformer"}:
        value = 128
    elif alias in {"image_cnn", "sequence_cnn"}:
        value = 256
    else:
        value = 512
    return max(32, value // 2) if regime == "pgd_linf_math" else value


def robust_parameters(dataset: str) -> dict[str, Any]:
    parameters = {
        "kronodroid": {"epsilon": 0.05, "steps": 2, "clip": (None, None)},
        "fashion_mnist": {"epsilon": 8.0 / 255.0, "steps": 2, "clip": (0.0, 1.0)},
        "uci_har": {"epsilon": 0.05, "steps": 2, "clip": (None, None)},
    }
    return parameters[dataset]


def adversarial_batch(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    parameters: dict[str, Any],
) -> Tensor:
    epsilon = float(parameters["epsilon"])
    steps = int(parameters["steps"])
    lower, upper = parameters["clip"]
    was_training = model.training
    model.eval()
    adversarial = inputs.detach() + torch.empty_like(inputs).uniform_(-epsilon, epsilon)
    if lower is not None or upper is not None:
        adversarial = adversarial.clamp(
            min=lower if lower is not None else -float("inf"),
            max=upper if upper is not None else float("inf"),
        )
    step_size = 1.5 * epsilon / max(steps, 1)
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + step_size * gradient.sign()
        delta = (adversarial - inputs).clamp(-epsilon, epsilon)
        adversarial = inputs + delta
        if lower is not None or upper is not None:
            adversarial = adversarial.clamp(
                min=lower if lower is not None else -float("inf"),
                max=upper if upper is not None else float("inf"),
            )
    model.train(was_training)
    return adversarial.detach()


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
        pin_memory=False,
    )


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for features, targets in loader:
        output = model(features.to(device))
        logits.append(output.detach().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = logits.argmax(axis=1)
    num_classes = len(class_names)
    result: dict[str, Any] = {
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "cross_entropy": float(log_loss(labels, probabilities, labels=list(range(num_classes)))),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(num_classes))).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(num_classes)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    try:
        if num_classes == 2:
            result["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
            result["average_precision"] = float(average_precision_score(labels, probabilities[:, 1]))
        else:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            )
    except ValueError as error:
        result["auc_warning"] = str(error)
    return result


def state_on_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def balanced_subset_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, maximum // len(classes))
    selected: list[int] = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        generator.shuffle(candidates)
        selected.extend(candidates[:per_class].tolist())
    generator.shuffle(selected)
    return np.asarray(selected[:maximum], dtype=np.int64)


def save_reference(path: Path, data: SuiteData, seed: int) -> None:
    cohort_indices = balanced_subset_indices(data.y_test, maximum=256, seed=seed + 3000)
    evaluation_indices = balanced_subset_indices(
        data.y_test[cohort_indices],
        maximum=min(64, len(cohort_indices)),
        seed=seed + 4000,
    )
    evaluation = cohort_indices[evaluation_indices]
    values_key = "images" if data.modality == "image" else "features"
    all_key = "all_test_images" if data.modality == "image" else "all_test_features"
    payload = {
        values_key: data.x_test[evaluation].astype(np.float32),
        "labels": data.y_test[evaluation].astype(np.int64),
        all_key: data.x_test[cohort_indices].astype(np.float32),
        "all_test_labels": data.y_test[cohort_indices].astype(np.int64),
        "feature_names": np.asarray(data.feature_names),
        "evaluation_ids": data.test_ids[evaluation],
        "cohort_ids": data.test_ids[cohort_indices],
    }
    np.savez_compressed(path, **payload)


def artifact_complete(path: Path) -> bool:
    required = ("model.pt", "meta.json", "metrics.json", "history.json", "reference.npz", "RUN_COMPLETE.json")
    return all((path / name).is_file() for name in required)


def train_run(
    data: SuiteData,
    alias: str,
    regime: str,
    seed: int,
    output: Path,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    target_validation_accuracy: float | None,
    max_train_batches: int | None,
) -> dict[str, Any]:
    seed_everything(seed)
    architecture = architecture_payload(alias, data, seed)
    model = build_model(architecture).to(device)
    size = batch_size(alias, regime)
    train_loader = make_loader(data.x_train, data.y_train, size, True, seed)
    val_loader = make_loader(data.x_val, data.y_val, size, False, seed)
    test_loader = make_loader(data.x_test, data.y_test, size, False, seed)
    weights = class_weights(data.y_train, data.num_classes)
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(max_epochs, 1))
    robust = robust_parameters(data.dataset) if regime == "pgd_linf_math" else None
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_score = -float("inf")
    stale_epochs = 0
    stopped_reason = "maximum_epochs"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    synchronise(device)
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_index, (features, targets) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            features = features.to(device)
            targets = targets.to(device)
            if robust is not None:
                features = adversarial_batch(model, features, targets, robust)
            optimiser.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running_loss += float(loss.detach().cpu()) * len(targets)
            seen += len(targets)
        scheduler.step()
        val_logits, val_labels = predict(model, val_loader, device)
        validation = metrics_from_logits(val_logits, val_labels, data.class_names)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "train_samples_seen": seen,
            "validation": validation,
            "learning_rate": float(optimiser.param_groups[0]["lr"]),
        }
        history.append(row)
        snapshot = {
            "model_state": state_on_cpu(model),
            "architecture": architecture,
            "epoch": epoch,
            "validation_macro_f1": validation["macro_f1"],
        }
        torch.save(snapshot, checkpoint_dir / f"epoch_{epoch:02d}.pt")
        score = float(validation["macro_f1"])
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_state = snapshot["model_state"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"    epoch={epoch}/{max_epochs} loss={row['train_loss']:.4f} "
            f"val_acc={validation['accuracy']:.4f} val_macro_f1={score:.4f}",
            flush=True,
        )
        if (
            target_validation_accuracy is not None
            and float(validation["accuracy"]) >= target_validation_accuracy
        ):
            stopped_reason = "target_validation_accuracy_reached"
            break
        if (
            target_validation_accuracy is None
            and epoch >= 2
            and regime == "standard"
            and score >= QUALITY_TARGETS[data.dataset]
        ):
            stopped_reason = "validation_quality_target_reached"
            break
        if epoch >= 2 and stale_epochs >= patience:
            stopped_reason = "validation_patience_exhausted"
            break

    if best_state is None:
        raise RuntimeError("training produced no validation-selected state")
    model.load_state_dict(best_state)
    test_logits, test_labels = predict(model, test_loader, device)
    test_metrics = metrics_from_logits(test_logits, test_labels, data.class_names)
    synchronise(device)
    elapsed = time.perf_counter() - started
    quality_passed = bool(test_metrics["macro_f1"] >= QUALITY_TARGETS[data.dataset])

    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": state_on_cpu(model),
        "architecture": architecture,
        "class_names": data.class_names,
        "feature_names": data.feature_names,
        "input_groups": data.input_groups,
        "dataset": data.dataset,
        "training_seed": seed,
        "training_regime": regime,
        "best_epoch": best_epoch,
    }
    torch.save(payload, output / "model.pt")
    save_reference(output / "reference.npz", data, seed)
    np.savez_compressed(
        output / "split_indices.npz",
        train_ids=data.train_ids,
        validation_ids=data.val_ids,
        test_ids=data.test_ids,
    )
    write_json(output / "preprocessing.json", data.preprocessing)
    write_json(output / "history.json", history)
    metrics = {
        "dataset": data.dataset,
        "architecture_alias": alias,
        "training_regime": regime,
        "seed": seed,
        "best_epoch": best_epoch,
        "stopped_reason": stopped_reason,
        "validation_best_macro_f1": best_score,
        "target_validation_accuracy": target_validation_accuracy,
        "test": test_metrics,
        "quality_target_macro_f1": QUALITY_TARGETS[data.dataset],
        "quality_gate_passed": quality_passed,
        "train_seconds": elapsed,
    }
    write_json(output / "metrics.json", metrics)
    metadata = {
        "name": f"{data.dataset}_{alias}_{regime}_seed_{seed}",
        "dataset": data.dataset,
        "architecture_alias": alias,
        "architecture": architecture,
        "training": {
            "seed": seed,
            "regime": regime,
            "max_epochs": max_epochs,
            "patience": patience,
            "target_validation_accuracy": target_validation_accuracy,
            "selection_metric": "validation macro F1",
            "test_evaluations_during_training": 0,
            "class_weighting": None if weights is None else weights.tolist(),
            "robust_parameters": robust,
        },
        "samples": {
            "train": len(data.y_train),
            "validation": len(data.y_val),
            "test": len(data.y_test),
        },
        "class_names": data.class_names,
        "feature_count": len(data.feature_names),
        "input_shape": list(data.x_train.shape[1:]),
        "modality": data.modality,
        "input_domain": data.input_domain,
        "source_files": data.source_files,
        "device": str(device),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "test_metrics": test_metrics,
        "quality_gate_passed": quality_passed,
        "limitations": [
            f"{max_epochs}-epoch cap prioritises suite coverage and does not establish architecture-optimal accuracy.",
            "Lp adversarial training on non-image inputs is mathematical stress, not a domain-valid mutation model.",
            "Formal certification support is operator-dependent and must be assessed separately by VRAVEN.",
        ],
    }
    write_json(output / "meta.json", metadata)
    completion = {
        "status": "complete",
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "model_sha256": sha256(output / "model.pt"),
        "reference_sha256": sha256(output / "reference.npz"),
        "quality_gate_passed": quality_passed,
    }
    write_json(output / "RUN_COMPLETE.json", completion)
    model.to("cpu")
    del model
    clear_device_cache(device)
    return {**metrics, "artifact": str(output.relative_to(ROOT))}


def regimes_for(dataset: str) -> tuple[str, ...]:
    return ("standard", "pgd_linf_math") if dataset in ROBUST_DATASETS else ("standard",)


def expected_run_count(datasets: list[str], seeds: list[int]) -> int:
    configurations = sum(
        len(ARCHITECTURES[dataset]) * len(regimes_for(dataset)) for dataset in datasets
    )
    return configurations * len(seeds)


def aggregate(output_dir: Path, expected: int) -> dict[str, Any]:
    runs = []
    failures = []
    for metrics_path in sorted(output_dir.glob("*/*/*/seed_*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["artifact"] = str(metrics_path.parent.relative_to(ROOT))
        runs.append(payload)
    for failure_path in sorted(output_dir.glob("*/*/*/seed_*/RUN_FAILED.json")):
        failures.append(json.loads(failure_path.read_text(encoding="utf-8")))
    summary = {
        "schema_version": 1,
        "expected_final_checkpoints": expected,
        "completed_final_checkpoints": len(runs),
        "failed_runs": len(failures),
        "quality_gate_passed": sum(bool(run.get("quality_gate_passed")) for run in runs),
        "total_train_seconds": sum(float(run.get("train_seconds", 0.0)) for run in runs),
        "training_performed_on_mps": any(
            "mps" in json.loads((ROOT / run["artifact"] / "meta.json").read_text()).get("device", "")
            for run in runs
        ),
        "runs": runs,
        "failures": failures,
    }
    write_json(output_dir / "SUITE_SUMMARY.json", summary)
    if runs:
        rows = [
            {
                "dataset": run["dataset"],
                "architecture": run["architecture_alias"],
                "regime": run["training_regime"],
                "seed": run["seed"],
                "best_epoch": run["best_epoch"],
                "test_accuracy": run["test"]["accuracy"],
                "test_balanced_accuracy": run["test"]["balanced_accuracy"],
                "test_macro_f1": run["test"]["macro_f1"],
                "quality_gate_passed": run["quality_gate_passed"],
                "train_seconds": run["train_seconds"],
                "artifact": run["artifact"],
            }
            for run in runs
        ]
        pd.DataFrame(rows).to_csv(output_dir / "SUITE_RESULTS.csv", index=False)
    return summary


def main() -> int:
    args = parse_args()
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.target_validation_accuracy is not None and not 0.0 < args.target_validation_accuracy <= 1.0:
        raise ValueError("--target-validation-accuracy must be in (0, 1]")
    device = pick_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if args.smoke else args.max_epochs
    max_train_batches = 2 if args.smoke and args.max_train_batches is None else args.max_train_batches
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    expected = expected_run_count(args.datasets, seeds)
    suite_manifest = {
        "title": "VRAVEN joint-certification publication model suite",
        "device": str(device),
        "datasets": args.datasets,
        "architectures": {dataset: ARCHITECTURES[dataset] for dataset in args.datasets},
        "robust_datasets": sorted(ROBUST_DATASETS.intersection(args.datasets)),
        "seeds": seeds,
        "max_epochs": max_epochs,
        "patience": args.patience,
        "target_validation_accuracy": args.target_validation_accuracy,
        "expected_final_checkpoints": expected,
        "smoke": args.smoke,
        "python_executable": sys.executable,
        "process_id": os.getpid(),
    }
    write_json(output_dir / "SUITE_MANIFEST.json", suite_manifest)
    print(
        f"device={device} datasets={len(args.datasets)} seeds={len(seeds)} "
        f"expected_final_checkpoints={expected} max_epochs={max_epochs}",
        flush=True,
    )

    completed = 0
    failed = 0
    for dataset in args.datasets:
        for seed in seeds:
            print(f"\n=== materialise dataset={dataset} seed={seed} ===", flush=True)
            data = DATA_LOADERS[dataset](seed)
            print(
                f"  shapes train={data.x_train.shape} val={data.x_val.shape} "
                f"test={data.x_test.shape} classes={data.class_names}",
                flush=True,
            )
            for alias in ARCHITECTURES[dataset]:
                for regime in regimes_for(dataset):
                    artifact = output_dir / dataset / alias / regime / f"seed_{seed:03d}"
                    if artifact_complete(artifact) and not args.force:
                        completed += 1
                        print(f"  skip complete {artifact.relative_to(ROOT)}", flush=True)
                        continue
                    print(
                        f"  [{completed + failed + 1}/{expected}] train {dataset}/{alias}/{regime}/seed={seed}",
                        flush=True,
                    )
                    try:
                        result = train_run(
                            data,
                            alias,
                            regime,
                            seed,
                            artifact,
                            device,
                            max_epochs,
                            args.patience,
                            args.min_delta,
                            args.target_validation_accuracy,
                            max_train_batches,
                        )
                        completed += 1
                        print(
                            f"    complete test_acc={result['test']['accuracy']:.4f} "
                            f"test_macro_f1={result['test']['macro_f1']:.4f} "
                            f"seconds={result['train_seconds']:.1f}",
                            flush=True,
                        )
                    except Exception as error:  # noqa: BLE001
                        failed += 1
                        failure = {
                            "dataset": dataset,
                            "architecture": alias,
                            "regime": regime,
                            "seed": seed,
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                        write_json(artifact / "RUN_FAILED.json", failure)
                        print(f"    FAILED {error!r}", flush=True)
                        if args.fail_fast:
                            raise
            del data
            clear_device_cache(device)
            summary = aggregate(output_dir, expected)
            print(
                f"  progress completed={summary['completed_final_checkpoints']}/{expected} "
                f"failed={summary['failed_runs']}",
                flush=True,
            )

    summary = aggregate(output_dir, expected)
    print(
        f"\nSUITE COMPLETE completed={summary['completed_final_checkpoints']}/{expected} "
        f"failed={summary['failed_runs']} quality_passed={summary['quality_gate_passed']}",
        flush=True,
    )
    return 0 if summary["completed_final_checkpoints"] == expected and not summary["failed_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
