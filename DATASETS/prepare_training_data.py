#!/usr/bin/env python3
"""Create canonical model-ready files from the immutable raw dataset suite.

This script performs format conversion only. It does not split datasets beyond
their official splits, fit preprocessing, select features, or train models.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def ready_dir(dataset: str) -> Path:
    destination = ROOT / dataset / "training_ready"
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def copy_canonical(dataset: str, source_name: str, destination_name: str) -> Path:
    source = ROOT / dataset / "raw" / source_name
    destination = ready_dir(dataset) / destination_name
    shutil.copy2(source, destination)
    return destination


def decode_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(
            lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
        )
    return frame


def prepare_kronodroid() -> dict:
    path = copy_canonical(
        "kronodroid", "kronodroid_combined.csv", "kronodroid_combined.csv"
    )
    return {
        "files": [path],
        "format": "CSV",
        "target": "Malware",
        "exclude_from_binary_inputs": ["Package", "sha256", "MalFamily"],
        "protocol_warning": (
            "Audit Detection_Ratio and Scanners for collection-time label leakage; "
            "use chronological splits for temporal claims."
        ),
    }


def read_idx_gzip(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic = struct.unpack(">I", stream.read(4))[0]
        if magic == 2051:
            count, rows, columns = struct.unpack(">III", stream.read(12))
            values = np.frombuffer(stream.read(), dtype=np.uint8)
            return values.reshape(count, rows, columns)
        if magic == 2049:
            count = struct.unpack(">I", stream.read(4))[0]
            values = np.frombuffer(stream.read(), dtype=np.uint8)
            return values.reshape(count)
        raise ValueError(f"unsupported IDX magic number {magic} in {path}")


def prepare_fashion_mnist() -> dict:
    raw = ROOT / "fashion_mnist" / "raw"
    destination = ready_dir("fashion_mnist") / "fashion_mnist.npz"
    arrays = {
        "x_train": read_idx_gzip(raw / "train-images-idx3-ubyte.gz"),
        "y_train": read_idx_gzip(raw / "train-labels-idx1-ubyte.gz"),
        "x_test": read_idx_gzip(raw / "t10k-images-idx3-ubyte.gz"),
        "y_test": read_idx_gzip(raw / "t10k-labels-idx1-ubyte.gz"),
    }
    np.savez_compressed(destination, **arrays)
    return {
        "files": [destination],
        "format": "NPZ: uint8 image tensors and integer labels",
        "target": "y_train/y_test",
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "recommended_models": ["CNN", "ViT", "MLP"],
    }


def prepare_uci_har() -> dict:
    raw = ROOT / "uci_har" / "raw" / "UCI HAR Dataset"
    destination = ready_dir("uci_har")

    x_train = np.loadtxt(raw / "train" / "X_train.txt", dtype=np.float32)
    x_test = np.loadtxt(raw / "test" / "X_test.txt", dtype=np.float32)
    y_train = np.loadtxt(raw / "train" / "y_train.txt", dtype=np.int64) - 1
    y_test = np.loadtxt(raw / "test" / "y_test.txt", dtype=np.int64) - 1
    subject_train = np.loadtxt(raw / "train" / "subject_train.txt", dtype=np.int64)
    subject_test = np.loadtxt(raw / "test" / "subject_test.txt", dtype=np.int64)

    feature_path = destination / "uci_har_features.npz"
    np.savez_compressed(
        feature_path,
        x_train=x_train,
        y_train=y_train,
        subject_train=subject_train,
        x_test=x_test,
        y_test=y_test,
        subject_test=subject_test,
    )

    signal_names = (
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    )

    def load_signals(split: str) -> np.ndarray:
        signal_dir = raw / split / "Inertial Signals"
        arrays = [
            np.loadtxt(signal_dir / f"{signal}_{split}.txt", dtype=np.float32)
            for signal in signal_names
        ]
        return np.stack(arrays, axis=-1)

    signals_train = load_signals("train")
    signals_test = load_signals("test")
    signal_path = destination / "uci_har_inertial_signals.npz"
    np.savez_compressed(
        signal_path,
        x_train=signals_train,
        y_train=y_train,
        subject_train=subject_train,
        x_test=signals_test,
        y_test=y_test,
        subject_test=subject_test,
        channel_names=np.asarray(signal_names),
    )
    return {
        "files": [feature_path, signal_path],
        "format": "NPZ: engineered features and 128-step × 9-channel sequences",
        "target": "y_train/y_test (converted from 1–6 to 0–5)",
        "shapes": {
            "features_train": list(x_train.shape),
            "features_test": list(x_test.shape),
            "signals_train": list(signals_train.shape),
            "signals_test": list(signals_test.shape),
        },
        "split_rule": "Preserve official subject-disjoint train/test split.",
        "recommended_models": ["MLP", "1-D CNN", "GRU", "Transformer"],
    }


def prepare_sms_spam() -> dict:
    source = ROOT / "sms_spam" / "raw" / "SMSSpamCollection"
    frame = pd.read_csv(
        source,
        sep="\t",
        names=["label", "text"],
        dtype=str,
        quoting=csv.QUOTE_NONE,
    )
    destination = ready_dir("sms_spam") / "sms_spam.csv"
    frame.to_csv(destination, index=False)
    return {
        "files": [destination],
        "format": "CSV UTF-8",
        "target": "label",
        "shape": list(frame.shape),
        "recommended_models": ["Text CNN", "GRU", "compact Transformer"],
    }


WDBC_COLUMNS = [
    "id",
    "diagnosis",
    *[
        f"{measure}_{statistic}"
        for statistic in ("mean", "se", "worst")
        for measure in (
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave_points",
            "symmetry",
            "fractal_dimension",
        )
    ],
]


def prepare_wdbc() -> dict:
    source = ROOT / "wdbc" / "raw" / "wdbc.data"
    frame = pd.read_csv(source, header=None, names=WDBC_COLUMNS)
    destination = ready_dir("wdbc") / "wdbc.csv"
    frame.to_csv(destination, index=False)
    return {
        "files": [destination],
        "format": "CSV",
        "target": "diagnosis",
        "excluded_inputs": ["id"],
        "shape": list(frame.shape),
        "recommended_models": ["MLP", "residual MLP", "tabular Transformer"],
    }


def prepare_eeg_eye_state() -> dict:
    source = ROOT / "eeg_eye_state" / "raw" / "EEG Eye State.arff"
    values, _ = arff.loadarff(source)
    frame = decode_object_columns(pd.DataFrame(values))
    frame["eyeDetection"] = pd.to_numeric(frame["eyeDetection"], downcast="integer")
    destination = ready_dir("eeg_eye_state") / "eeg_eye_state.csv"
    frame.to_csv(destination, index=False)
    return {
        "files": [destination],
        "format": "CSV in original chronological order",
        "target": "eyeDetection",
        "shape": list(frame.shape),
        "split_rule": "Use blocked chronological evaluation; never random-row leakage.",
        "recommended_models": ["1-D CNN", "GRU", "Transformer"],
    }


def prepare_ai4i() -> dict:
    path = copy_canonical(
        "ai4i_predictive_maintenance", "ai4i2020.csv", "ai4i2020.csv"
    )
    return {
        "files": [path],
        "format": "CSV",
        "target": "Machine failure",
        "excluded_inputs": ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        "protocol_warning": "Failure-mode indicator columns would leak the aggregate target.",
        "recommended_models": ["MLP", "residual MLP", "tabular Transformer"],
    }


def prepare_dry_bean() -> dict:
    source = ROOT / "dry_bean" / "raw" / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx"
    frame = pd.read_excel(source)
    destination = ready_dir("dry_bean") / "dry_bean.csv"
    frame.to_csv(destination, index=False)
    return {
        "files": [destination],
        "format": "CSV",
        "target": "Class",
        "shape": list(frame.shape),
        "recommended_models": ["MLP", "residual MLP", "tabular Transformer"],
    }


def prepare_bank_marketing() -> dict:
    source = (
        ROOT
        / "bank_marketing"
        / "raw"
        / "bank-additional"
        / "bank-additional-full.csv"
    )
    frame = pd.read_csv(source, sep=";")
    destination = ready_dir("bank_marketing") / "bank_marketing.csv"
    frame.to_csv(destination, index=False)
    return {
        "files": [destination],
        "format": "CSV; canonical bank-additional-full variant",
        "target": "y",
        "excluded_inputs": ["duration"],
        "shape": list(frame.shape),
        "protocol_warning": "Call duration is unavailable before the outcome and must be excluded.",
        "recommended_models": ["embedding MLP", "residual MLP", "tabular Transformer"],
    }


def prepare_wine_quality() -> dict:
    raw = ROOT / "wine_quality" / "raw"
    red = pd.read_csv(raw / "winequality-red.csv", sep=";")
    white = pd.read_csv(raw / "winequality-white.csv", sep=";")
    red.insert(0, "wine_type", "red")
    white.insert(0, "wine_type", "white")
    frame = pd.concat([red, white], ignore_index=True)
    destination = ready_dir("wine_quality") / "wine_quality.csv"
    frame.to_csv(destination, index=False)
    return {
        "files": [destination],
        "format": "CSV; red and white subsets combined with wine_type",
        "target": "quality",
        "shape": list(frame.shape),
        "protocol_warning": "Pre-register ordinal or multiclass target handling before training.",
        "recommended_models": ["ordinal MLP", "residual MLP", "tabular Transformer"],
    }


def main() -> None:
    builders = {
        "kronodroid": prepare_kronodroid,
        "fashion_mnist": prepare_fashion_mnist,
        "uci_har": prepare_uci_har,
        "sms_spam": prepare_sms_spam,
        "wdbc": prepare_wdbc,
        "eeg_eye_state": prepare_eeg_eye_state,
        "ai4i_predictive_maintenance": prepare_ai4i,
        "dry_bean": prepare_dry_bean,
        "bank_marketing": prepare_bank_marketing,
        "wine_quality": prepare_wine_quality,
    }

    records = []
    for dataset, builder in builders.items():
        print(f"prepare  {dataset}")
        record = builder()
        record["dataset"] = dataset
        record["files"] = [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in record["files"]
        ]
        records.append(record)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "format_conversion_only": True,
        "preprocessing_fitted": False,
        "feature_selection_performed": False,
        "training_performed": False,
        "datasets": records,
    }
    target = ROOT / "TRAINING_READY_MANIFEST.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest {target.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
