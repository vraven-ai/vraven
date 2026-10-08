from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from vraven.bundle import load_model_bundle
from vraven.cli import _demo_model, main


def _write_bundle(tmp_path):
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "sample.npy"
    torch.save(_demo_model().state_dict(), model_path)
    np.save(input_path, np.zeros((2, 12), dtype=np.float32))
    (tmp_path / "bundle_model.py").write_text(
        "from vraven.cli import _demo_model\n\n"
        "def build():\n"
        "    return _demo_model()\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "model": {
            "path": "model.pt",
            "format": "state_dict",
            "factory": "bundle_model:build",
            "factory_kwargs": {},
        },
        "input": {"path": "sample.npy"},
        "class_names": ["class_0", "class_1", "class_2"],
        "defaults": {
            "task": "explain",
            "preset": "quick",
            "format": "json",
            "output": "results/explanation.json",
        },
    }
    (tmp_path / "vraven.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_model_bundle_runs_without_repeating_model_configuration(tmp_path, capsys) -> None:
    _write_bundle(tmp_path)
    bundle = load_model_bundle(tmp_path)
    assert bundle.model.name == "model.pt"
    assert bundle.factory == "bundle_model:build"

    assert main(["bundle", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["model_format"] == "state_dict"

    assert main(["quick", str(tmp_path)]) == 0
    report_path = tmp_path / "results" / "explanation.json"
    assert report_path.exists()
    assert json.loads(report_path.read_text())["mode"] == "fast"


def test_model_bundle_rejects_paths_outside_directory(tmp_path) -> None:
    manifest = _write_bundle(tmp_path)
    manifest["input"]["path"] = "../outside.npy"
    (tmp_path / "vraven.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the bundle"):
        load_model_bundle(tmp_path)


def test_bundle_init_creates_valid_manifest(tmp_path, capsys) -> None:
    torch.save(_demo_model().state_dict(), tmp_path / "model.pt")
    np.save(tmp_path / "sample.npy", np.zeros((1, 12), dtype=np.float32))
    assert main(
        [
            "bundle-init",
            str(tmp_path),
            "--factory",
            "vraven.cli:_demo_model",
            "--format",
            "json",
            "--output",
            "results/report.json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["factory"] == "vraven.cli:_demo_model"
    assert (tmp_path / "vraven.json").is_file()
    assert load_model_bundle(tmp_path).defaults["output"] == "results/report.json"
