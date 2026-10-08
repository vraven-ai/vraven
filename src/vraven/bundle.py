"""Validated self-describing model bundles for the public VRAVEN CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAME = "vraven.json"
SUPPORTED_MODEL_FORMATS = {"state_dict", "pickle"}


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Resolved files and defaults from one contained model-bundle directory."""

    root: Path
    manifest: Path
    model: Path
    input: Path
    labels: Path | None
    model_format: str = "state_dict"
    factory: str | None = None
    factory_kwargs: dict[str, Any] = field(default_factory=dict)
    state_dict_key: str | None = None
    strip_prefix: str | None = None
    device: str = "cpu"
    defaults: dict[str, Any] = field(default_factory=dict)
    class_names: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "root": str(self.root),
            "manifest": str(self.manifest),
            "model": str(self.model),
            "input": str(self.input),
            "labels": str(self.labels) if self.labels else None,
            "model_format": self.model_format,
            "factory": self.factory,
            "factory_kwargs": self.factory_kwargs,
            "state_dict_key": self.state_dict_key,
            "strip_prefix": self.strip_prefix,
            "device": self.device,
            "defaults": self.defaults,
            "class_names": list(self.class_names),
        }


def _object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _contained_file(root: Path, value: Any, name: str, *, required: bool) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required in {MANIFEST_NAME}")
        return None
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"{name} must be relative to the bundle directory")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} must remain inside the bundle directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"Bundle {name} does not exist: {resolved}")
    return resolved


def _validate_output_path(root: Path, defaults: dict[str, Any]) -> None:
    value = defaults.get("output")
    if value is None:
        return
    relative = Path(str(value))
    if relative.is_absolute() or not (root / relative).resolve().is_relative_to(root):
        raise ValueError("defaults.output must be relative and remain inside the bundle directory")


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load and validate a public ``vraven.json`` model bundle."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Model bundle is not a directory: {root}")
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"Model bundle requires {MANIFEST_NAME}: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{MANIFEST_NAME} must contain a JSON object")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "1.0":
        raise ValueError("vraven.json schema_version must be '1.0'")

    model_config = _object(payload.get("model"), "model")
    input_config = _object(payload.get("input"), "input")
    defaults = _object(payload.get("defaults"), "defaults")
    _validate_output_path(root, defaults)
    model_format = str(model_config.get("format", "state_dict"))
    if model_format not in SUPPORTED_MODEL_FORMATS:
        choices = ", ".join(sorted(SUPPORTED_MODEL_FORMATS))
        raise ValueError(f"model.format must be one of: {choices}")
    factory = model_config.get("factory")
    if factory is not None and not isinstance(factory, str):
        raise ValueError("model.factory must be a module:object string")
    if model_format == "state_dict" and not factory:
        raise ValueError("A state_dict bundle requires model.factory")
    factory_kwargs = _object(model_config.get("factory_kwargs"), "model.factory_kwargs")
    for key in ("state_dict_key", "strip_prefix"):
        if model_config.get(key) is not None and not isinstance(model_config[key], str):
            raise ValueError(f"model.{key} must be a string when supplied")
    class_names = payload.get("class_names", [])
    if not isinstance(class_names, list) or not all(isinstance(item, str) for item in class_names):
        raise ValueError("class_names must be an array of strings")

    model_path = _contained_file(root, model_config.get("path"), "model.path", required=True)
    input_path = _contained_file(root, input_config.get("path"), "input.path", required=True)
    labels_path = _contained_file(root, input_config.get("labels"), "input.labels", required=False)
    assert model_path is not None and input_path is not None
    return ModelBundle(
        root=root,
        manifest=manifest,
        model=model_path,
        input=input_path,
        labels=labels_path,
        model_format=model_format,
        factory=factory,
        factory_kwargs=factory_kwargs,
        state_dict_key=model_config.get("state_dict_key"),
        strip_prefix=model_config.get("strip_prefix"),
        device=str(model_config.get("device", "cpu")),
        defaults=defaults,
        class_names=tuple(class_names),
    )
