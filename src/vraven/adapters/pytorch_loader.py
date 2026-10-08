"""Explicit PyTorch model loading helpers with safe defaults."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import torch

from vraven.core.errors import ModelLoadError

PyTorchModelFormat = Literal["state_dict", "torchscript", "pickle"]


def _extract_state_dict(payload: Any, state_dict_key: str | None) -> Mapping[str, torch.Tensor]:
    candidate = payload
    if state_dict_key is not None:
        if not isinstance(payload, Mapping) or state_dict_key not in payload:
            raise ModelLoadError(f"No state dictionary key {state_dict_key!r} exists in the checkpoint")
        candidate = payload[state_dict_key]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        candidate = payload["state_dict"]

    if not isinstance(candidate, Mapping) or not candidate:
        raise ModelLoadError(
            "The checkpoint does not contain a non-empty state dictionary. "
            "Supply state_dict_key for a nested checkpoint."
        )
    invalid = [name for name, value in candidate.items() if not isinstance(name, str) or not isinstance(value, torch.Tensor)]
    if invalid:
        raise ModelLoadError("The selected state dictionary contains non-tensor values")
    return OrderedDict((str(name), value) for name, value in candidate.items())


def _strip_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str | None,
) -> OrderedDict[str, torch.Tensor]:
    if not prefix:
        return OrderedDict(state_dict)
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        output[name[len(prefix) :] if name.startswith(prefix) else name] = tensor
    return output


def load_pytorch_model(
    path: str | Path,
    *,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    format: PyTorchModelFormat = "state_dict",
    state_dict_key: str | None = None,
    strip_prefix: str | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    allow_unsafe_pickle: bool = False,
) -> torch.nn.Module:
    """Load a trusted PyTorch model using an explicit format.

    ``state_dict`` is the default and uses ``torch.load(..., weights_only=True)``.
    It requires a caller-supplied architecture factory. ``torchscript`` uses
    ``torch.jit.load``. Python-pickle model loading is disabled unless
    ``allow_unsafe_pickle=True`` because untrusted pickle content may execute code.
    """

    source = Path(path)
    if not source.is_file():
        raise ModelLoadError(f"Model file does not exist: {source}")

    if format == "torchscript":
        try:
            model = torch.jit.load(str(source), map_location=map_location)
        except Exception as error:
            raise ModelLoadError(f"Unable to load TorchScript model {source}: {error}") from error
        model.eval()
        return model

    if format == "pickle":
        if not allow_unsafe_pickle:
            raise ModelLoadError(
                "Python-pickle model loading is disabled. Pass allow_unsafe_pickle=True only "
                "for a file you trust and control. Prefer a state_dict or TorchScript model."
            )
        warnings.warn(
            "Loading a pickled PyTorch model can execute code embedded in the file. "
            "Proceed only with a trusted model artefact.",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            model = torch.load(source, map_location=map_location, weights_only=False)
        except Exception as error:
            raise ModelLoadError(f"Unable to load trusted pickled model {source}: {error}") from error
        if not isinstance(model, torch.nn.Module):
            raise ModelLoadError("The pickled object is not a torch.nn.Module")
        model.eval()
        return model

    if format != "state_dict":
        raise ModelLoadError("format must be 'state_dict', 'torchscript' or 'pickle'")
    if model_factory is None:
        raise ModelLoadError(
            "state_dict loading requires model_factory because weights do not define the architecture"
        )
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
        state_dict = _strip_prefix(_extract_state_dict(payload, state_dict_key), strip_prefix)
        model = model_factory()
    except ModelLoadError:
        raise
    except Exception as error:
        raise ModelLoadError(f"Unable to read state dictionary {source}: {error}") from error
    if not isinstance(model, torch.nn.Module):
        raise ModelLoadError("model_factory must return a torch.nn.Module")
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except Exception as error:
        raise ModelLoadError(f"State dictionary is incompatible with the supplied architecture: {error}") from error
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        warnings.warn(
            "Loaded a non-strict state dictionary with missing or unexpected keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}",
            RuntimeWarning,
            stacklevel=2,
        )
    model.to(torch.device(map_location))
    model.eval()
    return model
