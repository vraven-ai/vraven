from __future__ import annotations

import pytest
import torch

import vraven
from vraven.core.errors import ModelLoadError


def factory() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))


def test_state_dict_loading_requires_and_uses_architecture_factory(tmp_path) -> None:
    torch.manual_seed(51)
    original = factory()
    path = tmp_path / "weights.pt"
    torch.save(original.state_dict(), path)
    with pytest.raises(ModelLoadError):
        vraven.load_pytorch_model(path)
    loaded = vraven.load_pytorch_model(path, model_factory=factory)
    inputs = torch.randn(4, 3)
    assert torch.allclose(original(inputs), loaded(inputs))
    assert loaded.training is False


def test_nested_state_dict_and_prefix_removal(tmp_path) -> None:
    original = factory()
    prefixed = {f"module.{name}": value for name, value in original.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 4, "state_dict": prefixed}, path)
    loaded = vraven.load_pytorch_model(path, model_factory=factory, strip_prefix="module.")
    assert all(torch.equal(a, b) for a, b in zip(original.state_dict().values(), loaded.state_dict().values(), strict=False))


def test_pickle_loading_is_explicitly_blocked_by_default(tmp_path) -> None:
    path = tmp_path / "model.pt"
    torch.save(factory(), path)
    with pytest.raises(ModelLoadError, match="disabled"):
        vraven.load_pytorch_model(path, format="pickle")
    with pytest.warns(RuntimeWarning):
        loaded = vraven.load_pytorch_model(path, format="pickle", allow_unsafe_pickle=True)
    assert isinstance(loaded, torch.nn.Module)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_torchscript_loading(tmp_path) -> None:
    scripted = torch.jit.trace(factory(), torch.randn(1, 3))
    path = tmp_path / "model.ts"
    scripted.save(str(path))
    loaded = vraven.load_pytorch_model(path, format="torchscript")
    assert isinstance(loaded, torch.nn.Module)


def test_model_loading_rejects_missing_files_and_unknown_formats(tmp_path) -> None:
    with pytest.raises(ModelLoadError, match="does not exist"):
        vraven.load_pytorch_model(tmp_path / "missing.pt", model_factory=factory)

    path = tmp_path / "weights.pt"
    torch.save(factory().state_dict(), path)
    with pytest.raises(ModelLoadError, match="format must"):
        vraven.load_pytorch_model(path, model_factory=factory, format="unknown")  # type: ignore[arg-type]


def test_model_loading_rejects_missing_nested_key(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"weights": factory().state_dict()}, path)
    with pytest.raises(ModelLoadError, match="No state dictionary key"):
        vraven.load_pytorch_model(
            path,
            model_factory=factory,
            state_dict_key="state_dict",
        )


def test_non_strict_state_dict_reports_incompatibilities(tmp_path) -> None:
    model = factory()
    state = model.state_dict()
    state.pop("2.bias")
    path = tmp_path / "incomplete.pt"
    torch.save(state, path)
    with pytest.warns(RuntimeWarning, match="missing or unexpected"):
        loaded = vraven.load_pytorch_model(path, model_factory=factory, strict=False)
    assert isinstance(loaded, torch.nn.Module)
