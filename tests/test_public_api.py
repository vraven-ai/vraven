from __future__ import annotations

import vraven
from vraven import adapters, core


def test_public_api_exports_runtime_entry_points() -> None:
    assert callable(vraven.explain)
    assert callable(vraven.explain_dataset)
    assert callable(vraven.compare_models)
    assert callable(vraven.compare_checkpoints)
    assert callable(vraven.load_pytorch_model)
    assert vraven.InputDomainSpec is not None


def test_adapter_and_core_subpackages_export_their_public_types() -> None:
    assert adapters.ModelAdapter is not None
    assert adapters.PyTorchAdapter is not None
    assert callable(adapters.resolve_adapter)
    assert callable(adapters.load_pytorch_model)
    assert issubclass(core.DeviceMismatchError, core.VRavenError)
    assert issubclass(core.ModelLoadError, core.VRavenError)
