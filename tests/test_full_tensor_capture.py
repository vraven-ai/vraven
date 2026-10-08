from __future__ import annotations

import pytest
import torch

import vraven
from vraven.core.errors import CapabilityError, DeviceMismatchError


def test_full_tensor_capture_is_explicit_ephemeral_and_releasable() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2))
    report = vraven.explain(model, torch.randn(3, 4), capture_full_tensors=True)
    names = report.retained_tensor_names()
    assert names
    assert report.full_tensor(names[0]).shape[0] == 3
    report.release_full_tensors()
    assert report.retained_tensor_names() == ()
    with pytest.raises(CapabilityError):
        report.full_tensor(names[0])


def test_full_tensors_are_not_retained_by_default() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    report = vraven.explain(model, torch.randn(1, 2))
    assert report.retained_tensor_names() == ()


def test_explicit_device_declaration_must_match_live_model() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(DeviceMismatchError):
        vraven.Explainer(model, vraven.VRavenConfig(device="cuda"))
