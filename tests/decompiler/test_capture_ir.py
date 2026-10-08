from __future__ import annotations

import torch

from vraven.decompiler import CaptureConfig, MechanismIR, ModelCaptureRouter


def test_export_capture_is_equivalent_and_serialisable(tmp_path, tiny_model, tiny_data):
    inputs, _ = tiny_data
    captured = ModelCaptureRouter(CaptureConfig()).capture(tiny_model, (inputs[:3],), {})
    assert captured.graph_module is not None
    assert captured.ir.coverage.numeric_equivalence_passed is True
    assert 0.0 <= captured.ir.coverage.operator_coverage <= 1.0
    assert 0.0 <= captured.ir.coverage.parameter_coverage <= 1.0
    assert captured.ir.nodes
    assert captured.ir.edges

    path = captured.ir.save(tmp_path / "mechanism_ir.json")
    restored = MechanismIR.load(path)
    assert restored.model_class == captured.ir.model_class
    assert restored.content_hash() == captured.ir.content_hash()

    with torch.no_grad():
        expected = tiny_model(inputs[:3])
        observed = captured.graph_module(inputs[:3])
    assert torch.allclose(expected, observed, atol=1e-5, rtol=1e-4)
