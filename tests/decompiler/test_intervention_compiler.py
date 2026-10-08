from __future__ import annotations

import torch

from vraven.decompiler import (
    CaptureConfig,
    GraphInterventionCompiler,
    InterventionKind,
    InterventionSpec,
    ModelCaptureRouter,
)


def test_graph_intervention_is_explicit_and_changes_output(tiny_model, tiny_data):
    inputs, _ = tiny_data
    captured = ModelCaptureRouter(CaptureConfig()).capture(tiny_model, (inputs[:2],), {})
    candidates = [
        node for node in captured.ir.mechanism_nodes()
        if node.intervention_capable and node.name.startswith("relu")
    ]
    assert candidates
    spec = InterventionSpec(node_id=candidates[0].name, kind=InterventionKind.ZERO)
    compiled = GraphInterventionCompiler(captured).compile([spec])
    with torch.no_grad():
        baseline = captured.graph_module(inputs[:8])
        changed = compiled(inputs[:8])
    assert compiled.gate_nodes
    assert "_vraven_gate_" in compiled.graph_code
    assert not torch.allclose(baseline, changed)
