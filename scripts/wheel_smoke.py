#!/usr/bin/env python3
"""Smoke-test an installed VRAVEN wheel through its public API only."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import vraven


def make_model(seed: int = 17) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 12),
        torch.nn.GELU(),
        torch.nn.Linear(12, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    ).eval()


def main() -> None:
    torch.manual_seed(29)
    model = make_model()
    inputs = torch.randn(6, 4)

    report = vraven.explain(
        model,
        inputs,
        target=1,
        mode="gradient",
        capture_full_tensors=True,
    )
    assert report.trace.layers
    assert report.trace.prediction is not None
    assert report.retained_tensor_names()
    assert report.full_tensor(report.retained_tensor_names()[0]).numel() > 0

    light = report.reasoning_dna(mode="publication_light")
    dark = report.reasoning_dna(mode="interactive_dark")
    assert light.layout.paper_bgcolor == "#FFFFFF"
    assert dark.layout.paper_bgcolor != "#FFFFFF"
    assert report.activation_energy(mode="publication_light").data
    assert report.reasoning_graph(mode="interactive_dark").data
    audit = report.scientific_audit()
    assert audit.evidence_tier == "observational_first_order"
    assert audit.concentration.effective_layers is not None
    report.release_full_tensors()
    assert report.retained_tensor_names() == ()

    causal_report = vraven.explain(model, inputs, target=1, mode="causal", causal_top_k=3)
    causal_audit = causal_report.scientific_audit()
    assert causal_audit.evidence_tier == "targeted_interventional"
    assert causal_audit.concordance.layer_names
    assert causal_report.mechanism_concordance(mode="publication_light").data

    verifier_model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    ).eval()
    verifier_input = torch.tensor([[0.5, -0.25]])
    verifier = vraven.IntervalBoundVerifier(verifier_model)
    formal = verifier.certify(
        verifier_input,
        epsilon_grid=(0.0, 0.01),
        radius_search_steps=4,
    )
    assert formal.sample_points[0].certified
    assert formal.verifier == "vraven_ibp_fx_v1"
    formal_mechanism = vraven.FormalCausalMechanismVerifier(verifier_model).certify(
        verifier_input,
        mechanism=vraven.MechanismPropertySpec(
            node_id="1",
            top_k=1,
            min_cosine=-1.0,
            max_relative_l2_drift=10.0,
        ),
        causal=vraven.CausalEffectPropertySpec(
            node_id="1",
            coordinate_indices=(0,),
            min_cosine=-1.0,
            max_relative_l2_drift=10.0,
            min_sign_stable_fraction=0.0,
        ),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
    )
    assert formal_mechanism.joint_radius_lower_bound >= 0.0
    assert formal_mechanism.verifier == "vraven_formal_causal_mechanism_relational_ibp_v3"

    with tempfile.TemporaryDirectory() as directory:
        trace_path = report.save_trace(Path(directory) / "trace.json")
        loaded = vraven.VRavenReport.load_trace(trace_path)
        assert loaded.trace.trace_id == report.trace.trace_id
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        assert "full_tensors" not in payload
        html_path = causal_report.export_html(
            Path(directory) / "report.html",
            visuals=("mechanism_concordance",),
        )
        assert "Scientific Evidence Audit" in html_path.read_text(encoding="utf-8")

    dataset = vraven.explain_dataset(
        model,
        inputs,
        targets=1,
        mode="fast",
        max_samples=6,
        fingerprint_dimensions=32,
    )
    assert dataset.fingerprints.shape == (6, 32)
    assert dataset.similarity_matrix().shape == (6, 6)
    assert dataset.reasoning_similarity(mode="publication_light").data
    assert dataset.reasoning_landscape(mode="interactive_dark").data
    assert dataset.reasoning_outliers(mode="publication_light").data

    identical_model = copy.deepcopy(model)
    comparison = vraven.compare_models(
        model,
        identical_model,
        inputs,
        targets=1,
        max_samples=4,
        fingerprint_dimensions=32,
    )
    assert np.allclose(comparison.fingerprint_similarity, 1.0, atol=1e-6)
    assert comparison.alignment(mode="publication_light").data

    earlier = make_model(seed=7)
    evolution = vraven.compare_checkpoints(
        [earlier, model],
        inputs[:2],
        target=1,
        checkpoint_names=["initial", "final"],
        fingerprint_dimensions=32,
    )
    assert evolution.similarity_to_final.shape == (2,)
    assert evolution.figure(mode="interactive_dark").data

    print(
        json.dumps(
            {
                "version": vraven.__version__,
                "layers": len(report.trace.layers),
                "dataset_samples": len(dataset.reports),
                "visual_modes": ["interactive_dark", "publication_light"],
                "status": "validated",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
