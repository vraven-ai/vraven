"""Numerical and semantic tests for formal interval robustness certificates."""

from __future__ import annotations

import copy
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vraven.decompiler.formal_robustness import (
    FormalRobustnessCertificate,
    IntervalBoundVerifier,
)
from vraven.decompiler.verification_protocol import InputDomainSpec


def _affine_classifier() -> nn.Module:
    model = nn.Sequential(nn.Linear(2, 2, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.0, 0.0], [2.0, -1.0]]))
    return model


def test_affine_certified_radius_matches_dual_norm_solution() -> None:
    model = _affine_classifier()
    x = torch.tensor([[1.0, 0.0]])
    certificate = IntervalBoundVerifier(model).certify(
        x,
        target=1,
        epsilon_grid=(0.0, 0.5, 0.7),
        radius_search_steps=24,
    )
    # Exact L-infinity radius is margin / ||w_1-w_0||_1 = 2 / 3.
    assert abs(certificate.certified_radius_lower_bound - (2.0 / 3.0)) < 1e-5
    assert certificate.sample_points[1].certified is True
    assert certificate.sample_points[2].certified is False


def test_interval_bounds_contain_all_box_corners() -> None:
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
    ).eval()
    torch.manual_seed(7)
    for parameter in model.parameters():
        parameter.data.uniform_(-0.7, 0.7)
    x = torch.tensor([[0.2, -0.1]])
    epsilon = 0.15
    result = IntervalBoundVerifier(model).verify_sample(x, epsilon=epsilon)
    assert result.status != "unsupported"
    lower = torch.tensor(result.output_lower_bounds)
    upper = torch.tensor(result.output_upper_bounds)
    for signs in product((-1.0, 1.0), repeat=2):
        corner = x + epsilon * torch.tensor([signs])
        with torch.inference_mode():
            logits = model(corner)[0]
        assert torch.all(logits >= lower - 1e-6)
        assert torch.all(logits <= upper + 1e-6)


def test_functional_residual_arithmetic_contains_all_box_corners() -> None:
    class ResidualClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.readout = nn.Linear(2, 2, bias=False)
            self.readout.weight.data.copy_(torch.eye(2))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            residual = inputs * 0.25
            return self.readout(inputs + residual - 0.1)

    model = ResidualClassifier().double().eval()
    x = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    epsilon = 0.15
    result = IntervalBoundVerifier(model).verify_sample(x, epsilon=epsilon)
    assert result.status != "unsupported"
    lower = torch.tensor(result.output_lower_bounds, dtype=torch.float64)
    upper = torch.tensor(result.output_upper_bounds, dtype=torch.float64)
    for signs in product((-1.0, 1.0), repeat=2):
        corner = x + epsilon * torch.tensor([signs], dtype=torch.float64)
        with torch.inference_mode():
            logits = model(corner)[0]
        assert torch.all(logits >= lower)
        assert torch.all(logits <= upper)


def test_shape_pad_and_axis_reindexing_bounds_contain_all_box_corners() -> None:
    class StructuralClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.readout = nn.Linear(3, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            padded = F.pad(inputs, (0, 1), value=-0.25)
            padded = padded.reshape(padded.shape[0], 3)
            reordered = padded.unsqueeze(1).transpose(1, 2)
            restored = reordered.transpose(1, 2).squeeze(1)
            return self.readout(restored)

    torch.manual_seed(13)
    model = StructuralClassifier().eval()
    x = torch.tensor([[0.2, -0.1]])
    epsilon = 0.15
    result = IntervalBoundVerifier(model).verify_sample(x, epsilon=epsilon)
    assert result.status != "unsupported"
    lower = torch.tensor(result.output_lower_bounds)
    upper = torch.tensor(result.output_upper_bounds)
    for signs in product((-1.0, 1.0), repeat=2):
        corner = x + epsilon * torch.tensor([signs])
        with torch.inference_mode():
            logits = model(corner)[0]
        assert torch.all(logits >= lower - 1e-6)
        assert torch.all(logits <= upper + 1e-6)


def test_tokenwise_linear_bias_uses_last_output_axis() -> None:
    class TokenClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(2, 3)
            self.readout = nn.Linear(3, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.readout(self.projection(inputs).mean(dim=1))

    torch.manual_seed(17)
    model = TokenClassifier().eval()
    x = torch.tensor([[[0.2, -0.1], [0.05, 0.4]]])
    epsilon = 0.1
    result = IntervalBoundVerifier(model).verify_sample(x, epsilon=epsilon)
    assert result.status != "unsupported"
    lower = torch.tensor(result.output_lower_bounds)
    upper = torch.tensor(result.output_upper_bounds)
    for signs in product((-1.0, 1.0), repeat=4):
        corner = x + epsilon * torch.tensor(signs).reshape_as(x)
        with torch.inference_mode():
            logits = model(corner)[0]
        assert torch.all(logits >= lower - 1e-6)
        assert torch.all(logits <= upper + 1e-6)


def test_transformer_frontend_reaches_explicit_layernorm_boundary() -> None:
    class LayerNormTokenClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(2, 4)
            self.position = nn.Parameter(torch.zeros(1, 2, 4))
            self.norm = nn.LayerNorm(4)
            self.readout = nn.Linear(4, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            tokens = self.projection(inputs) + self.position[:, : inputs.shape[1]]
            return self.readout(self.norm(tokens).mean(dim=1))

    model = LayerNormTokenClassifier().eval()
    result = IntervalBoundVerifier(model).verify_sample(torch.randn(1, 2, 2), epsilon=0.01)
    assert result.status == "unsupported"
    assert "LayerNorm" in str(result.unsupported_reason)


def test_supported_cnn_operator_ledger_and_zero_radius() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 2, 3, padding=1),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(8, 2),
    ).eval()
    x = torch.randn(1, 1, 4, 4)
    result = IntervalBoundVerifier(model).verify_sample(x, epsilon=0.0)
    assert result.certified is True
    assert [record.operator for record in result.layer_intervals] == [
        "Conv2d",
        "BatchNorm2d",
        "ReLU",
        "MaxPool2d",
        "Flatten",
        "Linear",
    ]


def test_unsupported_operator_is_visible_not_approximated() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=-1)).eval()
    result = IntervalBoundVerifier(model).verify_sample(torch.tensor([[0.5, -0.3]]), epsilon=0.1)
    assert result.status == "unsupported"
    assert result.certified is False
    assert "Softmax" in str(result.unsupported_reason)


def test_training_dropout_variant_uses_inference_semantics_and_restores_mode() -> None:
    model = nn.Sequential(nn.Dropout1d(0.5), nn.Linear(2, 2)).train()
    result = IntervalBoundVerifier(model).verify_sample(torch.tensor([[0.5, -0.3]]), epsilon=0.1)
    assert result.status != "unsupported"
    assert model.training is True


def test_verification_preserves_parameters_and_dtype() -> None:
    model = _affine_classifier()
    before = copy.deepcopy(model.state_dict())
    IntervalBoundVerifier(model).verify_sample(torch.tensor([[1.0, 0.0]]), epsilon=0.1)
    for name, value in model.state_dict().items():
        assert value.dtype == torch.float32
        assert torch.equal(value, before[name])


def test_cohort_certificate_separates_clean_errors_and_inconclusive() -> None:
    model = _affine_classifier()
    x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([1, 1])  # second sample is clean-wrong
    report = IntervalBoundVerifier(model).certify(
        x[:1],
        target=1,
        epsilon_grid=(0.1,),
        cohort_x=x,
        cohort_y=labels,
        radius_search_steps=4,
    )
    point = report.cohort_points[0]
    assert point.n_samples == 2
    assert point.n_clean_correct == 1
    assert point.n_certified == 1
    assert point.certified_accuracy == 0.5
    assert point.clean_conditional_certified_accuracy == 1.0


def test_certificate_round_trip_outputs_include_both_themes(tmp_path: Path) -> None:
    report = IntervalBoundVerifier(_affine_classifier()).certify(
        torch.tensor([[1.0, 0.0]]),
        target=1,
        epsilon_grid=(0.1, 0.8),
        radius_search_steps=8,
    )
    assert isinstance(report, FormalRobustnessCertificate)
    json_path = report.save(tmp_path / "certificate.json")
    html_path = report.write_html(tmp_path / "certificate.html")
    assert "certified_radius_lower_bound" in json_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "Publication light" in html
    assert "Inconclusive points are not adversarial counterexamples" in html


def test_declared_unbounded_input_semantics_are_distinct_from_missing_bounds() -> None:
    spec = InputDomainSpec(
        lower=None,
        upper=None,
        units="standardised feature units",
        preprocessing="z-score using frozen training statistics",
        semantic_domain="real-valued feature vector",
        source="dataset protocol v1",
    )
    report = IntervalBoundVerifier(_affine_classifier()).certify(
        torch.tensor([[1.0, 0.0]]),
        target=1,
        epsilon_grid=(0.1,),
        radius_search_steps=2,
        input_spec=spec,
    )
    assert report.input_domain == (None, None)
    assert report.input_spec == spec
    assert report.metadata["input_semantics_declared"] is True
