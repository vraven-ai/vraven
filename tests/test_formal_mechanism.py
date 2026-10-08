"""Soundness and semantic tests for formal causal-mechanism certificates."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import torch
import torch.nn as nn

import vraven
from vraven.decompiler.formal_mechanism import (
    CausalEffectPropertySpec,
    CausalEffectSurvivalPoint,
    FormalCausalMechanismVerifier,
    FormalMechanismApplicabilityRecord,
    JointMechanismPoint,
    MechanismPropertySpec,
    VectorSurvivalPoint,
    _cohort_point,
)
from vraven.decompiler.formal_robustness import FormalVerificationPoint
from vraven.decompiler.verification_protocol import InputDomainSpec


def _mechanism_model() -> nn.Sequential:
    model = (
        nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
        )
        model[0].bias.copy_(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64))
        model[2].bias.zero_()
    return model


def _planted_causal_model() -> nn.Sequential:
    model = (
        nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, -0.5], [0.25, 0.5]],
                dtype=torch.float64,
            )
        )
        model[0].bias.copy_(torch.tensor([2.0, 1.5, 0.75], dtype=torch.float64))
        model[2].weight.copy_(
            torch.tensor([[0.0, 0.0, 0.0], [1.2, -0.7, 0.5]], dtype=torch.float64)
        )
        model[2].bias.copy_(torch.tensor([0.0, 3.0], dtype=torch.float64))
    return model


def _margin(logits: torch.Tensor) -> torch.Tensor:
    return logits[..., 1] - logits[..., 0]


def _hidden_and_effects(
    model: nn.Sequential,
    x: torch.Tensor,
    coordinates: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model[1](model[0](x))
    original_margin = _margin(model[2](hidden))[0]
    effects: list[torch.Tensor] = []
    for coordinate in coordinates:
        intervened = hidden.clone()
        intervened[:, coordinate] = 0.0
        effects.append(original_margin - _margin(model[2](intervened))[0])
    return hidden[0], torch.stack(effects)


def test_joint_certificate_proves_three_distinct_properties() -> None:
    report = FormalCausalMechanismVerifier(_mechanism_model()).certify(
        torch.tensor([[1.0, 0.2]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(node_id="1", auto_top_k=2, max_screened_coordinates=3),
        epsilon_grid=(0.0, 0.01, 0.1),
        radius_search_upper=0.2,
        radius_search_steps=8,
    )
    assert report.decision_radius_lower_bound == 0.2
    assert report.representation_geometry_radius_lower_bound > 0
    assert report.representation_top_k_radius_lower_bound > 0
    assert report.representation_radius_lower_bound > 0
    assert report.causal_geometry_radius_lower_bound > 0
    assert report.causal_sign_radius_lower_bound > 0
    assert report.causal_effect_radius_lower_bound > 0
    assert report.joint_radius_lower_bound > 0
    assert report.points[1].decision.certified
    assert report.points[1].representation.certified
    assert report.points[1].causal_effect.certified
    assert report.points[1].jointly_certified


def test_representation_and_causal_effect_bounds_contain_all_box_corners() -> None:
    model = _mechanism_model()
    x = torch.tensor([[1.0, 0.2]], dtype=torch.float64)
    epsilon = 0.1
    report = FormalCausalMechanismVerifier(model).certify(
        x,
        target=1,
        mechanism=MechanismPropertySpec(
            node_id="1",
            top_k=1,
            max_relative_l2_drift=10.0,
            min_cosine=-1.0,
        ),
        causal=CausalEffectPropertySpec(
            node_id="1",
            coordinate_indices=(0, 2),
            max_relative_l2_drift=10.0,
            min_cosine=-1.0,
            min_sign_stable_fraction=0.0,
        ),
        epsilon_grid=(epsilon,),
        radius_search_steps=2,
    )
    point = report.points[0]
    representation_lower = torch.tensor(
        point.representation.coordinate_lower_bounds, dtype=torch.float64
    )
    representation_upper = torch.tensor(
        point.representation.coordinate_upper_bounds, dtype=torch.float64
    )
    effect_lower = torch.tensor(point.causal_effect.effect_lower_bounds, dtype=torch.float64)
    effect_upper = torch.tensor(point.causal_effect.effect_upper_bounds, dtype=torch.float64)
    clean_hidden, clean_effects = _hidden_and_effects(model, x, [0, 2])
    for signs in product((-1.0, 1.0), repeat=2):
        corner = x + epsilon * torch.tensor([signs], dtype=torch.float64)
        hidden, effects = _hidden_and_effects(model, corner, [0, 2])
        assert torch.all(hidden >= representation_lower)
        assert torch.all(hidden <= representation_upper)
        assert torch.all(effects >= effect_lower)
        assert torch.all(effects <= effect_upper)
        actual_rep_cosine = torch.nn.functional.cosine_similarity(clean_hidden, hidden, dim=0)
        actual_effect_cosine = torch.nn.functional.cosine_similarity(clean_effects, effects, dim=0)
        assert float(actual_rep_cosine.detach()) >= float(point.representation.cosine_lower_bound)
        assert float(actual_effect_cosine.detach()) >= float(point.causal_effect.cosine_lower_bound)


def test_relational_effect_bounds_are_exact_for_planted_affine_readout() -> None:
    """Shared-input cancellation recovers the analytical intervention intervals."""

    model = _planted_causal_model()
    centre = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    epsilon = 0.5
    report = FormalCausalMechanismVerifier(model).certify(
        centre,
        target=1,
        mechanism=MechanismPropertySpec(
            node_id="1",
            coordinate_indices=(0, 1, 2),
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            top_k=0,
            require_top_k_stability=False,
        ),
        causal=CausalEffectPropertySpec(
            node_id="1",
            coordinate_indices=(0, 1, 2),
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            min_sign_stable_fraction=1.0,
            minimum_clean_effect=0.1,
        ),
        epsilon_grid=(epsilon,),
        radius_search_upper=1.25,
        radius_search_steps=40,
    )

    point = report.points[0].causal_effect
    expected_lower = torch.tensor([2.04, -1.26, 0.1875], dtype=torch.float64)
    expected_upper = torch.tensor([3.24, -0.91, 0.5625], dtype=torch.float64)
    observed_lower = torch.tensor(point.effect_lower_bounds, dtype=torch.float64)
    observed_upper = torch.tensor(point.effect_upper_bounds, dtype=torch.float64)
    assert torch.all(observed_lower <= expected_lower)
    assert torch.all(observed_upper >= expected_upper)
    assert torch.allclose(observed_lower, expected_lower, atol=1e-12, rtol=0.0)
    assert torch.allclose(observed_upper, expected_upper, atol=1e-12, rtol=0.0)
    assert abs(report.causal_sign_radius_lower_bound - 1.0) < 2e-12
    assert "shared-input relational interval" in report.metadata["effect_bound_method"]


def test_relational_bounds_cancel_large_shared_output_path() -> None:
    """A shared high-gain path must not contaminate the intervention effect."""

    model = (
        nn.Sequential(
            nn.Linear(1, 2),
            nn.Identity(),
            nn.Linear(2, 2),
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0], [100.0]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([2.0, 0.0], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor([[0.0, 50.0], [1.0, 50.0]], dtype=torch.float64))
        model[2].bias.zero_()

    report = FormalCausalMechanismVerifier(model).certify(
        torch.tensor([[0.0]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(
            node_id="1",
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            top_k=0,
            require_top_k_stability=False,
        ),
        causal=CausalEffectPropertySpec(
            node_id="1",
            coordinate_indices=(0,),
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            minimum_clean_effect=0.1,
        ),
        epsilon_grid=(0.5,),
        radius_search_steps=2,
    )
    point = report.points[0].causal_effect
    assert point.effect_lower_bounds[0] <= 1.5
    assert point.effect_upper_bounds[0] >= 2.5
    assert abs(point.effect_lower_bounds[0] - 1.5) < 1e-10
    assert abs(point.effect_upper_bounds[0] - 2.5) < 1e-10
    assert point.checks["sign_stability"] is True


def test_relational_relu_secant_bound_contains_exact_crossing_interval() -> None:
    """The nonlinear relational transformer remains sound across a ReLU boundary."""

    model = (
        nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 2),
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
        model[2].weight.copy_(torch.tensor([[0.5], [2.0]], dtype=torch.float64))
        model[2].bias.copy_(torch.tensor([0.0, 2.0], dtype=torch.float64))

    report = FormalCausalMechanismVerifier(model).certify(
        torch.tensor([[0.2]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(
            node_id="0",
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            top_k=0,
            require_top_k_stability=False,
        ),
        causal=CausalEffectPropertySpec(
            node_id="0",
            coordinate_indices=(0,),
            baseline=0.0,
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            min_sign_stable_fraction=0.0,
            minimum_clean_effect=0.1,
        ),
        epsilon_grid=(0.5,),
        radius_search_steps=2,
    )
    point = report.points[0].causal_effect
    # e(x') = (2.0 - 0.5) ReLU(x'), x' in [-0.3, 0.7].
    assert point.effect_lower_bounds[0] <= 0.0
    assert point.effect_upper_bounds[0] >= 1.05
    assert abs(point.effect_lower_bounds[0]) < 1e-12
    assert abs(point.effect_upper_bounds[0] - 1.05) < 1e-12


def test_relational_residual_arithmetic_contains_dense_box_grid() -> None:
    class ResidualEffectModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(2, 3)
            self.b = nn.Linear(3, 3)
            self.out = nn.Linear(3, 2)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            hidden = torch.relu(self.a(value))
            residual = torch.relu(self.b(hidden) + hidden)
            return self.out(residual * 0.5)

    model = ResidualEffectModel().double().eval()
    with torch.no_grad():
        model.a.weight.copy_(
            torch.tensor([[1.0, -0.2], [0.3, 0.5], [-0.4, 0.8]], dtype=torch.float64)
        )
        model.a.bias.copy_(torch.tensor([0.5, 0.3, 0.7], dtype=torch.float64))
        model.b.weight.copy_(
            torch.tensor(
                [[0.2, 0.1, -0.3], [-0.4, 0.5, 0.2], [0.1, -0.2, 0.6]],
                dtype=torch.float64,
            )
        )
        model.b.bias.copy_(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))
        model.out.weight.copy_(
            torch.tensor([[0.2, -0.3, 0.4], [0.8, 0.1, -0.2]], dtype=torch.float64)
        )
        model.out.bias.copy_(torch.tensor([0.0, 2.0], dtype=torch.float64))

    centre = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    epsilon = 0.15
    report = FormalCausalMechanismVerifier(model).certify(
        centre,
        target=1,
        mechanism=MechanismPropertySpec(
            node_id="relu",
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            top_k=0,
            require_top_k_stability=False,
        ),
        causal=CausalEffectPropertySpec(
            node_id="relu",
            coordinate_indices=(0,),
            max_relative_l2_drift=100.0,
            min_cosine=-1.0,
            min_sign_stable_fraction=0.0,
            minimum_clean_effect=1e-8,
        ),
        epsilon_grid=(epsilon,),
        radius_search_steps=2,
    )
    causal_point = report.points[0].causal_effect
    effect_lower = causal_point.effect_lower_bounds[0]
    effect_upper = causal_point.effect_upper_bounds[0]
    for first in torch.linspace(-epsilon, epsilon, 11, dtype=torch.float64):
        for second in torch.linspace(-epsilon, epsilon, 11, dtype=torch.float64):
            sample = centre + torch.stack((first, second)).reshape(1, 2)
            hidden = torch.relu(model.a(sample))
            original = model.out(torch.relu(model.b(hidden) + hidden) * 0.5)
            intervened_hidden = hidden.clone()
            intervened_hidden[:, 0] = 0.0
            intervened = model.out(torch.relu(model.b(intervened_hidden) + intervened_hidden) * 0.5)
            effect = float(
                (original[:, 1] - original[:, 0] - intervened[:, 1] + intervened[:, 0]).item()
            )
            assert effect_lower <= effect <= effect_upper


def test_auto_selection_uses_intervention_effect_not_activation_alone() -> None:
    model = _mechanism_model()
    with torch.no_grad():
        model[2].weight[:, 0] = 0.0  # largest activation has no output effect
    report = FormalCausalMechanismVerifier(model).certify(
        torch.tensor([[1.0, 0.2]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(node_id="1", auto_top_k=1, max_screened_coordinates=3),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
    )
    assert report.selected_causal_coordinates[0] != 0
    assert abs(report.selected_coordinate_clean_effects[0]) > 0


def test_auto_selection_does_not_silently_shrink_requested_causal_basis() -> None:
    model = _mechanism_model()
    with torch.no_grad():
        model[2].weight[:, 0] = 0.0
        model[2].weight[:, 1] = 0.0
    report = FormalCausalMechanismVerifier(model).certify(
        torch.tensor([[1.0, 0.2]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(
            node_id="1",
            auto_top_k=2,
            max_screened_coordinates=3,
            minimum_clean_effect=1e-6,
        ),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
    )

    assert report.selected_causal_coordinates == [2]
    assert report.causal_coordinate_status == "inconclusive"
    assert report.causal_coordinate_eligible_count == 1
    assert report.causal_coordinate_required_count == 2
    assert "only 1/2" in str(report.causal_coordinate_reason)
    causal_point = report.points[0].causal_effect
    assert causal_point.status == "inconclusive"
    assert not causal_point.certified
    assert causal_point.checks["minimum_clean_effect"] is True
    assert causal_point.checks["required_coordinate_count"] is False
    assert report.causal_effect_radius_lower_bound == 0.0
    assert report.joint_radius_lower_bound == 0.0


def test_zero_effect_predeclared_coordinate_blocks_causal_certificate() -> None:
    model = _mechanism_model()
    with torch.no_grad():
        model[2].weight[:, 0] = 0.0
    report = FormalCausalMechanismVerifier(model).certify(
        torch.tensor([[1.0, 0.2]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(
            node_id="1",
            coordinate_indices=(0, 2),
            minimum_clean_effect=1e-6,
        ),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
    )

    assert report.selected_causal_coordinates == [0, 2]
    assert report.selected_coordinate_clean_effects[0] == 0.0
    assert report.selected_coordinate_clean_effect_lower_bounds[0] == 0.0
    assert report.causal_coordinate_status == "inconclusive"
    assert report.causal_coordinate_eligible_count == 1
    point = report.points[0]
    assert point.representation.certified
    assert point.causal_effect.status == "inconclusive"
    assert point.causal_effect.checks["minimum_clean_effect"] is False
    assert point.causal_effect.eligible_coordinate_count == 1
    assert point.causal_effect.required_coordinate_count == 2
    assert "only 1/2" in str(point.causal_effect.inconclusive_reason)
    assert not point.jointly_certified
    assert report.causal_geometry_radius_lower_bound == 0.0
    assert report.causal_sign_radius_lower_bound == 0.0
    assert report.causal_effect_radius_lower_bound == 0.0
    assert report.joint_radius_lower_bound == 0.0


def test_calibration_selected_coordinate_must_have_local_clean_effect() -> None:
    model = _mechanism_model()
    with torch.no_grad():
        model[2].weight.zero_()
        model[2].weight[1, 0] = 1.0
    report = FormalCausalMechanismVerifier(model).certify(
        torch.tensor([[-1.0, 0.2]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(
            node_id="1",
            auto_top_k=1,
            max_screened_coordinates=3,
            minimum_clean_effect=1e-6,
        ),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
        calibration_x=torch.tensor([[2.0, 0.2]], dtype=torch.float64),
        calibration_disjoint_from_evaluation=True,
    )

    assert report.selected_causal_coordinates == [0]
    assert report.coordinate_selection is not None
    assert report.coordinate_selection.selected_scores[0] >= 1e-6
    assert report.selected_coordinate_clean_effects == [0.0]
    assert report.causal_coordinate_status == "inconclusive"
    point = report.points[0].causal_effect
    assert point.status == "inconclusive"
    assert point.checks["minimum_clean_effect"] is False
    assert not report.points[0].jointly_certified


def test_calibration_split_freezes_coordinates_and_passes_selection_gate() -> None:
    report = FormalCausalMechanismVerifier(_mechanism_model()).certify(
        torch.tensor([[10.0, 0.2]], dtype=torch.float64),
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(node_id="1", auto_top_k=1, max_screened_coordinates=3),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
        input_spec=InputDomainSpec(
            lower=None,
            upper=None,
            units="synthetic units",
            preprocessing="none",
            semantic_domain="R^2 planted affine mechanism",
            source="test fixture",
        ),
        calibration_x=torch.tensor([[0.0, 4.0]], dtype=torch.float64),
        calibration_sample_ids=["calibration:independent-0"],
        calibration_labels=[1],
        calibration_disjoint_from_evaluation=True,
        evaluation_sample_ids=["evaluation:focus-0"],
    )
    assert report.selected_causal_coordinates == [1]
    assert report.coordinate_selection is not None
    assert report.coordinate_selection.mode == "calibration_split"
    assert report.coordinate_selection.strata == {"1": 1}
    checks = {
        check.name: check.verdict
        for check in report.publication_readiness.checks  # type: ignore[union-attr]
    }
    assert checks["declared_input_semantics"] == "pass"
    assert checks["independent_coordinate_selection"] == "pass"
    assert checks["independent_verifier_replay"] == "fail"
    assert report.publication_readiness is not None
    assert report.publication_readiness.status == ("research_evidence_external_validation_required")


def test_frozen_basis_cohort_reports_each_property_separately() -> None:
    model = _mechanism_model()
    cohort = torch.tensor([[1.0, 0.2], [0.8, 0.1]], dtype=torch.float64)
    report = FormalCausalMechanismVerifier(model).certify(
        cohort[:1],
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(node_id="1", coordinate_indices=(0, 2)),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
        cohort_x=cohort,
        cohort_y=[1, 1],
        selected_cohort_indices=[0, 1],
        cohort_epsilon_grid=(0.0, 0.01),
        evaluation_sample_ids=["cohort:0", "cohort:1"],
    )
    assert report.selected_cohort_indices == [0, 1]
    assert len(report.cohort_points) == 2
    zero = report.cohort_points[0]
    assert zero.n_samples == 2
    assert zero.n_clean_correct == 2
    assert zero.n_decision_certified == 2
    assert zero.n_representation_certified == 2
    assert zero.n_causal_effect_certified == 2
    assert zero.n_mechanism_certified == 2
    assert zero.n_jointly_certified == 2
    assert zero.n_decision_and_mechanism == 2
    assert zero.n_decision_only == 0
    assert zero.n_mechanism_only == 0
    assert zero.n_neither == 0
    assert zero.n_representation_applicable == 2
    assert zero.n_causal_applicable == 2
    assert zero.n_mechanism_applicable == 2
    assert zero.n_decision_and_mechanism_applicable == 2
    assert zero.n_applicable_decision_and_mechanism == 2
    assert zero.n_applicable_decision_only == 0
    assert zero.n_applicable_mechanism_only == 0
    assert zero.n_applicable_neither == 0
    assert zero.applicable_joint_certified_fraction == 1.0
    assert [record.sample_id for record in report.cohort_applicability_records] == [
        "cohort:0",
        "cohort:1",
    ]
    assert all(record.basis_epsilon == 0.0 for record in report.cohort_applicability_records)


def test_cohort_records_exact_decision_mechanism_contingency() -> None:
    def vector(certified: bool) -> VectorSurvivalPoint:
        return VectorSurvivalPoint(
            epsilon=0.1,
            status="certified" if certified else "inconclusive",
            certified=certified,
            absolute_l2_drift_upper_bound=0.0,
            clean_norm_lower_bound=1.0,
            relative_l2_drift_upper_bound=0.0,
            cosine_lower_bound=1.0,
            sign_stable_fraction=1.0,
            top_k_stable=True,
        )

    def result(
        *, decision: bool, representation: bool, causal: bool, label: int = 0
    ) -> JointMechanismPoint:
        decision_point = FormalVerificationPoint(
            epsilon=0.1,
            status="certified" if decision else "inconclusive",
            certified=decision,
            clean_prediction=label,
            target=label,
            clean_margin=1.0,
            certified_margin_lower_bound=1.0 if decision else -1.0,
            worst_competitor=1 - label,
        )
        causal_base = vector(causal)
        causal_point = CausalEffectSurvivalPoint(
            **{field: getattr(causal_base, field) for field in causal_base.__dataclass_fields__}
        )
        return JointMechanismPoint(
            epsilon=0.1,
            decision=decision_point,
            representation=vector(representation),
            causal_effect=causal_point,
            jointly_certified=decision and representation and causal,
        )

    def applicability(
        index: int, *, representation: bool = True, causal: bool = True, label: int = 0
    ) -> FormalMechanismApplicabilityRecord:
        representation_reason = None if representation else "representation not applicable"
        causal_reason = None if causal else "causal effect not applicable"
        reasons = [value for value in (representation_reason, causal_reason) if value]
        return FormalMechanismApplicabilityRecord(
            selected_cohort_index=index,
            sample_id=f"sample:{index}",
            label=label,
            basis_epsilon=0.0,
            representation_applicable=representation,
            causal_applicable=causal,
            mechanism_applicable=representation and causal,
            representation_applicability_status=(
                "applicable" if representation else "inconclusive"
            ),
            causal_applicability_status="applicable" if causal else "inconclusive",
            mechanism_applicability_status=(
                "applicable" if representation and causal else "inconclusive"
            ),
            representation_clean_norm_lower_bound=1.0 if representation else 0.0,
            representation_top_k_well_defined=bool(representation),
            causal_eligible_coordinate_count=1 if causal else 0,
            causal_required_coordinate_count=1,
            minimum_clean_effect=1e-8,
            representation_inapplicability_reason=representation_reason,
            causal_inapplicability_reason=causal_reason,
            mechanism_inapplicability_reason="; ".join(reasons) or None,
        )

    point = _cohort_point(
        0.1,
        [
            result(decision=True, representation=True, causal=True),
            result(decision=True, representation=False, causal=True),
            result(decision=False, representation=True, causal=True),
            result(decision=False, representation=True, causal=False),
            result(decision=True, representation=True, causal=False),
            result(decision=False, representation=False, causal=True),
        ],
        [0, 0, 0, 0, 0, 0],
        [
            applicability(0),
            applicability(1),
            applicability(2),
            applicability(3),
            applicability(4, causal=False),
            applicability(5, representation=False),
        ],
    )
    assert point.n_mechanism_certified == 2
    assert point.n_decision_and_mechanism == point.n_jointly_certified == 1
    assert point.n_decision_only == 2
    assert point.n_mechanism_only == 1
    assert point.n_neither == 2
    assert (
        point.n_decision_and_mechanism
        + point.n_decision_only
        + point.n_mechanism_only
        + point.n_neither
        == point.n_samples
    )
    assert point.per_class["0"]["mechanism_certified"] == 2
    assert point.per_class["0"]["decision_and_mechanism"] == 1
    assert point.n_representation_applicable == 5
    assert point.n_causal_applicable == 5
    assert point.n_mechanism_applicable == 4
    assert point.n_decision_and_representation_applicable == 3
    assert point.n_decision_and_causal_applicable == 2
    assert point.n_decision_and_mechanism_applicable == 2
    assert point.n_applicable_decision_and_mechanism == 1
    assert point.n_applicable_decision_only == 1
    assert point.n_applicable_mechanism_only == 1
    assert point.n_applicable_neither == 1
    assert point.applicable_decision_certified_fraction == 0.5
    assert point.applicable_mechanism_certified_fraction == 0.5
    assert point.applicable_joint_certified_fraction == 0.25
    assert point.per_class["0"]["n_causal_applicable"] == 5
    assert point.per_class["0"]["n_mechanism_applicable"] == 4
    assert point.per_class["0"]["n_applicable_neither"] == 1

    no_applicable_denominator = _cohort_point(
        0.1,
        [result(decision=True, representation=True, causal=False)],
        [0],
        [applicability(0, causal=False)],
    )
    assert no_applicable_denominator.n_mechanism_applicable == 0
    assert no_applicable_denominator.n_decision_only == 1
    assert no_applicable_denominator.n_applicable_decision_only == 0
    assert no_applicable_denominator.applicable_decision_certified_fraction is None
    assert no_applicable_denominator.applicable_mechanism_certified_fraction is None
    assert no_applicable_denominator.applicable_joint_certified_fraction is None

    class_point = _cohort_point(
        0.1,
        [
            result(decision=True, representation=True, causal=True, label=0),
            result(decision=True, representation=False, causal=True, label=1),
        ],
        [0, 1],
        [applicability(0, label=0), applicability(1, label=1)],
    )
    assert class_point.per_class["0"]["n_applicable_decision_and_mechanism"] == 1
    assert class_point.per_class["0"]["n_applicable_decision_only"] == 0
    assert class_point.per_class["1"]["n_applicable_decision_and_mechanism"] == 0
    assert class_point.per_class["1"]["n_applicable_decision_only"] == 1


def test_cohort_applicability_is_frozen_at_zero_and_separates_preconditions(
    tmp_path: Path,
) -> None:
    """Analytical clean cases separate top-k ties from zero intervention effects."""

    cohort = torch.tensor(
        [
            [1.0, 0.2],  # unique top-1 and two non-zero causal effects
            [0.0, 0.0],  # non-zero representation, but a three-way top-1 tie
            [-1.0, 1.0],  # unique top-1, but both declared causal effects are zero
        ],
        dtype=torch.float64,
    )
    report = FormalCausalMechanismVerifier(_mechanism_model()).certify(
        cohort[:1],
        target=1,
        mechanism=MechanismPropertySpec(node_id="1", top_k=1),
        causal=CausalEffectPropertySpec(
            node_id="1",
            coordinate_indices=(0, 2),
            minimum_clean_effect=1e-6,
        ),
        epsilon_grid=(0.0,),
        radius_search_steps=2,
        cohort_x=cohort,
        cohort_y=[1, 1, 1],
        selected_cohort_indices=[0, 1, 2],
        cohort_epsilon_grid=(0.0, 0.01),
        evaluation_sample_ids=["applicable", "top-k-tie", "zero-effects"],
    )

    assert len(report.cohort_applicability_records) == 3
    fully_applicable, top_k_tie, zero_effects = report.cohort_applicability_records
    assert fully_applicable.representation_applicable
    assert fully_applicable.causal_applicable
    assert fully_applicable.mechanism_applicable
    assert fully_applicable.mechanism_applicability_status == "applicable"
    assert not top_k_tie.representation_applicable
    assert top_k_tie.causal_applicable
    assert not top_k_tie.mechanism_applicable
    assert top_k_tie.mechanism_applicability_status == "inconclusive"
    assert "top-k" in str(top_k_tie.representation_inapplicability_reason)
    assert zero_effects.representation_applicable
    assert not zero_effects.causal_applicable
    assert not zero_effects.mechanism_applicable
    assert zero_effects.causal_eligible_coordinate_count == 0
    assert "only 0/2" in str(zero_effects.causal_inapplicability_reason)

    assert [point.n_representation_applicable for point in report.cohort_points] == [2, 2]
    assert [point.n_causal_applicable for point in report.cohort_points] == [2, 2]
    assert [point.n_mechanism_applicable for point in report.cohort_points] == [1, 1]
    assert [point.n_decision_and_mechanism_applicable for point in report.cohort_points] == [
        1,
        1,
    ]
    assert all(
        point.n_applicable_decision_and_mechanism
        + point.n_applicable_decision_only
        + point.n_applicable_mechanism_only
        + point.n_applicable_neither
        == point.n_mechanism_applicable
        for point in report.cohort_points
    )
    zero = report.cohort_points[0]
    assert zero.n_decision_certified == 3
    assert zero.n_decision_only == 2  # unconditional coverage retains excluded samples
    assert zero.n_applicable_decision_and_mechanism == 1
    assert zero.n_applicable_decision_only == 0
    assert zero.per_class["1"]["n_causal_applicable"] == 2
    assert zero.per_class["1"]["n_mechanism_applicable"] == 1
    assert sum(zero.representation_inapplicability_reasons.values()) == 1
    assert sum(zero.causal_inapplicability_reasons.values()) == 1

    json_text = report.save(tmp_path / "cohort.json").read_text(encoding="utf-8")
    html_text = report.write_html(tmp_path / "cohort.html").read_text(encoding="utf-8")
    assert '"cohort_applicability_records"' in json_text
    assert '"n_mechanism_applicable": 1' in json_text
    assert '"mechanism_applicability_status": "inconclusive"' in json_text
    assert "Applicability-conditioned survival ledger" in html_text
    assert "top-k-tie" in html_text
    assert "Samples outside A are applicability-inconclusive" in html_text


def test_invalid_node_and_coordinate_fail_explicitly() -> None:
    verifier = FormalCausalMechanismVerifier(_mechanism_model())
    x = torch.tensor([[1.0, 0.2]], dtype=torch.float64)
    try:
        verifier.certify(
            x,
            mechanism=MechanismPropertySpec(node_id="missing"),
            causal=CausalEffectPropertySpec(node_id="1", coordinate_indices=(0,)),
        )
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing mechanism node must fail explicitly")
    try:
        verifier.certify(
            x,
            mechanism=MechanismPropertySpec(node_id="1"),
            causal=CausalEffectPropertySpec(node_id="1", coordinate_indices=(99,)),
        )
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("invalid coordinate must fail explicitly")
    try:
        verifier.certify(
            x,
            mechanism=MechanismPropertySpec(node_id="1"),
            causal=CausalEffectPropertySpec(
                node_id="1",
                coordinate_indices=(0,),
                minimum_clean_effect=0.0,
            ),
        )
    except ValueError as error:
        assert "strictly positive" in str(error)
    else:
        raise AssertionError("zero clean-effect threshold must fail explicitly")


def test_public_api_model_state_and_dual_theme_outputs(tmp_path: Path) -> None:
    model = _mechanism_model().train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    report = vraven.FormalCausalMechanismVerifier(model).certify(
        torch.tensor([[1.0, 0.2]], dtype=torch.float64),
        target=1,
        mechanism=vraven.MechanismPropertySpec(node_id="1", top_k=1),
        causal=vraven.CausalEffectPropertySpec(node_id="1", coordinate_indices=(0, 2)),
        epsilon_grid=(0.0, 0.01),
        radius_search_steps=2,
    )
    assert model.training is True
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    json_path = report.save(tmp_path / "formal_mechanism.json")
    html_path = report.write_html(tmp_path / "formal_mechanism.html")
    assert "joint_radius_lower_bound" in json_path.read_text(encoding="utf-8")
    assert "causal_coordinate_status" in json_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "Publication light" in html
    assert "intervention-effect survival" in html
    assert "inconclusive component is not a counterexample" in html
    assert "Eligibility requires every coordinate" in html
    assert "Applicability-conditioned survival ledger" in html
    assert "applicability-inconclusive" in html
