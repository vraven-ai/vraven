"""Tests for the honest decompiler adversarial stress auditor."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vraven.decompiler import (
    AUTOATTACK_SUITE,
    DEFAULT_STRESS_SUITE,
    AdversarialAuditor,
    AdversarialAuditReport,
    AdversarialCertificate,
    RobustAccuracyReport,
    RobustnessProfileReport,
    build_activation_probe,
    build_robustness_profile,
)
from vraven.decompiler.adversarial import (
    AttackResult,
    _cross_entropy_loss,
    _dlr_loss,
    _input_gradient,
    _rank_concept_directions,
)
from vraven.decompiler.adversarial_mechanism import mechanism_instability_radius


class TinyLinear(nn.Module):
    """2-class linear model with a strongly weighted first feature."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[1, 0] = 5.0
            self.fc.weight[0, 0] = -5.0
            self.fc.weight[1, 1] = 0.1
            self.fc.weight[0, 1] = -0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_adversarial_certificate_alias() -> None:
    assert AdversarialCertificate is AdversarialAuditReport


def test_default_suite_covers_literature_attacks() -> None:
    expected = {
        "fgsm",
        "bim",
        "mifgsm",
        "pgd_linf",
        "pgd_l2",
        "apgd_ce",
        "apgd_dlr",
        "fab",
        "deepfool",
        "square",
    }
    assert set(DEFAULT_STRESS_SUITE) == expected
    assert AUTOATTACK_SUITE == ("apgd_ce", "apgd_dlr", "fab", "square")


def test_fgsm_flips_at_large_epsilon_not_tiny() -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[0.2, 0.0, 0.0, 0.0]])
    report = AdversarialAuditor(model).audit_sample(
        x,
        target=1,
        epsilon_grid=(1e-4, 0.5),
        attacks=("fgsm",),
    )
    tiny = next(a for a in report.attacks_tested if a.epsilon == 1e-4)
    large = next(a for a in report.attacks_tested if a.epsilon == 0.5)
    assert tiny.succeeded is False
    assert large.succeeded is True
    assert report.empirical_min_flip_eps_linf == 0.5


def test_pgd_and_bim_in_suite() -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[0.25, 0.0, 0.0, 0.0]])
    report = AdversarialAuditor(model).audit_sample(
        x,
        target=1,
        epsilon_grid=(0.05, 0.3),
        pgd_steps=8,
        attacks=("fgsm", "bim", "pgd_linf", "mifgsm"),
        seed=7,
    )
    families = {a.attack_family for a in report.attacks_tested}
    assert {"fgsm", "bim", "pgd_linf", "mifgsm"} <= families


def test_pgd_restarts_retain_the_strongest_candidate() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 2),
    ).eval()
    torch.manual_seed(19)
    x = torch.tensor([[0.25, -0.1, 0.05, 0.2]])
    target = int(model(x).argmax(dim=1).item())
    one = AdversarialAuditor(model).audit_sample(
        x,
        target=target,
        epsilon_grid=(0.2,),
        pgd_steps=5,
        pgd_restarts=1,
        attacks=("pgd_linf",),
        seed=17,
    )
    many = AdversarialAuditor(model).audit_sample(
        x,
        target=target,
        epsilon_grid=(0.2,),
        pgd_steps=5,
        pgd_restarts=4,
        attacks=("pgd_linf",),
        seed=17,
    )
    assert many.attacks_tested[0].adversarial_margin <= (
        one.attacks_tested[0].adversarial_margin + 1e-7
    )
    assert many.metadata["pgd_restarts"] == 4


def test_pgd_restart_budget_must_be_positive() -> None:
    model = TinyLinear().eval()
    with pytest.raises(ValueError, match="iteration budgets"):
        AdversarialAuditor(model).audit_sample(
            torch.tensor([[0.2, 0.0, 0.0, 0.0]]),
            target=1,
            epsilon_grid=(0.1,),
            attacks=("pgd_linf",),
            pgd_restarts=0,
        )


def test_fab_apgd_and_autoattack() -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[0.3, 0.0, 0.0, 0.0]])
    report = AdversarialAuditor(model).audit_sample(
        x,
        target=1,
        epsilon_grid=(0.25,),
        attacks=("apgd_ce", "apgd_dlr", "fab", "square"),
        pgd_steps=6,
        fab_iters=12,
        fab_restarts=1,
        square_queries=30,
        n_target_classes=1,
    )
    families = {a.attack_family for a in report.attacks_tested}
    assert {"apgd_ce", "apgd_dlr", "fab", "square"} <= families

    aa = AdversarialAuditor(model).audit_autoattack(
        x,
        epsilon=0.3,
        steps=6,
        fab_iters=10,
        fab_restarts=1,
        square_queries=20,
        n_target_classes=1,
        target=1,
    )
    assert aa.metadata["protocol"] == "vraven_ensemble_lite"
    assert "autoattack_robust" in aa.metadata


def test_mechanism_survival_and_instability() -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[0.35, 0.0, 0.0, 0.0]])
    report = AdversarialAuditor(model).audit_sample(
        x,
        target=1,
        epsilon_grid=(0.01, 0.1, 0.5),
        attacks=("fgsm", "bim"),
        pgd_steps=4,
        instability_tau=0.95,
        instability_metric="mechanism_worst_layer_cosine_survival",
    )
    assert report.metadata["audit_layers"] == [
        "robustness_stress",
        "mechanism_failure_analysis",
    ]
    assert "mechanism_survival_curve" in report.metadata
    assert any(a.mechanism_survival is not None for a in report.attacks_tested)
    assert report.metadata["mechanism_instability_metric"] == (
        "mechanism_worst_layer_cosine_survival"
    )
    assert set(report.metadata["mechanism_instability_radii_by_metric"]) == {
        "mechanism_cosine_survival",
        "mechanism_energy_survival",
        "mechanism_top_active_cosine_survival",
        "mechanism_worst_layer_cosine_survival",
    }
    # Soft linear model: survival should drop as ε grows on FGSM.
    fgsm = [a for a in report.attacks_tested if a.attack_family == "fgsm"]
    fgsm_sorted = sorted(fgsm, key=lambda a: a.epsilon)
    if all(a.mechanism_survival is not None for a in fgsm_sorted):
        assert fgsm_sorted[-1].mechanism_survival <= fgsm_sorted[0].mechanism_survival + 1e-6


def test_mechanism_instability_uses_the_declared_component() -> None:
    rows = [
        {
            "epsilon": 0.01,
            "norm": "linf",
            "prediction_flipped": False,
            "mechanism_survival": 0.4,
            "mechanism_cosine_survival": 0.99,
        },
        {
            "epsilon": 0.05,
            "norm": "linf",
            "prediction_flipped": False,
            "mechanism_survival": 0.2,
            "mechanism_cosine_survival": 0.6,
        },
    ]
    assert mechanism_instability_radius(rows, tau=0.7) == 0.01
    assert (
        mechanism_instability_radius(
            rows,
            tau=0.7,
            metric="mechanism_cosine_survival",
        )
        == 0.05
    )
    with pytest.raises(ValueError, match="unknown mechanism survival metric"):
        mechanism_instability_radius(rows, metric="opaque_score")


def test_batch_robust_accuracy() -> None:
    model = TinyLinear().eval()
    x = torch.tensor(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.35, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ]
    )
    y = torch.tensor([1, 1, 1])
    report = AdversarialAuditor(model).audit_batch(
        x,
        y,
        epsilon=0.3,
        protocol="autoattack",
        max_samples=3,
        steps=5,
        fab_iters=8,
        fab_restarts=1,
        square_queries=15,
        n_target_classes=1,
    )
    assert isinstance(report, RobustAccuracyReport)
    assert report.n_samples == 3
    assert 0.0 <= report.robust_accuracy <= 1.0
    assert report.protocol == "vraven_ensemble_lite"
    assert report.sampling_protocol == "stratified"
    assert report.selected_indices


def test_vectorised_batch_pgd_records_constraints_and_restart_union() -> None:
    model = TinyLinear().eval()
    x = torch.tensor(
        [
            [0.40, 0.0, 0.0, 0.0],
            [0.35, 0.0, 0.0, 0.0],
            [-0.40, 0.0, 0.0, 0.0],
            [-0.35, 0.0, 0.0, 0.0],
        ]
    )
    y = torch.tensor([1, 1, 0, 0])
    one = AdversarialAuditor(model).audit_batch(
        x,
        y,
        epsilon=0.3,
        protocol="pgd",
        max_samples=4,
        steps=5,
        pgd_restarts=1,
        seed=31,
    )
    many = AdversarialAuditor(model).audit_batch(
        x,
        y,
        epsilon=0.3,
        protocol="pgd",
        max_samples=4,
        steps=5,
        pgd_restarts=4,
        seed=31,
    )
    diagnostics = many.metadata["attack_diagnostics"]
    assert many.n_robust <= one.n_robust
    assert diagnostics["vectorised"] is True
    assert diagnostics["best_path_retained"] is True
    assert diagnostics["random_start_candidates_assessed"] is True
    assert diagnostics["gradient_sample_steps"] > 0
    assert diagnostics["nonfinite_gradient_sample_steps"] == 0
    assert diagnostics["linf_constraint_satisfied"] is True
    assert diagnostics["domain_constraint_satisfied"] is True
    assert diagnostics["restarts"] == 4


def test_linf_projection_is_inward_rounded_for_large_float32_coordinates() -> None:
    from vraven.decompiler.adversarial import _project_linf

    clean = torch.tensor([[123.456, -987.654]], dtype=torch.float32)
    candidate = clean + torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    epsilon = 0.001
    projected = _project_linf(candidate, clean, epsilon)
    observed = (projected.to(torch.float64) - clean.to(torch.float64)).abs().max().item()
    assert observed <= epsilon


def test_overall_robust_accuracy_counts_clean_errors_as_non_robust() -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[0.4, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]])
    y = torch.tensor([1, 0])  # second sample is intentionally clean-wrong
    report = AdversarialAuditor(model).audit_batch(
        x,
        y,
        epsilon=0.0,
        protocol="pgd",
        max_samples=2,
        sampling="head",
        steps=1,
    )
    assert report.n_correct_clean == 1
    assert report.n_robust == 1
    assert report.robust_accuracy == 0.5
    assert report.conditional_robust_accuracy == 1.0


def test_stratified_selection_covers_ordered_classes() -> None:
    model = TinyLinear().eval()
    x = torch.zeros(8, 4)
    x[:, 0] = torch.tensor([-0.4] * 4 + [0.4] * 4)
    y = torch.tensor([0] * 4 + [1] * 4)
    report = AdversarialAuditor(model).audit_batch(
        x,
        y,
        epsilon=0.0,
        protocol="pgd",
        max_samples=4,
        sampling="stratified",
        steps=1,
        seed=11,
    )
    assert report.metadata["selected_class_counts"] == {"0": 2, "1": 2}
    assert len(report.selected_indices) == 4


def test_cross_entropy_and_dlr_objectives_have_correct_invariances() -> None:
    logits = torch.tensor([[3.0, 1.0, -2.0], [0.5, 2.0, -1.0]])
    shifted = logits + 19.0
    targets = torch.tensor([0, 1])
    ce = torch.stack([_cross_entropy_loss(logits[i : i + 1], int(targets[i])) for i in range(2)])
    shifted_ce = torch.stack(
        [_cross_entropy_loss(shifted[i : i + 1], int(targets[i])) for i in range(2)]
    )
    dlr = torch.stack([_dlr_loss(logits[i : i + 1], int(targets[i])) for i in range(2)])
    shifted_dlr = torch.stack([_dlr_loss(shifted[i : i + 1], int(targets[i])) for i in range(2)])
    assert torch.allclose(ce, shifted_ce)
    assert torch.allclose(dlr, shifted_dlr)
    expected = torch.tensor([-(3.0 - 1.0) / (3.0 - (-2.0)), -(2.0 - 0.5) / (2.0 - (-1.0))])
    assert torch.allclose(dlr.reshape(-1), expected)


def test_input_gradient_does_not_overwrite_parameter_gradients() -> None:
    model = TinyLinear().eval()
    sentinel = torch.full_like(model.fc.weight, 7.0)
    model.fc.weight.grad = sentinel.clone()
    _input_gradient(
        model,
        torch.tensor([[0.2, 0.0, 0.0, 0.0]]),
        original_pred=1,
        output_path=None,
    )
    assert torch.equal(model.fc.weight.grad, sentinel)


def test_activation_probe_freezes_clean_coordinate_basis() -> None:
    layer = nn.Linear(6, 6, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(6))
    model = nn.Sequential(layer)
    probe, names, _kind = build_activation_probe(model, max_units=4)
    clean = probe(torch.tensor([[9.0, 8.0, 7.0, 6.0, 1.0, 0.0]]))
    adversarial = probe(torch.tensor([[1.0, 2.0, 3.0, 4.0, 99.0, 98.0]]))
    assert names == ["0[0]", "0[1]", "0[2]", "0[3]"]
    assert torch.equal(clean, torch.tensor([9.0, 8.0, 7.0, 6.0]))
    assert torch.equal(adversarial, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_robustness_profile_auc_and_monotone_envelope(tmp_path: Path) -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[0.4, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]])
    y = torch.tensor([1, 1])
    profile = AdversarialAuditor(model).audit_robustness_curve(
        x,
        y,
        epsilon_grid=(0.0, 0.1),
        protocol="pgd",
        max_samples=2,
        sampling="head",
        steps=2,
    )
    assert isinstance(profile, RobustnessProfileReport)
    assert 0.0 <= profile.robust_accuracy_auc <= 1.0
    assert profile.robust_accuracy_auc_bootstrap_95 is not None
    assert profile.simultaneous_dkw_95_half_width is not None
    assert profile.metadata["paired_bootstrap_replicates"] == 2000
    assert len(profile.metadata["per_sample_outcomes_by_epsilon"]) == 2
    assert all(
        right.monotone_robust_accuracy <= left.monotone_robust_accuracy
        for left, right in zip(profile.points, profile.points[1:], strict=False)
    )
    html_path = profile.write_html(tmp_path / "profile.html")
    html = html_path.read_text(encoding="utf-8")
    assert "Publication light" in html
    assert "Exact selected indices" in html
    assert "Flat-block query search" not in html  # PGD-only protocol


def test_robustness_curve_uses_per_sample_union_across_epsilon() -> None:
    def report(epsilon: float, robust: list[bool]) -> SimpleNamespace:
        n_robust = sum(robust)
        outcomes = [
            {
                "selected_index": index,
                "label": 1,
                "clean_correct": True,
                "robust": survives,
            }
            for index, survives in enumerate(robust)
        ]
        return SimpleNamespace(
            model_name="matched-union",
            protocol="pgd_linf",
            epsilon=epsilon,
            n_samples=2,
            n_correct_clean=2,
            n_robust=n_robust,
            clean_accuracy=1.0,
            robust_accuracy=n_robust / 2,
            conditional_robust_accuracy=n_robust / 2,
            attack_success_rate=1.0 - n_robust / 2,
            metadata={
                "clean_accuracy_wilson_95": [0.34, 1.0],
                "robust_accuracy_wilson_95": [0.09, 0.91],
                "conditional_robust_accuracy_wilson_95": [0.09, 0.91],
            },
            per_attack_successes={"pgd_linf": 2 - n_robust},
            per_attack_unique_flips={"pgd_linf": 2 - n_robust},
            attack_union_curve=[],
            selected_indices=[0, 1],
            per_class={},
            macro_robust_accuracy=n_robust / 2,
            per_sample_outcomes=outcomes,
            sampling_protocol="explicit_indices",
        )

    profile = build_robustness_profile(
        [
            report(0.1, [True, False]),
            report(0.2, [False, True]),
        ]
    )
    assert [point.n_robust for point in profile.points] == [2, 1, 1]
    assert [point.monotone_n_robust for point in profile.points] == [2, 1, 0]
    assert profile.endpoint_robust_accuracy == 0.0
    assert profile.metadata["monotone_envelope_uses_matched_per_sample_union"] is True
    assert profile.metadata["per_class_at_endpoint"]["1"]["robust"] == 0


def test_report_has_no_fake_scores_and_uses_succeeded() -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    report = AdversarialAuditor(model, model_name="tiny").audit_sample(
        x, epsilon_grid=(0.1,), attacks=("fgsm",)
    )
    payload = report.to_dict()
    assert "gan_immunity_score" not in payload
    assert "diffusion_immunity_score" not in payload
    assert "certified_epsilon_l2" not in payload
    assert "certified_epsilon_linf" not in payload
    assert "empirical_min_flip_eps_linf" in payload
    joined = " ".join(report.limitations).lower()
    assert "certified" in joined or "certificate" in joined
    attack = report.attacks_tested[0]
    assert isinstance(attack, AttackResult)
    assert attack.succeded == attack.succeeded
    assert "succeded" not in payload["attacks_tested"][0]


def test_model_parameters_unchanged() -> None:
    model = TinyLinear().eval()
    before = copy.deepcopy(model.state_dict())
    x = torch.tensor([[1.0, 0.2, -0.1, 0.0]])
    AdversarialAuditor(model).audit_sample(
        x,
        epsilon_grid=(0.05, 0.25),
        pgd_steps=4,
        attacks=("fgsm", "fab", "apgd_ce"),
        fab_iters=6,
        fab_restarts=1,
    )
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_concept_direction_attribution_ranks_planted_direction() -> None:
    delta = torch.tensor([[0.0, 3.0, 0.0, 0.0]])
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    ranked = _rank_concept_directions(
        delta,
        directions,
        concept_names=("alpha", "beta", "gamma"),
        top_k=2,
    )
    assert ranked[0].startswith("beta")


def test_audit_with_concept_directions_end_to_end() -> None:
    model = TinyLinear().eval()
    x = torch.tensor([[0.2, 0.0, 0.0, 0.0]])
    directions = torch.eye(4)
    report = AdversarialAuditor(model).audit_sample(
        x,
        epsilon_grid=(0.5,),
        attacks=("fgsm",),
        concept_directions=directions,
        concept_names=("f0", "f1", "f2", "f3"),
    )
    assert report.most_vulnerable_concept in {"f0", "f1"}
    top_two = {item.split()[0] for item in report.vulnerable_concepts[:2]}
    assert top_two == {"f0", "f1"}
