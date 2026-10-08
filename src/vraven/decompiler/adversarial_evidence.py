"""Statistical evidence and attack-adequacy diagnostics for adversarial audits.

The module contains no attack implementation. It converts measured attack outcomes
into separately defined statistical quantities so attack strength, sampling
uncertainty and robustness across perturbation budgets are not collapsed into one
opaque score.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(slots=True)
class AttackAdequacyAudit:
    """Diagnostics that can reveal, but never rule out, attack failure."""

    input_gradient_l1: float
    input_gradient_l2: float
    input_gradient_linf: float
    finite_gradient_fraction: float
    zero_gradient_fraction: float
    epsilon_monotonicity_violations: list[str] = field(default_factory=list)
    iterative_dominance_violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed_observed_checks(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed_observed_checks"] = self.passed_observed_checks
        return payload


@dataclass(slots=True)
class RobustnessCurvePoint:
    """One empirical robust-accuracy estimate at a declared L∞ budget."""

    epsilon: float
    n_samples: int
    n_correct_clean: int
    n_robust: int
    clean_accuracy: float
    robust_accuracy: float
    conditional_robust_accuracy: float
    attack_success_rate: float
    clean_accuracy_wilson_95: tuple[float, float]
    robust_accuracy_wilson_95: tuple[float, float]
    conditional_robust_accuracy_wilson_95: tuple[float, float]
    monotone_robust_accuracy: float = 0.0
    monotone_n_robust: int = 0
    monotone_robust_accuracy_wilson_95: tuple[float, float] = (0.0, 1.0)
    monotone_conditional_robust_accuracy: float = 0.0
    per_attack_successes: dict[str, int] = field(default_factory=dict)
    per_attack_unique_flips: dict[str, int] = field(default_factory=dict)
    attack_union_curve: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RobustnessProfileReport:
    """Robust accuracy across an ε interval with uncertainty and attack attribution."""

    model_name: str
    protocol: str
    norm: str
    points: list[RobustnessCurvePoint]
    robust_accuracy_auc: float
    clean_normalised_robust_auc: float
    endpoint_robust_accuracy: float
    endpoint_wilson_95: tuple[float, float]
    epsilon_at_50_percent_clean_retention: float | None
    monotonicity_violations: int
    robust_accuracy_auc_bootstrap_95: tuple[float, float] | None = None
    simultaneous_dkw_95_half_width: float | None = None
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    def write_html(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_profile_html(self), encoding="utf-8")
        return destination


def attack_adequacy_audit(
    attacks: Sequence[Any],
    input_gradient: Tensor,
    *,
    tolerance: float = 1e-7,
) -> AttackAdequacyAudit:
    """Evaluate observable indicators of weak or inconsistent attack optimisation.

    The checks are necessary diagnostics, not a proof that an attack is complete.
    A pass means only that the measured trials did not trigger these indicators.
    """

    gradient = input_gradient.detach().float().reshape(-1)
    if gradient.numel():
        finite = torch.isfinite(gradient)
        finite_fraction = float(finite.float().mean().item())
        safe = torch.where(finite, gradient, torch.zeros_like(gradient))
        zero_fraction = float((safe.abs() <= tolerance).float().mean().item())
        grad_l1 = float(torch.linalg.vector_norm(safe, ord=1).item())
        grad_l2 = float(torch.linalg.vector_norm(safe, ord=2).item())
        grad_linf = float(torch.linalg.vector_norm(safe, ord=float("inf")).item())
    else:
        finite_fraction = 0.0
        zero_fraction = 1.0
        grad_l1 = grad_l2 = grad_linf = 0.0

    monotonicity: list[str] = []
    by_family: dict[str, list[Any]] = {}
    for attack in attacks:
        if str(getattr(attack, "norm", "")) != "linf":
            continue
        by_family.setdefault(str(getattr(attack, "attack_family", "unknown")), []).append(attack)
    for family, trials in sorted(by_family.items()):
        ordered = sorted(trials, key=lambda item: float(getattr(item, "epsilon", 0.0)))
        success_seen = False
        for trial in ordered:
            succeeded = bool(getattr(trial, "succeeded", False))
            if success_seen and not succeeded:
                monotonicity.append(
                    f"{family}: success at a smaller ε was followed by failure at "
                    f"ε={float(getattr(trial, 'epsilon', 0.0)):g}"
                )
            success_seen = success_seen or succeeded

    dominance: list[str] = []
    fgsm_by_epsilon = {
        float(getattr(item, "epsilon", 0.0)): item for item in by_family.get("fgsm", ())
    }
    for stronger_family in ("pgd_linf", "apgd_ce"):
        for trial in by_family.get(stronger_family, ()):
            epsilon = float(getattr(trial, "epsilon", 0.0))
            fgsm = fgsm_by_epsilon.get(epsilon)
            if fgsm is None:
                continue
            fgsm_margin = float(getattr(fgsm, "adversarial_margin", float("nan")))
            trial_margin = float(getattr(trial, "adversarial_margin", float("nan")))
            if (
                math.isfinite(fgsm_margin)
                and math.isfinite(trial_margin)
                and trial_margin > fgsm_margin + tolerance
            ):
                dominance.append(
                    f"{stronger_family} reduced the decision margin less than FGSM at ε={epsilon:g}"
                )

    warnings: list[str] = []
    if finite_fraction < 1.0:
        warnings.append("The clean input gradient contains non-finite values.")
    if grad_l2 <= tolerance or zero_fraction >= 0.999:
        warnings.append(
            "The clean input gradient is numerically zero; gradient masking or local flatness may invalidate white-box conclusions."
        )
    if monotonicity:
        warnings.append(
            "Attack success is non-monotone across ε for at least one family; optimisation variance or attack failure may be present."
        )
    if dominance:
        warnings.append(
            "An iterative attack underperformed FGSM on margin reduction for at least one matched ε."
        )
    return AttackAdequacyAudit(
        input_gradient_l1=grad_l1,
        input_gradient_l2=grad_l2,
        input_gradient_linf=grad_linf,
        finite_gradient_fraction=finite_fraction,
        zero_gradient_fraction=zero_fraction,
        epsilon_monotonicity_violations=monotonicity,
        iterative_dominance_violations=dominance,
        warnings=warnings,
    )


def build_robustness_profile(reports: Sequence[Any]) -> RobustnessProfileReport:
    """Construct a worst-case monotone robustness profile from fixed-ε reports.

    Raw estimates remain available at every point. The displayed envelope is the
    cumulative minimum, reflecting the fact that a larger admissible perturbation
    set contains every smaller set. Upward raw fluctuations are counted rather
    than silently interpreted as improved robustness.
    """

    if not reports:
        raise ValueError("at least one fixed-epsilon report is required")
    ordered_reports = sorted(reports, key=lambda report: float(report.epsilon))
    first = ordered_reports[0]
    points: list[RobustnessCurvePoint] = []

    inserted_clean_point = float(first.epsilon) > 0.0
    if inserted_clean_point:
        clean_ci = _interval(first.metadata, "clean_accuracy_wilson_95")
        points.append(
            RobustnessCurvePoint(
                epsilon=0.0,
                n_samples=int(first.n_samples),
                n_correct_clean=int(first.n_correct_clean),
                n_robust=int(first.n_correct_clean),
                clean_accuracy=float(first.clean_accuracy),
                robust_accuracy=float(first.clean_accuracy),
                conditional_robust_accuracy=1.0 if first.n_correct_clean else 0.0,
                attack_success_rate=0.0,
                clean_accuracy_wilson_95=clean_ci,
                robust_accuracy_wilson_95=clean_ci,
                conditional_robust_accuracy_wilson_95=(
                    _interval(first.metadata, "conditional_robust_accuracy_wilson_95")
                    if not first.n_correct_clean
                    else _wilson_all_successes(int(first.n_correct_clean))
                ),
            )
        )

    for report in ordered_reports:
        conditional = float(getattr(report, "conditional_robust_accuracy", report.robust_accuracy))
        points.append(
            RobustnessCurvePoint(
                epsilon=float(report.epsilon),
                n_samples=int(report.n_samples),
                n_correct_clean=int(report.n_correct_clean),
                n_robust=int(report.n_robust),
                clean_accuracy=float(report.clean_accuracy),
                robust_accuracy=float(report.robust_accuracy),
                conditional_robust_accuracy=conditional,
                attack_success_rate=float(report.attack_success_rate),
                clean_accuracy_wilson_95=_interval(report.metadata, "clean_accuracy_wilson_95"),
                robust_accuracy_wilson_95=_interval(report.metadata, "robust_accuracy_wilson_95"),
                conditional_robust_accuracy_wilson_95=_interval(
                    report.metadata, "conditional_robust_accuracy_wilson_95"
                ),
                per_attack_successes=dict(getattr(report, "per_attack_successes", {})),
                per_attack_unique_flips=dict(getattr(report, "per_attack_unique_flips", {})),
                attack_union_curve=list(getattr(report, "attack_union_curve", [])),
            )
        )

    matched = _matched_outcome_matrix(
        ordered_reports,
        include_clean_point=inserted_clean_point,
    )
    running = 1.0
    violations = 0
    previous_raw: float | None = None
    for point_index, point in enumerate(points):
        if previous_raw is not None and point.robust_accuracy > previous_raw + 1e-12:
            violations += 1
        previous_raw = point.robust_accuracy
        if matched is None:
            running = min(running, point.robust_accuracy)
            monotone_count = min(
                point.n_samples,
                max(0, int(round(running * point.n_samples))),
            )
        else:
            monotone_count = int(matched["survival"][:, point_index].sum())
            running = monotone_count / point.n_samples
        point.monotone_n_robust = monotone_count
        point.monotone_robust_accuracy = running
        point.monotone_robust_accuracy_wilson_95 = _wilson_interval(
            monotone_count,
            point.n_samples,
        )
        point.monotone_conditional_robust_accuracy = (
            monotone_count / point.n_correct_clean if point.n_correct_clean else 0.0
        )

    epsilon_span = points[-1].epsilon - points[0].epsilon
    area = 0.0
    for left, right in zip(points, points[1:], strict=False):
        width = right.epsilon - left.epsilon
        area += width * (left.monotone_robust_accuracy + right.monotone_robust_accuracy) / 2.0
    robust_auc = area / epsilon_span if epsilon_span > 0 else points[-1].robust_accuracy
    clean_accuracy = points[0].clean_accuracy
    relative_auc = robust_auc / clean_accuracy if clean_accuracy > 0 else 0.0
    relative_auc = max(0.0, min(1.0, relative_auc))
    epsilon_50 = _epsilon_at_retention(points, target=0.5 * clean_accuracy)
    endpoint = points[-1]
    auc_interval, bootstrap_replicates, bootstrap_seed = _paired_auc_interval(
        ordered_reports,
        points,
    )
    dkw_half_width = math.sqrt(math.log(2.0 / 0.05) / (2.0 * endpoint.n_samples))
    limitations = [
        "The curve is an empirical lower-bound estimate of adversarial risk under the executed attack suite, not certified robustness.",
        "Wilson intervals quantify sampling uncertainty only; they do not include attack incompleteness or threat-model mismatch.",
        "The monotone envelope removes upward finite-sample or optimisation fluctuations but does not strengthen the underlying attacks.",
        "Perturbation norms are mathematical constraints and may not preserve domain validity or functionality.",
    ]
    return RobustnessProfileReport(
        model_name=str(first.model_name),
        protocol=str(first.protocol),
        norm="linf",
        points=points,
        robust_accuracy_auc=float(robust_auc),
        clean_normalised_robust_auc=float(relative_auc),
        endpoint_robust_accuracy=float(endpoint.monotone_robust_accuracy),
        endpoint_wilson_95=endpoint.monotone_robust_accuracy_wilson_95,
        epsilon_at_50_percent_clean_retention=epsilon_50,
        monotonicity_violations=violations,
        robust_accuracy_auc_bootstrap_95=auc_interval,
        simultaneous_dkw_95_half_width=float(dkw_half_width),
        limitations=limitations,
        metadata={
            "robust_accuracy_definition": (
                "number of evaluated samples that are clean-correct and remain unflipped "
                "under the attack union / number of all evaluated samples"
            ),
            "conditional_robust_accuracy_definition": (
                "number remaining unflipped / number clean-correct"
            ),
            "robust_accuracy_auc_definition": (
                "trapezoidal integral of the cumulative-minimum robust-accuracy curve "
                "divided by the evaluated epsilon interval width"
            ),
            "clean_normalised_robust_auc_definition": (
                "robust-accuracy AUC divided by clean accuracy"
            ),
            "epsilon_interval": [points[0].epsilon, points[-1].epsilon],
            "point_count": len(points),
            "sampling_protocol": getattr(first, "sampling_protocol", "unspecified"),
            "selected_indices": list(getattr(first, "selected_indices", [])),
            "per_class_at_first_epsilon": dict(getattr(first, "per_class", {})),
            "per_class_at_endpoint": _monotone_endpoint_per_class(
                matched,
                fallback=dict(getattr(ordered_reports[-1], "per_class", {})),
            ),
            "macro_robust_accuracy_at_first_epsilon": getattr(first, "macro_robust_accuracy", None),
            "evidence_scope": getattr(first, "metadata", {}).get("evidence_scope", "unspecified"),
            "robust_accuracy_auc_bootstrap_95": (
                None if auc_interval is None else list(auc_interval)
            ),
            "paired_bootstrap_replicates": bootstrap_replicates,
            "paired_bootstrap_seed": bootstrap_seed,
            "simultaneous_dkw_95_half_width": float(dkw_half_width),
            "simultaneous_band_definition": (
                "DKW 95% half-width sqrt(log(2/alpha)/(2n)), clipped to [0,1]; "
                "sampling uncertainty conditional on the fixed evaluator"
            ),
            "monotone_envelope_definition": (
                "Per-sample cumulative attack union across all evaluated epsilon budgets; "
                "an adversarial example found at a smaller epsilon remains valid at every "
                "larger budget. Aggregate cumulative minima are used only as a compatibility "
                "fallback when legacy reports lack matched per-sample outcomes."
            ),
            "monotone_envelope_uses_matched_per_sample_union": matched is not None,
            "per_sample_outcomes_by_epsilon": {
                f"{float(report.epsilon):.12g}": list(getattr(report, "per_sample_outcomes", []))
                for report in ordered_reports
            },
        },
    )


def _paired_auc_interval(
    ordered_reports: Sequence[Any],
    points: Sequence[RobustnessCurvePoint],
    *,
    replicates: int = 2000,
    seed: int = 24032026,
) -> tuple[tuple[float, float] | None, int, int]:
    """Paired non-parametric bootstrap interval for robustness-curve AUC."""

    if not ordered_reports or not points:
        return None, 0, int(seed)
    include_clean = len(points) == len(ordered_reports) + 1
    matched = _matched_outcome_matrix(
        ordered_reports,
        include_clean_point=include_clean,
    )
    if matched is None:
        return None, 0, int(seed)
    matrix = matched["survival"].astype(np.float64)
    n = int(matrix.shape[0])
    if n < 2:
        return None, 0, int(seed)
    epsilon_values = [float(point.epsilon) for point in points]
    x = np.asarray(epsilon_values, dtype=np.float64)
    span = float(x[-1] - x[0])
    sample_auc = np.trapezoid(matrix, x=x, axis=1) / span if span > 0 else matrix[:, -1]
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, n, size=(int(replicates), n))
    bootstrap = sample_auc[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return (float(lower), float(upper)), int(replicates), int(seed)


def _matched_outcome_matrix(
    ordered_reports: Sequence[Any],
    *,
    include_clean_point: bool,
) -> dict[str, Any] | None:
    """Return the matched per-sample attack-union survival matrix when available."""

    if not ordered_reports:
        return None
    outcomes = [list(getattr(report, "per_sample_outcomes", [])) for report in ordered_reports]
    if any(not rows for rows in outcomes):
        return None
    identities = [
        [
            (
                int(row["selected_index"]),
                None if row.get("label") is None else int(row["label"]),
            )
            for row in rows
        ]
        for rows in outcomes
    ]
    if any(identity != identities[0] for identity in identities[1:]):
        return None
    n = len(identities[0])
    if n < 1 or any(len(rows) != n for rows in outcomes):
        return None
    clean_columns = [
        np.asarray([bool(row["clean_correct"]) for row in rows], dtype=bool) for rows in outcomes
    ]
    if any(not np.array_equal(column, clean_columns[0]) for column in clean_columns[1:]):
        return None
    columns: list[np.ndarray] = []
    if include_clean_point:
        columns.append(clean_columns[0])
    columns.extend(
        np.asarray([bool(row["robust"]) for row in rows], dtype=bool) for rows in outcomes
    )
    raw = np.stack(columns, axis=1)
    return {
        "identities": identities[0],
        "clean_correct": clean_columns[0],
        "raw": raw,
        "survival": np.logical_and.accumulate(raw, axis=1),
    }


def _monotone_endpoint_per_class(
    matched: dict[str, Any] | None,
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate endpoint class statistics from the per-sample epsilon union."""

    if matched is None:
        return fallback
    identities = list(matched["identities"])
    clean = np.asarray(matched["clean_correct"], dtype=bool)
    endpoint = np.asarray(matched["survival"][:, -1], dtype=bool)
    labels = [identity[1] for identity in identities]
    if any(label is None for label in labels):
        return fallback
    output: dict[str, Any] = {}
    for label in sorted({int(value) for value in labels if value is not None}):
        mask = np.asarray([value == label for value in labels], dtype=bool)
        n = int(mask.sum())
        n_clean = int(np.logical_and(mask, clean).sum())
        n_robust = int(np.logical_and(mask, endpoint).sum())
        output[str(label)] = {
            "n": n,
            "clean": n_clean,
            "robust": n_robust,
            "clean_accuracy": n_clean / n,
            "robust_accuracy": n_robust / n,
            "conditional_robust_accuracy": n_robust / n_clean if n_clean else 0.0,
            "robust_accuracy_wilson_95": list(_wilson_interval(n_robust, n)),
            "aggregation": "per_sample_attack_union_across_epsilon",
        }
    return output


def _interval(metadata: dict[str, Any], key: str) -> tuple[float, float]:
    value = metadata.get(key, (0.0, 1.0))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return 0.0, 1.0


def _wilson_all_successes(trials: int) -> tuple[float, float]:
    return _wilson_interval(trials, trials)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided nominal 95% Wilson interval for a binomial proportion."""

    if trials <= 0 or successes < 0 or successes > trials:
        return 0.0, 1.0
    z = 1.959963984540054
    z2 = z * z
    proportion = successes / trials
    denominator = 1.0 + z2 / trials
    centre = (proportion + z2 / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def _epsilon_at_retention(
    points: Sequence[RobustnessCurvePoint],
    *,
    target: float,
) -> float | None:
    if not points or points[0].monotone_robust_accuracy <= target:
        return points[0].epsilon if points else None
    for left, right in zip(points, points[1:], strict=False):
        y0 = left.monotone_robust_accuracy
        y1 = right.monotone_robust_accuracy
        if y1 > target:
            continue
        if abs(y1 - y0) <= 1e-12:
            return right.epsilon
        fraction = (target - y0) / (y1 - y0)
        return float(left.epsilon + fraction * (right.epsilon - left.epsilon))
    return None


def _format_interval(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"[{float(value[0]):.3f}, {float(value[1]):.3f}]"
    return "—"


def _profile_html(report: RobustnessProfileReport) -> str:
    points = report.points
    width, height = 920.0, 330.0
    left, right, top, bottom = 62.0, 24.0, 24.0, 48.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_epsilon = max((point.epsilon for point in points), default=1.0) or 1.0

    def x_pos(epsilon: float) -> float:
        return left + plot_w * epsilon / max_epsilon

    def y_pos(value: float) -> float:
        return top + plot_h * (1.0 - max(0.0, min(1.0, value)))

    raw_path = " ".join(
        f"{x_pos(point.epsilon):.2f},{y_pos(point.robust_accuracy):.2f}" for point in points
    )
    envelope_path = " ".join(
        f"{x_pos(point.epsilon):.2f},{y_pos(point.monotone_robust_accuracy):.2f}"
        for point in points
    )
    whiskers = "".join(
        (
            f"<line class='ci' x1='{x_pos(point.epsilon):.2f}' "
            f"x2='{x_pos(point.epsilon):.2f}' "
            f"y1='{y_pos(point.monotone_robust_accuracy_wilson_95[0]):.2f}' "
            f"y2='{y_pos(point.monotone_robust_accuracy_wilson_95[1]):.2f}'/>"
            f"<circle class='point' cx='{x_pos(point.epsilon):.2f}' "
            f"cy='{y_pos(point.monotone_robust_accuracy):.2f}' r='4'/>"
        )
        for point in points
    )
    grid = "".join(
        f"<line class='gridline' x1='{left}' x2='{width - right}' y1='{y_pos(level)}' y2='{y_pos(level)}'/>"
        f"<text class='axis-label' x='{left - 10}' y='{y_pos(level) + 4}' text-anchor='end'>{level:.1f}</text>"
        for level in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    point_rows = "".join(
        "<tr>"
        f"<td>{point.epsilon:g}</td>"
        f"<td>{point.clean_accuracy:.3f}</td>"
        f"<td>{point.robust_accuracy:.3f}</td>"
        f"<td>{point.monotone_robust_accuracy:.3f}</td>"
        f"<td>[{point.monotone_robust_accuracy_wilson_95[0]:.3f}, {point.monotone_robust_accuracy_wilson_95[1]:.3f}]</td>"
        f"<td>{point.monotone_conditional_robust_accuracy:.3f}</td>"
        f"<td>{point.attack_success_rate:.3f}</td>"
        f"<td>{point.monotone_n_robust}/{point.n_samples}</td>"
        "</tr>"
        for point in points
    )
    final_union = points[-1].attack_union_curve if points else []
    attack_labels = {
        "apgd_ce": "APGD-CE (VRAVEN-lite)",
        "apgd_dlr": "APGD-DLR (VRAVEN-lite)",
        "fab": "FAB (VRAVEN-lite)",
        "square": "Flat-block query search",
        "pgd_linf": "PGD-L∞",
    }
    union_rows = "".join(
        "<tr>"
        f"<td>{escape(attack_labels.get(str(row.get('attack', '')), str(row.get('attack', '—'))))}</td>"
        f"<td>{escape(str(row.get('raw_successes', 0)))}</td>"
        f"<td>{escape(str(row.get('unique_successes', 0)))}</td>"
        f"<td>{escape(str(row.get('marginal_new_successes', 0)))}</td>"
        f"<td>{escape(str(row.get('cumulative_successes', 0)))}</td>"
        "</tr>"
        for row in final_union
    )
    limitations = "".join(f"<li>{escape(item)}</li>" for item in report.limitations)
    endpoint_classes = report.metadata.get("per_class_at_endpoint", {})
    class_rows = "".join(
        "<tr>"
        f"<td>{escape(str(label))}</td>"
        f"<td>{escape(str(row.get('n', 0)))}</td>"
        f"<td>{float(row.get('clean_accuracy', 0.0)):.3f}</td>"
        f"<td>{float(row.get('robust_accuracy', 0.0)):.3f}</td>"
        f"<td>{float(row.get('conditional_robust_accuracy', 0.0)):.3f}</td>"
        f"<td>{_format_interval(row.get('robust_accuracy_wilson_95'))}</td>"
        "</tr>"
        for label, row in sorted(endpoint_classes.items())
    )
    selected_indices = report.metadata.get("selected_indices", [])
    scope = str(report.metadata.get("evidence_scope", "unspecified"))
    epsilon_50 = (
        "not crossed"
        if report.epsilon_at_50_percent_clean_retention is None
        else f"{report.epsilon_at_50_percent_clean_retention:.4g}"
    )
    auc_interval = (
        "unavailable"
        if report.robust_accuracy_auc_bootstrap_95 is None
        else (
            f"[{report.robust_accuracy_auc_bootstrap_95[0]:.3f}, "
            f"{report.robust_accuracy_auc_bootstrap_95[1]:.3f}]"
        )
    )
    dkw_half_width = (
        "unavailable"
        if report.simultaneous_dkw_95_half_width is None
        else f"±{report.simultaneous_dkw_95_half_width:.3f}"
    )
    return f"""<!doctype html>
<html lang='en' data-theme='dark'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VRAVEN · Robustness profile · {escape(report.model_name)}</title>
<style>
:root{{--bg:#030A12;--panel:#07111F;--raised:#0B1728;--line:#1C344D;--text:#F1F5F9;--muted:#94A3B8;--accent:#00D4FF;--result:#32D74B;--ci:#FFD60A}}
html[data-theme='light']{{--bg:#fff;--panel:#fff;--raised:#f6f8fa;--line:#cbd5e1;--text:#111827;--muted:#526173;--accent:#006b8f;--result:#00796b;--ci:#8a5400}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,Arial,sans-serif}}main{{max-width:1260px;margin:auto;padding:28px}}
header,.result,.panel{{border:1px solid var(--line);background:var(--panel);border-radius:15px}}header{{padding:22px 24px;margin-bottom:16px;display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap}}h1{{font-size:25px;margin:5px 0}}h2{{font-size:15px;margin:0 0 12px}}p{{color:var(--muted);max-width:100ch}}.kicker{{color:var(--accent);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}}
.toggle{{display:flex;gap:4px;padding:4px;border:1px solid var(--line);border-radius:10px;background:var(--raised);height:max-content}}button{{border:0;background:transparent;color:var(--muted);padding:8px 12px;border-radius:7px;font-weight:700}}button.active{{background:var(--text);color:var(--bg)}}
.result{{padding:22px 24px;margin-bottom:16px;border-left:5px solid var(--result);background:linear-gradient(135deg,color-mix(in srgb,var(--result) 9%,var(--panel)),var(--panel) 58%)}}.headline{{font-size:clamp(21px,3vw,31px);font-weight:800;max-width:31ch;line-height:1.2;margin:8px 0}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:18px}}.metric{{border:1px solid var(--line);background:var(--raised);border-radius:9px;padding:11px}}.metric span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}}.metric strong{{font:750 18px ui-monospace,monospace}}
.panel{{padding:18px 20px;margin-bottom:16px;overflow:auto}}svg{{width:100%;height:auto;min-width:620px}}.gridline{{stroke:var(--line);stroke-width:1}}.axis{{stroke:var(--muted);stroke-width:1.2}}.axis-label{{fill:var(--muted);font-size:11px}}.raw{{fill:none;stroke:var(--muted);stroke-width:2;stroke-dasharray:5 4}}.envelope{{fill:none;stroke:var(--result);stroke-width:4}}.ci{{stroke:var(--ci);stroke-width:2}}.point{{fill:var(--accent)}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em}}details{{color:var(--muted)}}footer{{color:var(--muted);font-size:12px;padding:10px 3px 24px}}
</style><script>function setTheme(t){{document.documentElement.dataset.theme=t;document.querySelectorAll('[data-theme-choice]').forEach(b=>b.classList.toggle('active',b.dataset.themeChoice===t))}}document.addEventListener('DOMContentLoaded',()=>{{document.querySelectorAll('[data-theme-choice]').forEach(b=>b.onclick=()=>setTheme(b.dataset.themeChoice));setTheme('dark')}})</script>
</head><body><main><header><div><div class='kicker'>VRAVEN adversarial evidence · {escape(scope)}</div><h1>Robustness–perturbation profile</h1><p>{escape(report.model_name)} · {escape(report.protocol)} · L∞ threat model · n={points[-1].n_samples}</p></div><div class='toggle'><button data-theme-choice='dark'>Dark interactive</button><button data-theme-choice='light'>Publication light</button></div></header>
<section class='result'><div class='kicker'>Primary scientific result</div><div class='headline'>Robustness is measured across the perturbation interval, not at one convenient ε.</div><p>The cumulative worst-case curve retains an average robust accuracy of <strong>{report.robust_accuracy_auc:.3f}</strong> across the tested interval. Relative to clean accuracy, the retained-area fraction is <strong>{report.clean_normalised_robust_auc:.3f}</strong>.</p><div class='metrics'><div class='metric'><span>Robust-accuracy AUC</span><strong>{report.robust_accuracy_auc:.3f}</strong></div><div class='metric'><span>Paired bootstrap AUC 95% CI</span><strong>{auc_interval}</strong></div><div class='metric'><span>Simultaneous DKW 95% half-width</span><strong>{dkw_half_width}</strong></div><div class='metric'><span>Clean-normalised AUC</span><strong>{report.clean_normalised_robust_auc:.3f}</strong></div><div class='metric'><span>Endpoint robust accuracy</span><strong>{report.endpoint_robust_accuracy:.3f}</strong></div><div class='metric'><span>Endpoint Wilson 95% CI</span><strong>[{report.endpoint_wilson_95[0]:.3f}, {report.endpoint_wilson_95[1]:.3f}]</strong></div><div class='metric'><span>50% retention ε</span><strong>{epsilon_50}</strong></div><div class='metric'><span>Raw monotonicity violations</span><strong>{report.monotonicity_violations}</strong></div></div></section>
<section class='panel'><h2>Robust accuracy over ε</h2><svg viewBox='0 0 {width:.0f} {height:.0f}' role='img' aria-label='Robust accuracy curve with Wilson intervals'><line class='axis' x1='{left}' x2='{left}' y1='{top}' y2='{height - bottom}'/><line class='axis' x1='{left}' x2='{width - right}' y1='{height - bottom}' y2='{height - bottom}'/>{grid}<polyline class='raw' points='{raw_path}'/><polyline class='envelope' points='{envelope_path}'/>{whiskers}<text class='axis-label' x='{width / 2}' y='{height - 10}' text-anchor='middle'>L∞ perturbation budget ε</text><text class='axis-label' transform='translate(15 {height / 2}) rotate(-90)' text-anchor='middle'>robust accuracy</text></svg><p>Solid: matched per-sample attack union across all budgets up to ε. Dashed: raw fixed-ε estimates. Amber: pointwise nominal 95% Wilson intervals for the union curve.</p></section>
<section class='panel'><h2>Point estimates and sampling uncertainty</h2><table><thead><tr><th>ε</th><th>Clean acc.</th><th>Raw fixed-ε robust acc.</th><th>Attack-union robust acc.</th><th>Union Wilson 95% CI</th><th>Union conditional survival</th><th>Raw ASR</th><th>Union robust / all</th></tr></thead><tbody>{point_rows}</tbody></table></section>
<section class='panel'><h2>Attack-union contribution at maximum ε</h2><table><thead><tr><th>Attack</th><th>Raw successes</th><th>Unique successes</th><th>Marginal new</th><th>Cumulative union</th></tr></thead><tbody>{union_rows or '<tr><td colspan="5">No attack attribution was recorded.</td></tr>'}</tbody></table></section>
<section class='panel'><h2>Endpoint class balance and uncertainty</h2><table><thead><tr><th>Class</th><th>n</th><th>Clean acc.</th><th>Robust acc.</th><th>Conditional</th><th>Wilson 95% CI</th></tr></thead><tbody>{class_rows or '<tr><td colspan="6">No labelled class strata were recorded.</td></tr>'}</tbody></table><p>Sampling: {escape(str(report.metadata.get("sampling_protocol", "unspecified")))}. Exact selected indices: <code>{escape(json.dumps(selected_indices))}</code>.</p></section>
<section class='panel'><details><summary>Definitions and evidence boundaries</summary><ul>{limitations}</ul><pre>{escape(json.dumps(report.metadata, indent=2))}</pre></details></section><footer>Every point uses the same evaluated cohort and declared attack protocol. Wilson intervals are pointwise; the separately reported DKW band is simultaneous across the fixed curve, conditional on this evaluator.</footer></main></body></html>"""
