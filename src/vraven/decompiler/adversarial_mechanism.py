"""Layer-2 adversarial mechanism analysis (ΔM / survival / failure families).

Frozen-M design: build one probe on the clean sample, re-score each
adversarial input. Prefer Reverse concept probes when supplied; otherwise
use an activation fingerprint (no second ``reverse()``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

MechanismProbe = Callable[[Tensor], Tensor]


@dataclass(slots=True)
class MechanismSnapshot:
    """Compact ΔM between clean and adversarial mechanism vectors."""

    survival: float
    delta_l1: float
    delta_linf: float
    suppressed_units: list[str] = field(default_factory=list)
    activated_units: list[str] = field(default_factory=list)
    first_divergence: str | None = None
    probe_kind: str = "activation_fingerprint"
    cosine_survival: float | None = None
    energy_survival: float | None = None
    top_active_cosine_survival: float | None = None
    worst_layer_cosine_survival: float | None = None
    worst_layer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cosine_survival(a: Tensor, b: Tensor) -> float:
    """Mechanism survival S = cos(M(x), M(x_adv))."""

    va = a.detach().float().reshape(-1)
    vb = b.detach().float().reshape(-1)
    if va.numel() == 0 or vb.numel() == 0 or va.numel() != vb.numel():
        return float("nan")
    na = torch.linalg.vector_norm(va)
    nb = torch.linalg.vector_norm(vb)
    if float(na.item()) <= 1e-12 or float(nb.item()) <= 1e-12:
        return float("nan")
    return float((va @ vb / (na * nb)).item())


def _energy_survival(a: Tensor, b: Tensor) -> float:
    """1 − ‖ΔM‖₂ / (‖M‖₂ + ‖M_adv‖₂) — sensitive when cosine is sticky."""

    va = a.detach().float().reshape(-1)
    vb = b.detach().float().reshape(-1)
    denom = (
        torch.linalg.vector_norm(va) + torch.linalg.vector_norm(vb)
    ).clamp_min(1e-12)
    rel = float(torch.linalg.vector_norm(vb - va).item()) / float(denom.item())
    return float(max(0.0, 1.0 - rel))


def survival_components(
    a: Tensor,
    b: Tensor,
    *,
    unit_names: Sequence[str] | None = None,
) -> dict[str, float | str | None]:
    """Return separately interpretable mechanism-survival measurements.

    The conservative survival value is their finite minimum. Keeping each
    component visible prevents a change in vector energy or one layer from being
    misreported as a global angular collapse.
    """

    va = a.detach().float().reshape(-1)
    vb = b.detach().float().reshape(-1)
    global_cosine = cosine_survival(va, vb)
    energy = _energy_survival(va, vb)
    top_active: float | None = None
    if va.numel() >= 4:
        thresh = float(va.abs().median().item())
        mask = va.abs() >= thresh
        if int(mask.sum().item()) >= 2:
            top_active = cosine_survival(va[mask], vb[mask])
    layer_scores: dict[str, float] = {}
    if unit_names is not None and len(unit_names) == int(va.numel()):
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(unit_names):
            layer = str(name).split("[", 1)[0]
            groups.setdefault(layer, []).append(i)
        for layer, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long, device=va.device)
            layer_scores[layer] = cosine_survival(va[idx_t], vb[idx_t])
    finite_layers = {key: value for key, value in layer_scores.items() if math.isfinite(value)}
    worst_layer = min(finite_layers, key=finite_layers.get) if finite_layers else None
    worst_layer_score = finite_layers.get(worst_layer) if worst_layer is not None else None
    candidates = [global_cosine, energy, top_active, worst_layer_score]
    finite = [float(score) for score in candidates if score is not None and math.isfinite(score)]
    return {
        "conservative_survival": float(min(finite)) if finite else float("nan"),
        "cosine_survival": global_cosine,
        "energy_survival": energy,
        "top_active_cosine_survival": top_active,
        "worst_layer_cosine_survival": worst_layer_score,
        "worst_layer": worst_layer,
    }


def sensitive_survival(
    a: Tensor,
    b: Tensor,
    *,
    unit_names: Sequence[str] | None = None,
) -> float:
    """Return the conservative minimum of the declared survival components."""

    return float(
        survival_components(a, b, unit_names=unit_names)["conservative_survival"]
    )


def compare_mechanisms(
    m_clean: Tensor,
    m_adv: Tensor,
    *,
    unit_names: Sequence[str] | None = None,
    top_k: int = 5,
    probe_kind: str = "activation_fingerprint",
) -> MechanismSnapshot:
    """Compute survival and top suppressed / activated coordinates of ΔM."""

    a = m_clean.detach().float().reshape(-1)
    b = m_adv.detach().float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("mechanism vectors must share length")
    delta = b - a
    components = survival_components(a, b, unit_names=unit_names)
    survival = float(components["conservative_survival"])
    delta_l1 = float(delta.abs().sum().item())
    delta_linf = float(delta.abs().max().item()) if delta.numel() else 0.0

    names: list[str]
    if unit_names is not None and len(unit_names) == int(a.numel()):
        names = [str(n) for n in unit_names]
    else:
        names = [f"u{i}" for i in range(int(a.numel()))]

    k = min(int(top_k), int(delta.numel()))
    suppressed: list[str] = []
    activated: list[str] = []
    first = None
    if k > 0:
        vals, idx = torch.topk((-delta), k=k)  # most negative Δ = suppressed
        for i, v in zip(idx.tolist(), vals.tolist(), strict=True):
            if float(v) <= 0:
                break
            suppressed.append(f"{names[i]} (Δ={-float(v):.4g})")
        vals_p, idx_p = torch.topk(delta, k=k)
        for i, v in zip(idx_p.tolist(), vals_p.tolist(), strict=True):
            if float(v) <= 0:
                break
            activated.append(f"{names[i]} (Δ={float(v):.4g})")
        abs_vals, abs_idx = torch.topk(delta.abs(), k=1)
        if float(abs_vals[0].item()) > 0:
            i0 = int(abs_idx[0].item())
            first = names[i0]
    return MechanismSnapshot(
        survival=survival,
        delta_l1=delta_l1,
        delta_linf=delta_linf,
        suppressed_units=suppressed,
        activated_units=activated,
        first_divergence=first,
        probe_kind=probe_kind,
        cosine_survival=float(components["cosine_survival"]),
        energy_survival=float(components["energy_survival"]),
        top_active_cosine_survival=(
            None
            if components["top_active_cosine_survival"] is None
            else float(components["top_active_cosine_survival"])
        ),
        worst_layer_cosine_survival=(
            None
            if components["worst_layer_cosine_survival"] is None
            else float(components["worst_layer_cosine_survival"])
        ),
        worst_layer=(
            None if components["worst_layer"] is None else str(components["worst_layer"])
        ),
    )


def build_activation_probe(
    model: nn.Module,
    *,
    max_units: int = 256,
) -> tuple[MechanismProbe, list[str], str]:
    """Hook Linear/Conv modules and return a mean-pooled activation fingerprint."""

    module_names: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module_names.append((name, module))
    if not module_names:
        # Fallback: flatten input itself as the only "mechanism" coordinate.
        def _input_probe(x: Tensor) -> Tensor:
            return x.detach().float().reshape(-1)

        return _input_probe, ["input"], "input_flatten"

    # Budget units across layers.
    per_layer = max(4, int(max_units) // max(len(module_names), 1))
    unit_names: list[str] = []
    frozen_indices: dict[str, Tensor | None] = {}

    def probe(x: Tensor) -> Tensor:
        buckets: dict[str, Tensor] = {}
        handles = []

        def _make_hook(key: str):
            def _hook(_module: nn.Module, _inp: Any, out: Any) -> None:
                if isinstance(out, tuple):
                    out = out[0]
                if not isinstance(out, Tensor):
                    return
                t = out.detach().float()
                if t.ndim >= 3:
                    # Mean over spatial / sequence dims → [B, C]
                    dims = tuple(range(2, t.ndim))
                    t = t.mean(dim=dims)
                t = t.reshape(t.shape[0], -1)[0]
                if key in frozen_indices:
                    idx = frozen_indices[key]
                    if idx is not None:
                        t = t[idx.to(device=t.device)]
                    buckets[key] = (t, idx)
                elif t.numel() > per_layer:
                    # Select on the first (clean) probe call, then freeze the exact
                    # coordinates for every adversarial comparison.
                    _, idx = torch.topk(t.abs(), k=per_layer)
                    idx, _ = torch.sort(idx)
                    frozen_indices[key] = idx.detach().cpu()
                    t = t[idx]
                    buckets[key] = (t, frozen_indices[key])
                else:
                    frozen_indices[key] = None
                    buckets[key] = (t, None)

            return _hook

        for name, module in module_names:
            handles.append(module.register_forward_hook(_make_hook(name)))
        try:
            model.eval()
            with torch.inference_mode():
                _ = model(x)
        finally:
            for h in handles:
                h.remove()

        parts: list[Tensor] = []
        names_local: list[str] = []
        for name, _module in module_names:
            if name not in buckets:
                continue
            t, idx = buckets[name]
            parts.append(t)
            if idx is None:
                names_local.extend(f"{name}[{i}]" for i in range(int(t.numel())))
            else:
                names_local.extend(f"{name}[{int(i)}]" for i in idx.tolist())
        if not parts:
            return x.detach().float().reshape(-1)
        # Cache names on first call via closure mutation for caller.
        if not unit_names:
            unit_names.extend(names_local)
        return torch.cat(parts, dim=0)

    # Warm-up to populate unit_names with a zero probe is deferred to caller.
    return probe, unit_names, "activation_fingerprint"


def ensure_unit_names(
    probe: MechanismProbe,
    unit_names: list[str],
    x: Tensor,
) -> list[str]:
    """Run probe once so lazy name lists are populated."""

    vec = probe(x)
    if unit_names:
        if len(unit_names) != int(vec.numel()):
            return [f"u{i}" for i in range(int(vec.numel()))]
        return list(unit_names)
    return [f"u{i}" for i in range(int(vec.numel()))]


def mechanism_instability_radius(
    trials: Sequence[dict[str, Any]],
    *,
    tau: float = 0.7,
    metric: str = "mechanism_survival",
) -> float | None:
    """Smallest L∞ ε where one declared survival metric crosses ``tau`` pre-flip.

    ``metric`` must name a visible, machine-readable column in each trial row.
    Keeping the choice explicit avoids treating the historical minimum of
    heterogeneous survival measurements as though it were one natural metric.
    """

    allowed = {
        "mechanism_survival",
        "mechanism_cosine_survival",
        "mechanism_energy_survival",
        "mechanism_top_active_cosine_survival",
        "mechanism_worst_layer_cosine_survival",
    }
    if metric not in allowed:
        raise ValueError(f"unknown mechanism survival metric: {metric}")

    candidates: list[float] = []
    for row in trials:
        if row.get("norm") not in {None, "linf"}:
            continue
        if row.get("prediction_flipped"):
            continue
        surv = row.get(metric)
        eps = row.get("epsilon")
        if surv is None or eps is None:
            continue
        if float(surv) < float(tau):
            candidates.append(float(eps))
    if not candidates:
        return None
    return min(candidates)


def survival_margin_rows(
    attacks: Sequence[Any],
) -> list[dict[str, Any]]:
    """Aggregate per-(family, ε) survival and margin for the report table."""

    best: dict[tuple[str, float], dict[str, Any]] = {}
    for attack in attacks:
        fam = str(getattr(attack, "attack_family", ""))
        eps = float(getattr(attack, "epsilon", 0.0))
        key = (fam, eps)
        survival = getattr(attack, "mechanism_survival", None)
        row = {
            "family": fam,
            "epsilon": eps,
            "margin": float(getattr(attack, "adversarial_margin", float("nan"))),
            "mechanism_survival": None if survival is None else float(survival),
            "mechanism_cosine_survival": getattr(
                attack, "mechanism_cosine_survival", None
            ),
            "mechanism_energy_survival": getattr(
                attack, "mechanism_energy_survival", None
            ),
            "mechanism_top_active_cosine_survival": getattr(
                attack, "mechanism_top_active_cosine_survival", None
            ),
            "mechanism_worst_layer_cosine_survival": getattr(
                attack, "mechanism_worst_layer_cosine_survival", None
            ),
            "mechanism_worst_layer": getattr(attack, "mechanism_worst_layer", None),
            "flipped": bool(getattr(attack, "succeeded", False)),
            "norm": str(getattr(attack, "norm", "linf")),
            "prediction_flipped": bool(getattr(attack, "prediction_flipped", False)),
        }
        prev = best.get(key)
        if prev is None:
            best[key] = row
        else:
            # Prefer the lower survival (worst mechanism) at that cell.
            ps = prev.get("mechanism_survival")
            cs = row.get("mechanism_survival")
            if cs is not None and (ps is None or float(cs) < float(ps)):
                best[key] = row
    return sorted(best.values(), key=lambda r: (r["family"], r["epsilon"]))


def cross_attack_agreement(
    family_deltas: dict[str, Tensor],
) -> dict[str, float]:
    """Pairwise cosine similarity of ΔM vectors across attack families."""

    families = sorted(family_deltas)
    out: dict[str, float] = {}
    for i, fa in enumerate(families):
        for fb in families[i + 1 :]:
            da = family_deltas[fa].detach().float().reshape(-1)
            db = family_deltas[fb].detach().float().reshape(-1)
            if da.numel() != db.numel():
                raise ValueError(
                    "attack-family mechanism deltas must share an identical frozen basis"
                )
            if da.numel() == 0:
                continue
            na = torch.linalg.vector_norm(da).clamp_min(1e-12)
            nb = torch.linalg.vector_norm(db).clamp_min(1e-12)
            sim = float((da @ db / (na * nb)).item())
            out[f"{fa}↔{fb}"] = sim
    return out


def cluster_failure_families(
    family_deltas: dict[str, Tensor],
    *,
    threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Deterministic complete-link clusters of attack-family mechanism deltas."""

    families = sorted(family_deltas)
    if not families:
        return []
    assigned: set[str] = set()
    clusters: list[dict[str, Any]] = []
    agreement = cross_attack_agreement(family_deltas)
    for fam in families:
        if fam in assigned:
            continue
        members = [fam]
        assigned.add(fam)
        for other in families:
            if other in assigned:
                continue
            similarities = [
                agreement.get("↔".join(sorted((member, other)))) for member in members
            ]
            if similarities and all(
                value is not None and float(value) >= float(threshold)
                for value in similarities
            ):
                members.append(other)
                assigned.add(other)
        within = [
            agreement["↔".join(sorted((left, right)))]
            for i, left in enumerate(members)
            for right in members[i + 1 :]
            if "↔".join(sorted((left, right))) in agreement
        ]
        clusters.append(
            {
                "family_id": f"F{len(clusters) + 1}",
                "members": members,
                "n_members": len(members),
                "minimum_pairwise_cosine": min(within) if within else None,
                "mean_pairwise_cosine": float(np.mean(within)) if within else None,
                "clustering_rule": f"complete-link cosine >= {float(threshold):g}",
                "note": (
                    "Attacks that produce similar internal ΔM "
                    "(same weakness cluster)."
                ),
            }
        )
    return clusters


def failure_mode_spectrum(family_deltas: dict[str, Tensor]) -> dict[str, Any]:
    """Quantify the dimensionality of successful attack-family mechanism changes.

    Rows are L2-normalised ΔM vectors. If ``p`` contains the normalised squared
    singular values, effective failure modes are ``exp(-sum(p log p))``. The
    value lies in ``[1, rank]`` and is not a causal-mechanism count.
    """

    families = sorted(family_deltas)
    if not families:
        return {
            "families": [],
            "rank": 0,
            "effective_failure_modes": 0.0,
            "dominant_mode_energy_fraction": None,
            "spectrum": [],
        }
    width = min(int(family_deltas[family].numel()) for family in families)
    if width <= 0:
        return {
            "families": families,
            "rank": 0,
            "effective_failure_modes": 0.0,
            "dominant_mode_energy_fraction": None,
            "spectrum": [],
        }
    rows = []
    for family in families:
        row = family_deltas[family].detach().float().reshape(-1)[:width]
        row = row / torch.linalg.vector_norm(row).clamp_min(1e-12)
        rows.append(row)
    # MPS does not currently implement SVD.  The family matrix is a small
    # diagnostic object, so evaluate its spectrum on CPU for portable,
    # deterministic reporting without changing the attack/model device.
    matrix = torch.stack(rows, dim=0).cpu()
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(1e-12)
    positive = probabilities[probabilities > 1e-12]
    entropy = -torch.sum(positive * torch.log(positive))
    effective = float(torch.exp(entropy).item()) if positive.numel() else 0.0
    return {
        "families": families,
        "rank": int(positive.numel()),
        "effective_failure_modes": effective,
        "dominant_mode_energy_fraction": (
            float(probabilities.max().item()) if probabilities.numel() else None
        ),
        "spectrum": [float(value) for value in probabilities.tolist()],
        "definition": (
            "exp(Shannon entropy) of normalised squared singular values of the "
            "L2-normalised attack-family mechanism-delta matrix"
        ),
    }


def build_concept_probe_from_directions(
    directions: Tensor | np.ndarray,
    *,
    concept_names: Sequence[str] | None = None,
) -> tuple[MechanismProbe, list[str], str]:
    """Project flattened input onto frozen concept / feature directions."""

    dirs = torch.as_tensor(directions, dtype=torch.float32)
    if dirs.ndim == 1:
        dirs = dirs.unsqueeze(0)
    dirs = dirs.reshape(dirs.shape[0], -1)
    norms = torch.linalg.vector_norm(dirs, ord=2, dim=1).clamp_min(1e-12)
    dirs = dirs / norms.unsqueeze(1)
    names = (
        [str(n) for n in concept_names]
        if concept_names is not None and len(concept_names) == dirs.shape[0]
        else [f"C{i}" for i in range(dirs.shape[0])]
    )

    def probe(x: Tensor) -> Tensor:
        flat = x.detach().float().reshape(-1)
        if flat.numel() != dirs.shape[1]:
            raise ValueError(
                f"concept probe width {dirs.shape[1]} != input width {flat.numel()}"
            )
        return dirs @ flat

    return probe, names, "concept_directions"
