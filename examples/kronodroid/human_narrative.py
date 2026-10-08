"""Turn measured VRAVEN / DarkSide numbers into plain-English “aha” text.

The mathematics stays in the figures. This module only translates what a human
analyst should take away — without inventing causal certainty the metrics do not
support.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _name(feature_names: Sequence[str], index: int) -> str:
    if 0 <= index < len(feature_names):
        return str(feature_names[index])
    return f"feature[{index}]"


def _bucket(name: str) -> str:
    if name.startswith("Nr") or name in {
        "Activities",
        "FilesInsideAPK",
        "CFileSize",
        "UFileSize",
        "TimesSubmitted",
        "NrContactedIps",
        "TotalIntentFilters",
        "total_perm",
    }:
        return "APK structure"
    if name.isupper() or name.startswith(
        ("BIND_", "ACCESS_", "READ_", "WRITE_", "RECEIVE_", "SEND_")
    ):
        return "Android permission"
    if name[:1].islower():
        return "runtime syscall"
    return "other signal"


def story_decision(
    *,
    true_label: str,
    pred_label: str,
    p_malware: float,
    model_name: str = "cnn1d",
) -> str:
    agree = true_label == pred_label
    confidence = (
        "very sure"
        if p_malware >= 0.95 or p_malware <= 0.05
        else "moderately sure"
        if p_malware >= 0.8 or p_malware <= 0.2
        else "somewhat unsure"
    )
    lines = [
        "IN PLAIN ENGLISH — what did the model decide?",
        "",
        f"This app was labelled {true_label} in the dataset.",
        f"The {model_name} model predicted {pred_label} "
        f"with malware probability {_pct(p_malware)} ({confidence}).",
    ]
    if agree:
        lines.append(
            "So far, so good: the prediction matches the label. "
            "The rest of this report shows which signals and internal units carried that call."
        )
    else:
        lines.append(
            "This is a mismatch worth studying: the model and the label disagree. "
            "Use the attribution and counterfactual sections to see what the model latched onto."
        )
    lines.extend(
        [
            "",
            "Think of the model as a very fast checklist reader. It does not “understand” malware "
            "the way an analyst does — it amplifies patterns in syscalls, permissions and APK "
            "structure that, in training, lined up with malware or benign apps.",
        ]
    )
    return "\n".join(lines)


def story_top_features(
    feature_names: Sequence[str],
    mean_abs: Sequence[float],
    focus_abs: Sequence[float],
    order: Sequence[int],
    *,
    top_k: int = 8,
) -> str:
    picks = list(order[:top_k])
    buckets: dict[str, list[str]] = {}
    for idx in picks:
        name = _name(feature_names, int(idx))
        buckets.setdefault(_bucket(name), []).append(name)

    lines = [
        "IN PLAIN ENGLISH — what evidence did it look at?",
        "",
        "When we ask “which inputs moved the malware score?”, these stood out most:",
    ]
    for rank, idx in enumerate(picks, 1):
        name = _name(feature_names, int(idx))
        lines.append(
            f"  {rank}. {name}  ({_bucket(name)})"
            f"  — cohort strength {float(mean_abs[idx]):.4g},"
            f" focus strength {float(focus_abs[idx]):.4g}"
        )
    lines.append("")
    lines.append("Grouped by kind of evidence:")
    for kind, names in buckets.items():
        lines.append(f"  • {kind}: {', '.join(names)}")
    lines.extend(
        [
            "",
            "How to read this: a high score here means “this signal helped push the malware "
            "decision on this sample / cohort”. It is not a legal proof that the app is malicious, "
            "and it is not the only possible explanation — just the clearest measured one.",
        ]
    )
    return "\n".join(lines)


def story_counterfactual(cf: Any, feature_names: Sequence[str]) -> str:
    if cf is None:
        return (
            "IN PLAIN ENGLISH — what would flip the decision?\n\n"
            "No counterfactual was found for this sample. That can mean the malware score is "
            "stubborn under small continuous changes, or the search did not succeed in time."
        )
    flipped = bool(cf.success)
    lines = [
        "IN PLAIN ENGLISH — what would flip the decision?",
        "",
        f"Original prediction class: {cf.original_prediction} → "
        f"after edit: {cf.counterfactual_prediction}"
        f" ({'FLIPPED' if flipped else 'did not fully flip'}).",
        f"Malware score went from {cf.original_target_score:.4g} to "
        f"{cf.counterfactual_target_score:.4g}.",
        "",
        "The smallest measured continuous changes that mattered most were:",
    ]
    for index, delta in zip(cf.top_changed_indices[:8], cf.delta_values[:8], strict=False):
        direction = "increase" if delta > 0 else "decrease"
        lines.append(
            f"  • {_name(feature_names, int(index))}: {direction} by {abs(float(delta)):.4g}"
        )
    lines.extend(
        [
            "",
            "Human takeaway: if changing these few signals is enough to make the model say "
            "“benign”, then those signals are doing real work in the decision. "
            "If almost everything must change, the decision is more distributed.",
        ]
    )
    return "\n".join(lines)


def story_perturbation(result: Any) -> str:
    if result is None:
        return "IN PLAIN ENGLISH — is the explanation brittle?\n\nNo perturbation test available."
    lines = [
        "IN PLAIN ENGLISH — is the explanation brittle?",
        "",
        "We gently noise, mask and shuffle the input and watch whether the malware score holds.",
        "Higher stability ≈ the decision does not collapse under small messiness.",
        "",
    ]
    for curve in result.curves:
        last = float(curve.stability[-1]) if curve.stability else float("nan")
        feel = (
            "sturdy"
            if last >= 0.8
            else "somewhat fragile"
            if last >= 0.5
            else "brittle"
        )
        lines.append(
            f"  • {curve.kind}: ends at {last:.3f} → feels {feel}"
        )
    lines.extend(
        [
            "",
            "Aha: a brittle model can look confident while leaning on a few brittle cues. "
            "A sturdy curve means the same story survives small distortions.",
        ]
    )
    return "\n".join(lines)


def story_circuit(circuit: Any) -> str:
    if circuit is None:
        return "IN PLAIN ENGLISH — what inside the net was enough?\n\nNo sparse circuit was recovered."
    metrics = circuit.metrics if hasattr(circuit, "metrics") else circuit
    retained = int(metrics.get("retained_groups", 0))
    candidates = int(metrics.get("candidate_groups", retained))
    retention = float(metrics.get("score_retention", float("nan")))
    lines = [
        "IN PLAIN ENGLISH — what inside the net was enough?",
        "",
        f"We tested groups of internal units and kept a small core that still preserves "
        f"the malware decision (score retention ≈ {retention:.3f}).",
        f"Kept {retained} of {candidates} candidate groups.",
        "",
        "Aha: you do not need the whole network to tell this story. A thinner “circuit” of "
        "units carries most of the measured decision for this sample. The wiring figure then "
        "shows how those units couple through real weights × activations.",
        "",
        "Caveat: this is an approximate ε-sufficient core, not a unique mathematical brain map.",
    ]
    return "\n".join(lines)


def story_shortcut(shortcut: Any) -> str:
    if shortcut is None:
        return "IN PLAIN ENGLISH — is it using a cheap shortcut?\n\nNo shortcut audit available."
    metrics = shortcut.metrics if hasattr(shortcut, "metrics") else shortcut
    strongest = metrics.get("strongest_group", "?")
    lines = [
        "IN PLAIN ENGLISH — is it using a cheap shortcut?",
        "",
        f"Among declared feature families, the strongest measured lever was: {strongest}.",
        "",
        "Aha: if one family (e.g. permissions alone) can swing the score while others barely "
        "matter, the model may be taking a short path that an adversary can game. "
        "If several families all matter, the story is more balanced.",
    ]
    rows = list((shortcut.tables or {}).get("groups", [])) if hasattr(shortcut, "tables") else []
    if rows:
        lines.append("")
        lines.append("Family effects (absolute relative score drop):")
        for row in sorted(rows, key=lambda r: abs(r.get("relative_score_drop", 0.0)), reverse=True):
            lines.append(
                f"  • {row.get('group')}: |Δ|={abs(float(row.get('relative_score_drop', 0.0))):.4g}"
                f"  flipped={row.get('prediction_flipped')}"
            )
    return "\n".join(lines)


def story_class_field(unit_field: Any) -> str:
    if unit_field is None:
        return (
            "IN PLAIN ENGLISH — do malware and benign look different inside?\n\n"
            "No class-conditional unit field available."
        )
    metrics = unit_field.metrics if hasattr(unit_field, "metrics") else unit_field
    layer = metrics.get("largest_unit_contrast_layer", metrics.get("largest_distribution_shift_layer", "?"))
    flips = metrics.get("sign_flips", metrics.get("reported_unit_flips", "?"))
    return "\n".join(
        [
            "IN PLAIN ENGLISH — do malware and benign look different inside?",
            "",
            f"The strongest internal contrast between classes showed up around layer {layer}.",
            f"Units that reverse sign between classes: {flips}.",
            "",
            "Aha: this is the model’s internal “accent” for malware vs benign. "
            "It does not by itself prove those units cause the label — it shows where the "
            "representations diverge most clearly in this cohort.",
        ]
    )


def story_comparison(comparison: Any) -> str:
    if comparison is None:
        return "IN PLAIN ENGLISH — do two models agree?\n\nNo comparison available."
    s = comparison.summary()
    agree = float(s.get("prediction_agreement_rate", 0.0))
    sim = float(s.get("mean_fingerprint_similarity", 0.0))
    left = s.get("left_name", "model A")
    right = s.get("right_name", "model B")
    feel = (
        "mostly the same story"
        if agree >= 0.9 and sim >= 0.7
        else "overlapping but not identical"
        if agree >= 0.7
        else "quite different internal stories"
    )
    return "\n".join(
        [
            "IN PLAIN ENGLISH — do two models agree?",
            "",
            f"{left} vs {right}: prediction agreement {_pct(agree)}, "
            f"mean reasoning-fingerprint similarity {sim:.3f}.",
            f"That feels like: {feel}.",
            "",
            "Aha: agreement on the label is not the same as agreement on the why. "
            "Two models can both say “malware” while using different internal routes.",
        ]
    )


def story_how_to_read_report() -> str:
    return "\n".join(
        [
            "HOW TO READ THIS REPORT (human guide)",
            "",
            "1. Start with the plain-English cards — they are the “aha” layer.",
            "2. Then open the matching figure if you want the measured evidence.",
            "3. Bars and heatmaps answer: which input signals mattered?",
            "4. Circuit / wiring / anatomy answer: which insides carried the decision?",
            "5. Perturbation / uncertainty answer: would a small mess change the story?",
            "6. Counterfactual answers: what would I need to change to flip the call?",
            "",
            "Remember: VRAVEN measures decision signals inside the network. "
            "It does not claim the network thinks like a person.",
        ]
    )
