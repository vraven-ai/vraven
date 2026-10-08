"""Model-output extraction and prediction summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vraven.core.errors import UnsupportedOutputError


@dataclass(slots=True)
class TargetScoreTrace:
    """Compact prediction view returned by DarkSide intervention helpers."""

    target_score: float
    predicted_index: int | None
    selected_scores: list[float] | None = None
    output_kind: str | None = None


@dataclass(slots=True)
class TargetSelection:
    """Result of selecting a target score from a live model output."""

    trace: TargetScoreTrace
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None


@dataclass(slots=True)
class PredictionComputation:
    logits: torch.Tensor
    score: torch.Tensor
    selected_scores: torch.Tensor
    target_indices: torch.Tensor | None
    output_kind: str
    probabilities: torch.Tensor | None
    predicted_classes: torch.Tensor | None
    confidence: torch.Tensor | None
    entropy: torch.Tensor | None
    margin: torch.Tensor | None


def extract_primary_tensor(output: Any) -> torch.Tensor:
    """Extract logits or the primary tensor from common framework outputs."""

    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, Mapping):
        for key in ("logits", "prediction", "predictions", "output", "scores"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        for value in output.values():
            try:
                return extract_primary_tensor(value)
            except UnsupportedOutputError:
                continue
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return extract_primary_tensor(value)
            except UnsupportedOutputError:
                continue
    raise UnsupportedOutputError(
        "VRAVEN could not identify a tensor output. Return a tensor, a mapping containing "
        "'logits', or an object exposing a tensor .logits attribute."
    )


def _resolve_targets(
    target: int | Sequence[int] | torch.Tensor | None,
    predicted: torch.Tensor,
) -> torch.Tensor:
    batch = predicted.numel()
    if target is None:
        return predicted.detach()
    if isinstance(target, int):
        return torch.full((batch,), target, dtype=torch.long, device=predicted.device)
    if isinstance(target, torch.Tensor):
        targets = target.to(device=predicted.device, dtype=torch.long).reshape(-1)
    else:
        targets = torch.as_tensor(list(target), device=predicted.device, dtype=torch.long).reshape(-1)
    if targets.numel() == 1 and batch > 1:
        targets = targets.repeat(batch)
    if targets.numel() != batch:
        raise ValueError(f"Expected {batch} target values but received {targets.numel()}")
    return targets


def compute_prediction(
    logits: torch.Tensor,
    target: int | Sequence[int] | torch.Tensor | None,
) -> PredictionComputation:
    """Create target scores and calibrated descriptive quantities.

    Softmax and sigmoid values are treated as predictive distributions only for
    conventional classification-shaped outputs. Arbitrary hidden tensors are not
    converted into confidence values.
    """

    if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        raise UnsupportedOutputError("The model output must be a non-empty tensor")

    if logits.ndim == 0:
        scalar = logits.reshape(1)
        return PredictionComputation(
            logits=logits,
            score=scalar.mean(),
            selected_scores=scalar,
            target_indices=None,
            output_kind="regression",
            probabilities=None,
            predicted_classes=None,
            confidence=None,
            entropy=None,
            margin=None,
        )

    if logits.ndim == 1:
        values = logits.reshape(-1)
        probability_one = torch.sigmoid(values)
        probabilities = torch.stack((1.0 - probability_one, probability_one), dim=-1)
        predicted = probability_one.ge(0.5).long()
        targets = _resolve_targets(target, predicted)
        selected = torch.where(targets.eq(1), values, -values)
        entropy = -(
            probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()
        ).sum(dim=1) / torch.log(torch.tensor(2.0, device=logits.device))
        confidence = probabilities.max(dim=1).values
        margin = (probabilities[:, 1] - probabilities[:, 0]).abs()
        return PredictionComputation(
            logits=logits,
            score=selected.mean(),
            selected_scores=selected,
            target_indices=targets,
            output_kind="binary_classification",
            probabilities=probabilities,
            predicted_classes=predicted,
            confidence=confidence,
            entropy=entropy,
            margin=margin,
        )

    if logits.ndim == 2 and logits.shape[-1] == 1:
        return compute_prediction(logits.squeeze(-1), target)

    if logits.ndim != 2:
        raise UnsupportedOutputError(
            "VRAVEN currently supports scalar, [batch], [batch, 1] and [batch, classes] "
            f"outputs. Received shape {tuple(logits.shape)}. Supply a readout function in a "
            "future adapter for segmentation or structured outputs."
        )

    probabilities = torch.softmax(logits, dim=-1)
    predicted = probabilities.argmax(dim=-1)
    targets = _resolve_targets(target, predicted)
    if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
        raise ValueError("A requested target class is outside the model output range")
    selected_logits = logits.gather(1, targets[:, None]).squeeze(1)
    confidence = probabilities.max(dim=1).values
    class_count = logits.shape[-1]
    entropy = -(
        probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()
    ).sum(dim=1)
    if class_count > 1:
        entropy = entropy / torch.log(torch.tensor(float(class_count), device=logits.device))
        top_two = probabilities.topk(k=2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
    else:  # pragma: no cover, handled above
        margin = torch.ones_like(confidence)
    return PredictionComputation(
        logits=logits,
        score=selected_logits.mean(),
        selected_scores=selected_logits,
        target_indices=targets,
        output_kind="multiclass_classification",
        probabilities=probabilities,
        predicted_classes=predicted,
        confidence=confidence,
        entropy=entropy,
        margin=margin,
    )


def select_target(
    output: Any,
    *,
    target: int | Sequence[int] | torch.Tensor | None = None,
    class_dim: int | None = None,
    probability_transform: str | None = None,
    target_reduction: str | None = None,
    class_names: Any = None,
    target_label: Any = None,
    output_path: Any = None,
) -> TargetSelection:
    """Select the target score from a model output.

    Extra keyword arguments are accepted for DarkSide / newer-API compatibility
    and are currently unused by the 0.2.x runtime.
    """

    del class_dim, probability_transform, target_reduction, class_names, target_label, output_path
    logits = extract_primary_tensor(output)
    prediction = compute_prediction(logits, target)
    predicted = prediction.predicted_classes
    predicted_index = (
        None if predicted is None or predicted.numel() == 0 else int(predicted[0].item())
    )
    return TargetSelection(
        trace=TargetScoreTrace(
            target_score=float(prediction.score.detach().cpu()),
            predicted_index=predicted_index,
            selected_scores=prediction.selected_scores.detach().cpu().tolist(),
            output_kind=prediction.output_kind,
        ),
        logits=logits.detach(),
        probabilities=(
            None if prediction.probabilities is None else prediction.probabilities.detach()
        ),
    )
