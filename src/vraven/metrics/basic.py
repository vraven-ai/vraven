"""Low-level reduced tensor statistics used by VRAVEN."""

from __future__ import annotations

import math

import torch


def _sample_for_entropy(value: torch.Tensor, limit: int) -> torch.Tensor:
    flat = value.reshape(-1)
    if flat.numel() <= limit:
        return flat
    # Deterministic evenly spaced sampling avoids introducing a random state.
    indices = torch.linspace(0, flat.numel() - 1, steps=limit, device=flat.device).long()
    return flat.index_select(0, indices)


def histogram_entropy(
    tensor: torch.Tensor,
    *,
    bins: int = 32,
    sample_limit: int = 262_144,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return normalised histogram entropy of arbitrary activations.

    This is activation-dispersion entropy, not predictive uncertainty. The value
    lies in ``[0, 1]`` when numerical precision permits.
    """

    value = _sample_for_entropy(tensor.detach().float(), sample_limit)
    if value.numel() <= 1:
        return torch.zeros((), device=value.device)
    minimum = value.min()
    maximum = value.max()
    if torch.isclose(minimum, maximum):
        return torch.zeros((), device=value.device)
    histogram = torch.histc(value, bins=bins, min=float(minimum), max=float(maximum))
    probability = histogram / histogram.sum().clamp_min(epsilon)
    non_zero = probability > 0
    entropy = -(probability[non_zero] * probability[non_zero].log()).sum()
    return entropy / math.log(float(bins))


def activation_statistics(
    tensor: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    capture_entropy: bool = True,
    histogram_bins: int = 32,
    entropy_sample_limit: int = 262_144,
) -> dict[str, float]:
    """Compute compact activation statistics on the tensor's existing device.

    A single compact tensor is copied to CPU after all reductions. Full activations
    are never transferred by this function.
    """

    if tensor.numel() == 0:
        return {
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "energy": 0.0,
            "mean_energy": 0.0,
            "l1": 0.0,
            "absolute_mean": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "sparsity": 0.0,
            "positive_fraction": 0.0,
            "negative_fraction": 0.0,
            "dispersion_entropy": 0.0,
        }

    value = tensor.detach().float()
    mean = value.mean()
    variance = value.var(unbiased=False)
    std = variance.sqrt()
    squared = value.square()
    absolute = value.abs()
    energy = squared.sum()
    mean_energy = squared.mean()
    l1 = absolute.sum()
    absolute_mean = absolute.mean()
    minimum = value.min()
    maximum = value.max()
    sparsity = absolute.lt(epsilon).float().mean()
    positive_fraction = value.gt(epsilon).float().mean()
    negative_fraction = value.lt(-epsilon).float().mean()
    entropy = (
        histogram_entropy(
            value,
            bins=histogram_bins,
            sample_limit=entropy_sample_limit,
        )
        if capture_entropy
        else torch.full((), float("nan"), device=value.device)
    )
    reduced = torch.stack(
        (
            mean,
            variance,
            std,
            energy,
            mean_energy,
            l1,
            absolute_mean,
            minimum,
            maximum,
            sparsity,
            positive_fraction,
            negative_fraction,
            entropy,
        )
    ).cpu()
    keys = (
        "mean",
        "variance",
        "std",
        "energy",
        "mean_energy",
        "l1",
        "absolute_mean",
        "minimum",
        "maximum",
        "sparsity",
        "positive_fraction",
        "negative_fraction",
        "dispersion_entropy",
    )
    return {key: float(reduced[index]) for index, key in enumerate(keys)}


def gradient_statistics(gradient: torch.Tensor) -> dict[str, float]:
    if gradient.numel() == 0:
        return {"norm": 0.0, "mean_abs": 0.0}
    value = gradient.detach().float()
    reduced = torch.stack((torch.linalg.vector_norm(value), value.abs().mean())).cpu()
    return {"norm": float(reduced[0]), "mean_abs": float(reduced[1])}


def robust_scale(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Median and interquartile scaling with a safe standard-deviation fallback."""

    value = values.float()
    if value.numel() == 0:
        return value
    median = value.median()
    q1 = torch.quantile(value, 0.25)
    q3 = torch.quantile(value, 0.75)
    spread = q3 - q1
    if float(spread.abs()) < epsilon:
        spread = value.std(unbiased=False)
    if float(spread.abs()) < epsilon:
        return value - median
    return (value - median) / spread
