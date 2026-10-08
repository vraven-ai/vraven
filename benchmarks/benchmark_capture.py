"""Repeatable benchmark for native, fast and gradient execution overhead."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

import vraven


def synchronise(device: torch.device) -> None:
    """Synchronise asynchronous backends before timing boundaries."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_samples(
    callable_object: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
    device: torch.device,
) -> list[float]:
    """Return wall-clock samples after an explicit untimed warm-up."""

    for _ in range(warmup):
        callable_object()
    synchronise(device)

    results: list[float] = []
    for _ in range(repeats):
        synchronise(device)
        started = time.perf_counter()
        callable_object()
        synchronise(device)
        results.append(time.perf_counter() - started)
    return results


def distribution(values: list[float]) -> dict[str, float]:
    """Return latency statistics in milliseconds."""

    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, max(0, round(0.9 * (len(ordered) - 1))))
    return {
        "median_ms": statistics.median(ordered) * 1000.0,
        "mean_ms": statistics.mean(ordered) * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
        "p90_ms": ordered[p90_index] * 1000.0,
    }


def native_inference(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run the fair no-gradient inference baseline."""

    with torch.inference_mode():
        return model(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    for name in ("repeats", "batch", "width"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be zero or greater")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(args.width, args.width * 2),
        torch.nn.GELU(),
        torch.nn.Linear(args.width * 2, args.width),
        torch.nn.ReLU(),
        torch.nn.Linear(args.width, 8),
    ).to(device).eval()
    inputs = torch.randn(args.batch, args.width, device=device)

    callables: dict[str, Callable[[], object]] = {
        "native": lambda: native_inference(model, inputs),
        "vraven_fast": lambda: vraven.explain(model, inputs, mode="fast"),
        "vraven_gradient": lambda: vraven.explain(model, inputs, target=1, mode="gradient"),
    }
    samples = {
        name: timed_samples(
            call,
            repeats=args.repeats,
            warmup=args.warmup,
            device=device,
        )
        for name, call in callables.items()
    }
    metrics = {name: distribution(values) for name, values in samples.items()}
    native_median = metrics["native"]["median_ms"]
    overhead = {
        name: values["median_ms"] / native_median
        for name, values in metrics.items()
        if name != "native"
    }

    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "vraven": vraven.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "configuration": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "batch": args.batch,
            "width": args.width,
            "seed": args.seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "latency": metrics,
        "median_overhead_ratio": overhead,
        "notes": [
            "Native latency uses torch.inference_mode().",
            "Gradient mode includes target-specific reverse-mode differentiation.",
            "Results describe this environment and model only; they are not universal claims.",
        ],
    }

    print(f"device={device}")
    for name, values in metrics.items():
        print(f"{name} median={values['median_ms']:.3f} ms")
    for name, ratio in overhead.items():
        print(f"{name}/native={ratio:.3f}x")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote={args.json_output}")


if __name__ == "__main__":
    main()
