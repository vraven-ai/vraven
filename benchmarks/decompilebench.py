"""Run the VRAVEN PyTorch-native planted-mechanism benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vraven.decompiler.decompilebench import run_decompile_bench


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("planted_concept_mlp", "planted_residual_mlp"), default="planted_concept_mlp")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--profile", choices=("quick", "standard"), default="quick")
    parser.add_argument("--output", default="decompilebench_result.json")
    args = parser.parse_args()
    benchmark, result = run_decompile_bench(model_name=args.model, samples=args.samples, profile=args.profile)
    output = Path(args.output)
    output.write_text(json.dumps(benchmark.to_dict(), indent=2), encoding="utf-8")
    result.save(output.with_suffix("").with_name(output.stem + "_evidence"))
    print(json.dumps(benchmark.to_dict(), indent=2))
    return 0 if benchmark.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
