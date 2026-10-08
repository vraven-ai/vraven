from __future__ import annotations

from vraven.decompiler.decompilebench import run_decompile_bench


def test_planted_benchmark_runs():
    bench, result = run_decompile_bench(samples=48, profile="quick", seed=11)
    assert bench.capture_equivalence is True
    assert bench.operator_coverage >= 0.75
    assert result.programme.rules
    assert 0.0 <= result.programme.fidelity.validation_decision_agreement <= 1.0
