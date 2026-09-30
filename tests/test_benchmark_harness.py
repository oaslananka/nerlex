from __future__ import annotations

import gc

import pytest

from benchmarks import inference as benchmark
from nerlex.compiler import CentroidConfig, MultinomialNBConfig


def test_measure_reports_expected_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((0, 1_000, 2_000, 4_000, 5_000, 8_000, 9_000, 13_000))
    monkeypatch.setattr(benchmark, "perf_counter_ns", lambda: next(ticks))
    gc_was_enabled = gc.isenabled()

    metrics = benchmark._measure(lambda: None, iterations=4, warmup=0)

    assert gc.isenabled() is gc_was_enabled
    assert metrics["iterations"] == 4
    assert metrics["median_us"] == pytest.approx(2.5)
    assert metrics["mean_us"] == pytest.approx(2.5)
    assert metrics["p95_us"] == pytest.approx(4.0)
    assert metrics["min_us"] == pytest.approx(1.0)
    assert metrics["max_us"] == pytest.approx(4.0)
    assert metrics["throughput_per_second_from_median"] == pytest.approx(400_000.0)


def test_measure_handles_zero_timer_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "perf_counter_ns", lambda: 0)

    metrics = benchmark._measure(lambda: None, iterations=2, warmup=0)

    assert metrics["median_us"] == 0.0
    assert metrics["throughput_per_second_from_median"] is None


def test_synthetic_fixture_generation_is_deterministic() -> None:
    spec = benchmark._spec(3)
    first = benchmark._records(
        spec=spec,
        count=24,
        unique_tokens_per_record=2,
    )
    second = benchmark._records(
        spec=spec,
        count=24,
        unique_tokens_per_record=2,
    )

    assert first == second
    assert len(first) == 24
    assert first[0].request.decision_id == "benchmark-routing"
    assert "unique_0_0" in str(first[0].request.state)


@pytest.mark.parametrize(
    "backend",
    (MultinomialNBConfig(), CentroidConfig()),
    ids=("multinomial-nb", "centroid-cosine"),
)
def test_benchmark_scenario_emits_complete_metrics(
    backend: MultinomialNBConfig | CentroidConfig,
) -> None:
    result = benchmark._benchmark_scenario(
        scenario={
            "name": "test",
            "class_count": 2,
            "record_count": 60,
            "unique_tokens_per_record": 2,
        },
        backend=backend,
        iterations=10,
        warmup=1,
    )

    assert result["backend"] == backend.kind
    assert result["scenario"]["class_count"] == 2
    assert result["model"]["class_count"] == 2
    assert result["model"]["token_count"] > 20
    assert result["artifact_bytes"]["total"] > 0

    operations = result["operations"]
    assert set(operations) == {
        "predict",
        "predict_calibrated",
        "runtime_decide",
        "artifact_warm_load",
        "runtime_construct_close",
    }
    for metrics in operations.values():
        assert metrics["iterations"] > 0
        assert metrics["median_us"] >= 0.0
        assert metrics["p95_us"] >= metrics["min_us"]

    representative = result["representative_result"]
    assert representative["route"] == "local"
    assert representative["abstained"] is False


def test_environment_metadata_has_required_shape() -> None:
    environment = benchmark._environment()

    assert environment["python"]
    assert environment["python_implementation"]
    assert environment["platform"]
    assert "cpu_count" in environment
