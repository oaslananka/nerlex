from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Any
from uuid import UUID

from nerlex.calibration import (
    TemperatureCalibrationConfig,
    fit_temperature,
    load_calibration_artifact,
    predict_calibrated,
    write_calibration_artifact,
)
from nerlex.compiler import (
    CentroidConfig,
    CompilerArtifact,
    MultinomialNBConfig,
    compile_snapshot,
    load_compiler_artifact,
    predict,
    write_compiler_artifact,
)
from nerlex.dataset import SnapshotConfig, SplitConfig, build_snapshot
from nerlex.evaluation import (
    EmpiricalRiskGateConfig,
    fit_empirical_risk_gate,
    load_gate_artifact,
    write_gate_artifact,
)
from nerlex.runtime import LocalCascadeRuntime, RuntimeBundle
from nerlex.spec import (
    Candidate,
    CandidateMode,
    DecisionKind,
    DecisionRequest,
    DecisionSpec,
    LabelObservation,
    LabelSource,
)
from nerlex.trace import TraceRecord

_BACKENDS = (
    MultinomialNBConfig(),
    CentroidConfig(),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Nerlex reference inference without optional native backends."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Measured calls per operation and scenario.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup calls per operation and scenario.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.iterations < 10:
        raise SystemExit("--iterations must be at least 10.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative.")

    scenarios = (
        {
            "name": "small",
            "class_count": 2,
            "record_count": 160,
            "unique_tokens_per_record": 1,
        },
        {
            "name": "medium",
            "class_count": 8,
            "record_count": 480,
            "unique_tokens_per_record": 4,
        },
    )

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for backend in _BACKENDS:
            results.append(
                _benchmark_scenario(
                    scenario=scenario,
                    backend=backend,
                    iterations=args.iterations,
                    warmup=args.warmup,
                )
            )

    payload = {
        "schema_version": 1,
        "benchmark": "reference-inference-v1",
        "environment": _environment(),
        "parameters": {
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "results": results,
        "notes": [
            "Numbers are host- and run-specific; GitHub-hosted runners are noisy.",
            "Use these results for relative engineering decisions, not universal claims.",
            "No optional native, NumPy, ONNX, or accelerator dependency is used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _benchmark_scenario(
    *,
    scenario: dict[str, int | str],
    backend: MultinomialNBConfig | CentroidConfig,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    name = str(scenario["name"])
    class_count = int(scenario["class_count"])
    record_count = int(scenario["record_count"])
    unique_tokens_per_record = int(scenario["unique_tokens_per_record"])

    spec = _spec(class_count)
    records = _records(
        spec=spec,
        count=record_count,
        unique_tokens_per_record=unique_tokens_per_record,
    )

    snapshot_started = perf_counter_ns()
    snapshot = build_snapshot(
        spec,
        records,
        config=SnapshotConfig(
            split=SplitConfig(
                train=0.60,
                calibration=0.20,
                test=0.20,
                seed=f"benchmark-{name}",
            )
        ),
    )
    snapshot_ms = _elapsed_ms(snapshot_started)

    compile_started = perf_counter_ns()
    compiler = compile_snapshot(snapshot, backend)
    compile_ms = _elapsed_ms(compile_started)

    calibration_started = perf_counter_ns()
    calibration = fit_temperature(
        compiler,
        snapshot,
        config=TemperatureCalibrationConfig(iterations=32),
    )
    calibration_ms = _elapsed_ms(calibration_started)

    gate_started = perf_counter_ns()
    gate = fit_empirical_risk_gate(
        compiler,
        calibration,
        snapshot,
        EmpiricalRiskGateConfig(max_empirical_risk=1.0, min_accepted=1),
    )
    gate_ms = _elapsed_ms(gate_started)

    bundle = RuntimeBundle(
        compiler=compiler,
        calibration=calibration,
        gate=gate,
    )
    query = _query_state(class_index=0, unique_tokens_per_record=unique_tokens_per_record)
    request = DecisionRequest(
        request_id=UUID(int=9_000_000 + class_count),
        decision_id=spec.decision_id,
        spec_version=spec.version,
        state=query,
    )

    with tempfile.TemporaryDirectory(prefix="nerlex-benchmark-") as temp:
        root = Path(temp)
        compiler_path = write_compiler_artifact(compiler, root)
        calibration_path = write_calibration_artifact(calibration, root)
        gate_path = write_gate_artifact(gate, root)

        artifact_bytes = {
            "compiler": compiler_path.stat().st_size,
            "calibration": calibration_path.stat().st_size,
            "gate": gate_path.stat().st_size,
        }
        artifact_bytes["total"] = sum(artifact_bytes.values())

        def load_bundle() -> RuntimeBundle:
            return RuntimeBundle(
                compiler=load_compiler_artifact(compiler_path),
                calibration=load_calibration_artifact(calibration_path),
                gate=load_gate_artifact(gate_path),
            )

        load_metrics = _measure(load_bundle, iterations=max(25, iterations // 10), warmup=5)

        def construct_runtime() -> None:
            runtime = LocalCascadeRuntime(bundle)
            runtime.close()

        construct_metrics = _measure(
            construct_runtime,
            iterations=max(50, iterations // 5),
            warmup=10,
        )

        predict_metrics = _measure(
            lambda: predict(compiler, query),
            iterations=iterations,
            warmup=warmup,
        )
        calibrated_metrics = _measure(
            lambda: predict_calibrated(compiler, calibration, query),
            iterations=iterations,
            warmup=warmup,
        )

        with LocalCascadeRuntime(bundle) as runtime:
            runtime_metrics = _measure(
                lambda: runtime.decide(request),
                iterations=iterations,
                warmup=warmup,
            )
            representative = runtime.decide(request)

    return {
        "scenario": {
            "name": name,
            "class_count": class_count,
            "record_count": record_count,
            "unique_tokens_per_record": unique_tokens_per_record,
        },
        "backend": backend.kind,
        "model": _model_stats(compiler),
        "artifact_bytes": artifact_bytes,
        "setup_ms": {
            "snapshot": snapshot_ms,
            "compile": compile_ms,
            "calibration_fit": calibration_ms,
            "gate_fit": gate_ms,
        },
        "operations": {
            "predict": predict_metrics,
            "predict_calibrated": calibrated_metrics,
            "runtime_decide": runtime_metrics,
            "artifact_warm_load": load_metrics,
            "runtime_construct_close": construct_metrics,
        },
        "representative_result": {
            "route": representative.route.value,
            "abstained": representative.abstained,
            "selected": representative.selected,
            "confidence": representative.confidence,
        },
    }


def _measure(
    operation: Callable[[], object],
    *,
    iterations: int,
    warmup: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        operation()

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        samples: list[int] = []
        for _ in range(iterations):
            started = perf_counter_ns()
            operation()
            samples.append(perf_counter_ns() - started)
    finally:
        if was_enabled:
            gc.enable()

    ordered = sorted(samples)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    median_ns = statistics.median(ordered)
    mean_ns = statistics.fmean(ordered)
    return {
        "iterations": iterations,
        "median_us": median_ns / 1000.0,
        "mean_us": mean_ns / 1000.0,
        "p95_us": ordered[p95_index] / 1000.0,
        "min_us": ordered[0] / 1000.0,
        "max_us": ordered[-1] / 1000.0,
        "throughput_per_second_from_median": 1_000_000_000.0 / median_ns,
    }


def _spec(class_count: int) -> DecisionSpec:
    return DecisionSpec(
        decision_id="benchmark-routing",
        version="1",
        kind=DecisionKind.CHOICE,
        description="Deterministic synthetic benchmark routing decision.",
        candidate_mode=CandidateMode.STATIC,
        candidates=tuple(
            Candidate(
                key=f"class-{index}",
                description=f"Synthetic benchmark class {index}.",
            )
            for index in range(class_count)
        ),
    )


def _records(
    *,
    spec: DecisionSpec,
    count: int,
    unique_tokens_per_record: int,
) -> tuple[TraceRecord, ...]:
    return tuple(
        _trace(
            spec=spec,
            index=index,
            unique_tokens_per_record=unique_tokens_per_record,
        )
        for index in range(count)
    )


def _trace(
    *,
    spec: DecisionSpec,
    index: int,
    unique_tokens_per_record: int,
) -> TraceRecord:
    class_index = index % len(spec.candidates)
    request_id = UUID(int=index + 1)
    unique = " ".join(
        f"unique-{index}-{token_index}"
        for token_index in range(unique_tokens_per_record)
    )
    state = (
        f"classsignal{class_index} classsignal{class_index} "
        f"shared routing benchmark {unique}"
    )
    return TraceRecord(
        request=DecisionRequest(
            request_id=request_id,
            decision_id=spec.decision_id,
            spec_version=spec.version,
            state=state,
        ),
        labels=(
            LabelObservation(
                observation_id=UUID(int=1_000_000 + index),
                request_id=request_id,
                source=LabelSource.OUTCOME,
                value=f"class-{class_index}",
                source_id="reference-benchmark",
            ),
        ),
    )


def _query_state(*, class_index: int, unique_tokens_per_record: int) -> str:
    unknown = " ".join(
        f"query-unknown-{index}"
        for index in range(unique_tokens_per_record)
    )
    return (
        f"classsignal{class_index} classsignal{class_index} "
        f"shared routing benchmark {unknown}"
    )


def _model_stats(artifact: CompilerArtifact) -> dict[str, int]:
    model = artifact.model
    if hasattr(model, "vocabulary"):
        token_count = len(model.vocabulary)
    else:
        token_count = len(model.idf)
    return {
        "class_count": len(model.classes),
        "token_count": token_count,
    }


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "github_sha": os.getenv("GITHUB_SHA"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_runner_name": os.getenv("RUNNER_NAME"),
        "github_runner_os": os.getenv("RUNNER_OS"),
        "github_runner_arch": os.getenv("RUNNER_ARCH"),
    }


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000.0


if __name__ == "__main__":
    main()
