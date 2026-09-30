# Inference benchmarks

This directory measures the current Nerlex reference implementation before any optimized
or native inference backend is introduced.

`inference.py` builds deterministic synthetic decision fixtures for both supported local
compiler families and records:

- raw compiler `predict` latency;
- calibrated prediction latency;
- full local runtime decision latency;
- warm artifact load and validation latency;
- runtime construction/close latency;
- compiler, calibration, and gate artifact sizes;
- snapshot/compile/calibration/gate setup time;
- class/token counts and runner metadata.

The benchmark intentionally uses only Nerlex's existing dependencies. Both normal CI and
the benchmark workflow resolve from the repository's canonical `uv.lock`; there is no
second benchmark-specific dependency lock. The benchmark does not add NumPy, ONNX
Runtime, native extensions, or accelerators.

## Run locally

```bash
uv sync --locked --extra dev --no-install-project --no-build
uv run --no-sync --no-build python benchmarks/inference.py
```

For a shorter exploratory run:

```bash
uv run --no-sync --no-build python benchmarks/inference.py --iterations 100 --warmup 20
```

The output path is intentionally fixed to `benchmark.json` so CI cannot be redirected to
an arbitrary filesystem location through benchmark CLI arguments.

## Interpreting results

Benchmark numbers are specific to the host, Python version, model fixture, and run.
GitHub-hosted runners are shared infrastructure and can be noisy. Treat a single run as a
reproducible baseline and engineering signal, not as a universal performance claim.

The benchmark is designed to answer whether the reference path is materially expensive
enough to justify an additional inference backend. Any optimized backend should be
evaluated against the same fixtures and must preserve prediction/calibration semantics
within an explicitly documented tolerance.
