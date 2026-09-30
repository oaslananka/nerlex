# Reference inference baseline — 2026-09-30

This document preserves the first reproducible performance baseline produced by
`benchmarks/inference.py` after PR #14.

The benchmark is evidence for engineering decisions, not a universal performance claim or
an SLO. GitHub-hosted runners are shared infrastructure, so repeated attempts can vary
materially.

## Environment

- Python: 3.12.14
- Architecture: x86_64
- Reported CPU count: 4
- Runner: GitHub-hosted Ubuntu
- Benchmark workflow run: `36667951903`
- Source PR: #14
- Two consecutive attempts were run against the same PR merge ref.
- Benchmark dependencies were exact-version, wheel-hash locked.
- No NumPy, ONNX Runtime, native extension, or accelerator dependency was used.

## Fixtures

| Scenario | Classes | Token count | Records |
| --- | ---: | ---: | ---: |
| small | 2 | 102 | 160 |
| medium | 8 | 1,123 | 480 |

Synthetic request and label timestamps are fixed so fixture construction is deterministic.

## Artifact sizes

| Scenario | Backend | Compiler + calibration + gate |
| --- | --- | ---: |
| small | multinomial NB | 10,600 bytes |
| small | cosine centroid | 9,092 bytes |
| medium | multinomial NB | 322,813 bytes |
| medium | cosine centroid | 78,575 bytes |

## Two-attempt latency ranges

All values below are per-operation medians or p95 values from two consecutive benchmark
attempts on the same GitHub workflow run.

| Scenario | Backend | raw predict median | calibrated median | full runtime median | full runtime p95 | warm artifact load median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small | multinomial NB | 9.0–11.2 µs | 16.3–20.4 µs | 22.5–28.6 µs | 24.0–30.0 µs | 0.80–1.04 ms |
| small | cosine centroid | 11.7–15.1 µs | 18.6–24.6 µs | 24.6–32.2 µs | 27.6–37.1 µs | 0.77–1.03 ms |
| medium | multinomial NB | 16.8–22.1 µs | 28.0–36.7 µs | 34.6–45.2 µs | 37.1–50.1 µs | 16.45–21.33 ms |
| medium | cosine centroid | 19.0–24.7 µs | 30.2–39.3 µs | 37.3–47.4 µs | 40.1–55.8 µs | 4.66–6.06 ms |

Runtime construction/close remained approximately 2.3–3.2 µs in these attempts.

## Interpretation

The measured local decision path is in the tens-of-microseconds range for both reference
compiler families at the tested sizes. The larger and more visibly scaling cost is artifact
load/validation, especially for multinomial NB where the medium JSON artifact is roughly
323 KB.

These results do not establish that inference will remain cheap for every real workload.
They do establish that the current repository has no measured evidence that would justify
adding an optimized/native inference dependency solely to reduce per-request prediction
latency.

If optimization becomes necessary, the first investigation should separate:

1. artifact serialization/load/validation cost;
2. tokenizer and sparse feature construction cost;
3. raw predictor arithmetic;
4. calibration and runtime object-validation overhead.

Any future optimized backend should be compared against this reference harness and must
preserve class selection and calibrated probability semantics within an explicitly
documented tolerance.
