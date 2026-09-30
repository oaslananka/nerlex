# ADR-0010: Defer an optimized inference backend until measured need exists

- Status: Accepted
- Date: 2026-09-30

## Context

Nerlex v0.1 already has deterministic CPU-local multinomial Naive Bayes and cosine
centroid compilers, calibrated prediction, selective gating, live fallback serving, and
evidence-gated canary rollout.

The roadmap previously listed a portable optimized inference backend as the next possible
product layer. Candidate implementations could include a denser numeric representation,
NumPy, ONNX Runtime, or a native extension. Each option would add packaging, artifact,
numerical-parity, and maintenance complexity.

Before introducing that complexity, PR #14 added a reproducible benchmark for the existing
reference path. The benchmark measures raw prediction, calibrated prediction, complete
local-runtime decisions, artifact loading, runtime construction, artifact size, and model
shape for both current compiler families.

The preserved baseline is documented in
`docs/benchmarks/reference-inference-baseline-2026-09-30.md`.

## Decision

Nerlex will **not** add an optimized/native inference backend in v0.1 based on the current
evidence.

The existing pure-Python deterministic predictors remain:

- the default local execution path;
- the portability baseline;
- the correctness oracle for any future optimized representation.

The project will not add NumPy, ONNX Runtime, Rust/C++ extensions, or another native
runtime solely for theoretical speed.

The reference benchmark remains an engineering-evidence workflow. It is intentionally not
a merge-time performance gate because GitHub-hosted runner variance is material and the
project does not yet have a workload-specific latency SLO.

## Evidence

Two consecutive attempts of the final PR #14 benchmark head used a deterministic medium
fixture with 8 classes and 1,123 tokens.

Observed full local-runtime medians were approximately:

- multinomial NB: 34.6–45.2 microseconds;
- cosine centroid: 37.3–47.4 microseconds.

Observed p95 values were approximately:

- multinomial NB: 37.1–50.1 microseconds;
- cosine centroid: 40.1–55.8 microseconds.

The more significant scaling signal was artifact load/validation:

- medium multinomial NB: about 16.45–21.33 milliseconds for a 322,813-byte artifact set;
- medium cosine centroid: about 4.66–6.06 milliseconds for a 78,575-byte artifact set.

The benchmark attempts also showed enough hosted-run variance that a single absolute
microbenchmark value should not drive architecture.

## Consequences

- v0.1 keeps a dependency-light CPU-only runtime.
- The repository avoids introducing a second execution backend and its parity/versioning
  surface before it is needed.
- Performance work should first identify the actual bottleneck in representative workloads.
- If startup or cold-load cost becomes material, artifact representation/loading should be
  investigated before assuming predictor arithmetic is the problem.
- Existing compiler artifacts remain canonical JSON and safe from arbitrary-code
  deserialization.
- Calibration and gate lineage remain independent of any future optimized execution
  backend.

## Revisit conditions

This decision should be revisited when at least one of the following is demonstrated with
representative measurements:

1. local inference is a material contributor to an application's end-to-end latency or
   throughput limit;
2. real artifact sizes/model shapes materially exceed the current benchmark surface;
3. artifact load/startup cost violates an actual application requirement;
4. a portable optimized backend provides a meaningful measured improvement while
   preserving exact class decisions and documented probability tolerance;
5. packaging or deployment requirements change such that the current pure-Python
   representation is no longer appropriate.

If the decision is revisited, the proposal must define backend identity, artifact
versioning, numerical tolerance, parity tests, unsupported-model behavior, packaging
impact, and fallback to the reference predictor.
