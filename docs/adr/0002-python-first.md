# ADR-0002: Use Python for the initial implementation

- Status: Accepted
- Date: 2026-09-26

## Context

The first product risk is whether the decision-compilation lifecycle is useful, not whether Python orchestration overhead is the runtime bottleneck.

## Decision

Implement the v0.x control plane and first runtime in Python.

## Consequences

Python provides direct access to ML, calibration, data, and ONNX tooling. A Rust/C++/native runtime is deferred until profiling or deployment constraints demonstrate a concrete need.
