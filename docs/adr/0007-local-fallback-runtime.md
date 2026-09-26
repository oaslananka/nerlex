# ADR-0007: Local runtime is a calibrated cascade with explicit fallback

- Status: Accepted
- Date: 2026-09-26

## Context

A compiled local classifier is useful only if runtime behavior preserves the evidence and abstention semantics established during calibration and evaluation.

A runtime that silently treats raw scores as calibrated confidence, invents a new threshold, or hides fallback failures would break the provenance guarantees established by the compiler/calibration/evaluation pipeline.

## Decision

Nerlex v0.1 uses a two-stage runtime cascade:

```text
request
  -> validate against compiled DecisionSpec
  -> local compiler
  -> calibration artifact
  -> empirical gate
     -> eligible: return local result
     -> ineligible: explicit abstention or configured fallback
```

The runtime bundle binds exactly one compiler artifact, one calibration artifact, and one empirical gate artifact. Construction fails closed if compiler, snapshot, calibration split, class-set, or gate lineage does not match.

The runtime never chooses a new confidence threshold. It applies only the threshold already stored in the gate artifact. An abstain-all gate always defers.

Fallback is provider-neutral. The core runtime accepts a callable that returns a typed `FallbackDecision`; provider SDKs and network clients remain outside the core package.

Runtime results preserve route identity and exact artifact IDs. Local and fallback results are therefore distinguishable in traces.

## Failure semantics

- incompatible request/spec: fail before inference;
- incompatible artifacts: fail when constructing the runtime bundle;
- local inference/calibration failure: raise a local runtime error;
- fallback timeout: raise an explicit timeout error;
- fallback exception/invalid output: raise an explicit fallback execution error;
- no configured fallback: return an explicit local abstention.

The thread-based timeout is a caller-facing deadline, not a hard process kill. Remote fallback clients should also enforce transport-level timeouts.

## Consequences

This design keeps v0.1 dependency-light and testable while preserving the distinction between evidence-backed local decisions and external fallback decisions.

Production promotion, canarying, drift detection, and provider-specific fallback integrations remain separate lifecycle concerns.
