# ADR-0008: Shadow evidence separates teacher agreement from truth

- Status: Accepted
- Date: 2026-09-26

## Context

A local decision path should not be promoted merely because it agrees with the teacher that produced training labels. Teacher agreement can measure distillation fidelity, but it cannot substitute for observed outcomes, adjudication, or another explicit truth source.

Shadow evaluation must also be side-effect free: the authoritative application decision remains unchanged while the local path is replayed for evidence.

## Decision

Nerlex shadow replay:

- executes the compiled + calibrated local path against recorded requests;
- applies the already-fitted empirical gate without inventing a new threshold;
- records whether each request would be locally eligible or require fallback;
- compares local predictions to non-abstained teacher observations separately;
- resolves truth only through an explicit non-teacher provenance priority;
- reports teacher agreement and outcome/truth correctness as different metrics;
- records route-versus-outcome evidence, including whether a deferred local prediction would have been correct;
- never invokes the external fallback provider during offline shadow replay;
- sorts evidence by request identity and hashes the complete source trace set;
- produces an immutable content-addressed report bound to the DecisionSpec, compiler, calibration, gate, and source trace hash.

If multiple teacher observations disagree, the teacher comparison is marked ambiguous rather than choosing one silently. Conflicting truth labels at the same selected provenance priority fail closed.

## Consequences

Shadow reports can support later promotion/canary decisions without changing production behavior or silently conflating teacher fidelity with task accuracy.

No universal promotion threshold is defined here. Promotion policy remains a later lifecycle layer that consumes evidence.
