# ADR-0009: Promotion is evidence-gated and canary rollout is deterministic

- Status: Accepted
- Date: 2026-09-30

## Context

Nerlex now has immutable compiler, calibration, gate, evaluation, and shadow artifacts.
The remaining lifecycle gap is deciding when an evidence-backed local path may receive
authoritative traffic and how that exposure can be increased or rolled back without
discarding provenance.

A production rollout must preserve several distinctions already established by earlier
ADRs:

- a calibration-split empirical gate is not a universal safety guarantee;
- final test evidence must not be reused to fit thresholds;
- teacher agreement is not truth accuracy;
- shadow execution is side-effect free and does not make the authoritative decision;
- a runtime result must still identify whether the actual decision came from the local
  path or fallback.

A single hard-coded promotion threshold would be misleading across workloads. Random
per-request canary assignment would also make evidence harder to reproduce and can move
requests between cohorts as rollout percentages change.

## Decision

Nerlex v0.1 will treat promotion as an explicit, evidence-gated lifecycle with immutable
assessment and rollout artifacts. It will not autonomously promote a candidate to
production.

The lifecycle becomes:

```text
evaluation + shadow evidence
  -> explicit promotion criteria
  -> immutable promotion assessment
  -> explicit operator/application approval
  -> immutable canary plan
  -> deterministic cohort assignment
  -> local cascade or control/fallback path
  -> captured outcomes
  -> explicit advance / hold / rollback
```

### 1. Promotion criteria are explicit and workload-specific

The core package will not define a universal "safe to promote" threshold.

A caller supplies a typed promotion policy with predeclared criteria. The first v0.1
policy should be intentionally small and based on evidence already produced by Nerlex,
including at minimum:

- maximum selective risk on the sealed test split;
- minimum selective coverage on the sealed test split;
- minimum truth-labeled eligible shadow sample count;
- maximum truth-based risk for locally eligible shadow decisions;
- minimum observed local eligibility/coverage in shadow replay.

Criteria that depend on truth must use non-teacher truth provenance. Teacher agreement
may remain diagnostic evidence but must not satisfy a truth-accuracy promotion criterion.

An abstain-all gate cannot pass a policy that requires positive local coverage.

### 2. Promotion assessment is immutable evidence, not deployment state

A promotion assessment binds exactly one runtime lineage and its evidence:

- DecisionSpec hash;
- compiler artifact ID;
- calibration artifact ID;
- empirical gate artifact ID;
- evaluation report ID;
- shadow report ID;
- exact promotion policy;
- per-criterion observed values and pass/fail results;
- overall pass/fail result.

Construction fails closed if the evaluation report or shadow report does not belong to
the exact compiler/calibration/gate lineage being assessed.

The assessment is content-addressed and immutable. Wall-clock approval time, operator
identity, environment name, and mutable deployment state are not part of its evidence
identity.

A passing assessment means only that the supplied criteria were satisfied by the
referenced evidence. It is not a statistical or production-safety guarantee.

### 3. Canary activation is an explicit action

A passing promotion assessment does not itself change traffic.

The application/operator explicitly creates or activates a canary plan that references:

- the passing promotion assessment;
- the exact runtime bundle lineage;
- a rollout fraction;
- a stable assignment seed/version;
- the previous plan ID when the plan is part of a ramp or rollback chain.

The core library remains provider-neutral and does not own a hosted deployment control
plane.

### 4. Canary assignment is deterministic, stable, and independent of model output

Canary/control assignment happens before local inference and uses only stable,
non-sensitive identity such as:

```text
hash(assignment_version, assignment_seed, decision_id, spec_version, request_id)
```

The hash is mapped to a stable bucket in `[0, 1)`.

A request is assigned to the canary cohort when its bucket is below the plan's rollout
fraction. Increasing the fraction therefore produces nested cohorts: requests already in
the canary cohort stay there.

Assignment must not depend on request state, predicted label, model confidence, truth,
teacher output, latency, or later outcome. This avoids confidence-dependent rollout
selection and makes replay reproducible.

### 5. Canary assignment and local eligibility are separate dimensions

Being assigned to the canary cohort does not force a local decision.

For a canary-assigned request:

1. the existing `RuntimeBundle` and calibrated local cascade execute;
2. the already-fitted empirical gate decides local eligibility;
3. an ineligible request follows the configured fallback behavior.

For a control-assigned request, the authoritative control/fallback path executes directly.

Therefore Nerlex must distinguish at least:

- canary-assigned fraction;
- locally eligible fraction within canary;
- actual local-serving fraction;
- fallback/control-serving fraction.

The existing `DecisionResult.route` continues to identify the actual decision source
(`local`, `fallback`, etc.). Canary/control cohort identity must be recorded separately
rather than introducing a `canary` route that would hide the true decision source.

### 6. Canary evidence preserves rollout and artifact lineage

Canary observations must record enough identity to reconstruct the serving decision:

- canary plan ID;
- promotion assessment ID;
- compiler/calibration/gate artifact IDs;
- cohort assignment;
- actual decision route;
- request identity;
- later truth/outcome provenance when available.

Rollout metrics must keep control assignment, canary assignment, gate abstention, local
serving, fallback serving, and truth-bearing outcomes distinct.

Teacher agreement must remain separate from truth-based canary quality.

### 7. Advance, hold, and rollback remain explicit in v0.1

Nerlex v0.1 will not autonomously increase rollout or promote to 100%.

A rollout change creates or activates a new immutable canary plan. Raising the fraction,
holding it, reducing it, setting it to zero, or moving to full rollout is therefore an
explicit action with a traceable plan identity.

Rollback does not mutate or delete prior artifacts. It activates a previous known-good
plan or a new zero/local-disabled plan and records the relationship to the plan being
replaced.

Future versions may add automated recommendations or guarded control-plane actions, but
they must consume the same evidence and cannot silently weaken lineage or truth
requirements.

### 8. Failure semantics

The first implementation should fail closed on control-plane/evidence errors while
protecting authoritative request handling:

- incompatible promotion evidence: reject assessment construction;
- failing promotion criteria: produce a valid failed assessment, not an exception;
- canary plan referencing a failed/incompatible assessment: reject plan construction;
- invalid rollout fraction/assignment configuration: reject plan construction;
- request/spec incompatibility: reject before cohort assignment or inference;
- canary local gate abstention: use existing fallback semantics;
- canary local runtime failure: must be observable and must not be reported as a
  successful local decision;
- control/fallback failure: preserve the explicit runtime failure semantics from
  ADR-0007.

Whether an application converts a local-runtime exception into an emergency control-path
retry is an integration policy and must be explicit; the core library must not silently
hide such failures.

## Initial implementation boundary

The smallest v0.1 implementation should introduce:

1. a typed promotion policy;
2. a deterministic, immutable promotion assessment;
3. a deterministic, immutable canary plan;
4. a pure cohort-assignment function;
5. regression tests for lineage validation, policy evaluation, deterministic/nested
   assignment, tamper detection, and separation of cohort identity from decision route.

A provider-specific deployment system, dashboard, autonomous ramp controller, autonomous
rollback controller, drift service, or hosted control plane remains out of scope.

Because current shadow evidence supports static choice decisions, the first promotion
assessment is effectively limited to that same evidence-compatible surface. Broader
decision kinds should be enabled only when equivalent truth-bearing shadow/canary
evidence exists.

## Consequences

- Promotion becomes reproducible and auditable without pretending one threshold fits all
  workloads.
- Rollout cohorts remain stable as exposure changes.
- The existing calibrated gate continues to control local eligibility; canary percentage
  controls exposure, not model confidence.
- Actual decision route remains observable independently of rollout cohort.
- Operators retain explicit control over production state in v0.1.
- Later hosted or automated rollout systems can build on immutable evidence and plan
  identities instead of replacing the core contracts.
