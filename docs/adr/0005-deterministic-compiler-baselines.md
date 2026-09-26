# ADR-0005: v0.1 ships deterministic dependency-light compiler baselines

- Status: Accepted
- Date: 2026-09-26

## Context

Nerlex must prove the compile/artifact/evaluation lifecycle before introducing heavyweight model stacks or claiming Jev-equivalent behavior.

The first compiler layer also needs to make train/calibration/test boundaries mechanically obvious.

## Decision

v0.1 begins with two CPU-local fixed-class baselines:

1. multinomial Naive Bayes over deterministic Unicode word features;
2. cosine centroid classification over deterministic TF-IDF sparse vectors.

Both train **only** on the snapshot's train split.

Compiled artifacts are canonical JSON and contain:

- exact DecisionSpec and spec hash;
- source snapshot ID;
- train split hash/count;
- compiler configuration/version;
- deterministic model parameters;
- artifact content hash;
- explicit `calibrated=false`.

No pickle or arbitrary-code deserialization is used.

Dynamic candidate decisions are rejected explicitly until a semantic candidate scorer is implemented.

## Consequences

These baselines are not presented as Jev clones or state-of-the-art decision models. They establish safe, reproducible compiler and artifact contracts that stronger future backends can implement.
