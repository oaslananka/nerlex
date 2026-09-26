# ADR-0005: Start compilation with safe deterministic local baselines

- Status: Accepted
- Date: 2026-09-26

## Context

Nerlex needs a concrete compile stage before calibration, selective evaluation, shadowing, and promotion can be exercised end to end.

The first compiler implementation should establish the durable artifact and data-boundary contracts without forcing a heavyweight ML dependency or unsafe executable model serialization.

## Decision

Nerlex v0.1 starts with two dependency-light CPU compiler families for static typed decisions:

1. **Multinomial Naive Bayes** over deterministic Unicode token counts.
2. **Centroid/cosine** scoring over deterministic sparse TF-IDF vectors.

Both families:

- train from the verified **train split only**;
- reject dynamic-candidate DecisionSpecs until a semantic candidate scorer exists;
- require every output label to be represented in training data;
- emit JSON-only artifacts rather than pickle/joblib objects;
- bind artifact identity to the source snapshot ID, train-split SHA-256, DecisionSpec hash, compiler kind/version, configuration, labels, and learned payload;
- validate artifact identity when loading;
- return scores and softmax probabilities marked explicitly as **uncalibrated**.

The compiler backend interface is a protocol so stronger semantic or optimized backends can be added without changing the public compile/predict lifecycle.

## Consequences

These baselines are reference implementations, not claims of parity with proprietary decision models. Their value is reproducibility, auditability, and a safe contract for the next stages: calibration and sealed evaluation.

Dynamic candidates remain intentionally unsupported in these first supervised baselines. Supporting them requires a candidate-conditioned semantic scorer rather than a fixed class head.
