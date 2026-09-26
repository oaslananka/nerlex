# ADR-0001: Position Nerlex as an AI Decision Compiler

- Status: Accepted
- Date: 2026-09-26

## Context

Open-source typed-decision models, semantic routers, evaluation systems, and observability platforms already exist. Building another single-model Jev-style clone would couple the project to a crowded implementation layer.

The broader engineering problem is the lifecycle required to migrate a repeated decision from a general/remote teacher to a narrower local path without losing provenance, calibration evidence, fallback, or rollback.

## Decision

Nerlex is an **AI Decision Compiler**.

Its durable lifecycle is:

```text
define -> capture -> snapshot -> compile -> calibrate -> evaluate
       -> shadow -> gate -> canary -> promote/rollback
```

Models/providers are backends, not the product identity.

## Consequences

- Jev, Laya, classical models, encoders, and future systems may become adapters/backends.
- Evaluation and artifact provenance are core product features.
- Nerlex does not need to reproduce any proprietary architecture.
