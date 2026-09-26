# ADR-0003: Teacher output is provenance, not ground truth

- Status: Accepted
- Date: 2026-09-26

## Context

A smaller model can agree with a teacher while reproducing the teacher's mistakes. Treating every teacher decision as ground truth makes evaluation misleading.

## Decision

Nerlex keeps label provenance explicit. At minimum, it distinguishes:

- teacher decision
- human label
- observed outcome
- deterministic rule label
- adjudicated label

Dataset construction must select a label-resolution policy explicitly.

## Consequences

Teacher agreement and real task accuracy are separate metrics. Distillation may use teacher data, but public evaluation must not silently relabel it as adjudicated truth.
