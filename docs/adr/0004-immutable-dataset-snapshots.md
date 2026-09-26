# ADR-0004: Dataset snapshots are immutable and split by stable request identity

- Status: Accepted
- Date: 2026-09-26

## Context

Model training, probability calibration, and final evaluation must not silently reuse the same examples. Rebuilding a dataset should also produce evidence that can be compared across machines and over time.

Random row-order-dependent splitting is unsuitable because a different export order can reshuffle examples between train, calibration, and final test sets.

Teacher decisions are useful training signals, but they are not automatically ground truth.

## Decision

Nerlex dataset snapshots:

1. record an explicit ordered label-source priority;
2. exclude teacher decisions unless `teacher` is explicitly present in that priority;
3. assign each request to exactly one train/calibration/test split using a stable hash of the split seed and request ID;
4. sort examples deterministically within each split;
5. hash the exact bytes of every split;
6. derive the snapshot ID from the DecisionSpec, snapshot configuration, source/inclusion counts, and split artifacts;
7. omit wall-clock creation time from snapshot identity;
8. verify split hashes and counts before accepting an on-disk snapshot.

Conflicting labels from the same selected source fail closed instead of being resolved silently.

## Consequences

- Rebuilding from the same trace set and configuration is byte-reproducible.
- Input ordering does not change split membership.
- Calibration and final test sets are explicit first-class artifacts.
- Snapshot directories can be content-addressed and treated as immutable.
- Changing label policy, split seed, DecisionSpec, or dataset contents creates a different snapshot ID.
