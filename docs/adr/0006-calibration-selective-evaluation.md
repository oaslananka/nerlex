# ADR-0006: Calibration and selective evaluation use dedicated held-out splits

- Status: Accepted
- Date: 2026-09-26

## Context

Raw softmax-like scores from local compiler baselines are not automatically calibrated probabilities. A confidence threshold selected from the final test set would also contaminate the evidence used to judge the system.

## Decision

Nerlex v0.1 uses:

1. scalar temperature scaling fitted **only** on the calibration split;
2. an optional empirical confidence gate fitted **only** on the calibration split;
3. final quality/calibration/selective metrics computed **only** on the test split.

Calibration artifacts record:
- source compiler artifact ID;
- source dataset snapshot ID;
- calibration split hash/count;
- calibration configuration;
- fitted temperature;
- content-derived artifact ID.

Empirical risk gates record:
- compiler and calibration artifact IDs;
- calibration split hash/count;
- predeclared empirical maximum risk and minimum accepted count;
- fitted confidence threshold, or an explicit abstain-all result.

Evaluation reports record:
- source compiler/calibration/gate artifact IDs;
- exact test split hash/count;
- raw and calibrated classification/calibration metrics;
- risk-coverage curve and AURC;
- selective coverage/risk for a threshold fitted before test evaluation.

## Limitations

The empirical gate is not a distribution-free safety guarantee. Calibration and empirical risk can degrade under distribution shift. Production use therefore still requires monitoring, fallback, and workload-specific acceptance criteria.

The test split is for evaluation, not threshold selection.
