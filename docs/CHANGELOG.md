# Changelog

## 0.3.0 — 2026-09-15

- aligned the held-out selected-worst implementation with the final manuscript:
  all 35 unordered balanced splits, reverse directions, and tied-minimum averaging;
- added wording/seed ordering invariance and tie regression tests;
- added a metric-local paper sensitivity summary for reproducible intervals;
- added CPU verification from released human labels and automatic artifacts;
- included immutable model/evaluator revisions and frozen lock configurations;
- added a safe re-analysis entry point that preserves original derived outputs;
- packaged final audit labels, adjudication, result provenance, and manuscript;
- documented Linux/Windows setup, GPU validation, restart, and v0.2.4 restoration.

## 0.2.4 — 2026-09-10

- introduced the two-rater human-primary audit and adjudication gate;
- made automatic semantic scores secondary diagnostics;
- added generator-specific evaluator agreement and metric differences;
- retained crash-safe per-image and per-batch checkpoints.
