# Reproducibility and restoration guide

ParaSeedBench distinguishes four reproducibility targets. They should not be
collapsed into a single claim.

## 1. Verify the paper numbers from released evidence

This is CPU-only and does not require the generated PNGs:

```bash
python -m pip install --no-build-isolation -e ".[test]"
python -m paraseedbench.verify_release
```

The verifier checks:

- 96 frozen main scenes and their hash;
- the exact seed list `[11,29,47,71,101,131,173,211]`;
- all evidence-file SHA-256 checksums;
- 1,536 unique audit IDs and complete 4×8 model–scene grids;
- 144 pre-adjudication image-state disagreements;
- adjudication coverage exactly equal to those disagreements;
- inter-rater joint and atomic agreement;
- human-primary accuracy, disagreement, variance components, and corrected
  held-out selected-worst values;
- full-suite automatic variance components and runtime counts.

## 2. Re-analyze a completed run

Keep these files together:

```text
run_manifest.json
semantic_scores.csv
dino_embeddings.npz
embedding_chunks/
scores/
images/
pipelines/
repeatability_v2/
evaluation_info.json
```

Then run:

```bash
python -m paraseedbench.reanalyze_v2 \
  --input /absolute/path/to/the/run \
  --dataset data/scenes_v2_main.jsonl
```

The command verifies the manifest's dataset hash and the complete score/embedding
grid. It writes new products to `analysis_v030` by default. It does not reopen the
generation contract or overwrite the original analysis.

## 3. Resume the original v0.2.4 workstation run

The run contract hashes all Python files. Therefore a v0.3.0 source tree correctly
refuses to masquerade as the v0.2.4 writer. Use the exact source archive in
`compatibility/` to finish or resume the original generation/evaluation. It has
source hash:

```text
147de68ac48de0c0f4aea04e0f16d6f8389abde77c7a12a5ac47f81727969c27
```

After the immutable image/score grid is complete, use v0.3.0 only for new derived
analysis or audit scoring.

The original run manifest was not available while this release was assembled.
When workstation access returns, copy it to:

```text
paper_artifacts/provenance/original_run_manifest.json
```

and validate it:

```bash
python scripts/validate_imported_run.py
```

This checks it against the original audit's recorded manifest hash and prints the
captured environment. Until that file is added, do not claim that every installed
package/driver version is recoverable from Git alone.

## 4. Regenerate the full experiment

The immutable lock files contain prompt hashes, model/evaluator commits, settings,
and seeds. On an RTX 6000 Ada:

```bash
bash run_v2.sh configs/smoke_v2.lock.yaml
bash run_v2.sh configs/main_v2.lock.yaml
```

This regenerates the same experimental design, not guaranteed bit-identical PNGs
on different hardware, drivers, or library versions. Same-process duplicate-input
checks gate each pipeline. Exact pixel equivalence across machines is not claimed.

## Provenance caveat for the automatic CSVs

`paper_artifacts/automatic_main/` preserves the v0.2.4 automatic analysis uploaded
after the 11,520-image run. The legacy `crossfit_worst_accuracy` entries used one
even/odd split. The paper does not report automatic M96 held-out-selected-worst
values. All other automatic endpoints used in the paper are unaffected.

The raw audited image mapping and resolved labels are available, so
`paper_artifacts/human_audit/heldout_selected_worst_summary.csv` was recomputed
under v0.3.0 with all 35 unordered balanced splits. Its rounded values and 2,000
replicate metric-local bootstrap intervals are exactly those in the manuscript.

To correct the full automatic held-out metric later, copy the original
`semantic_scores.csv` and run `paraseedbench.reanalyze_v2`; aggregate case-level
CSVs alone do not contain enough information to reconstruct all balanced splits.

## Output integrity and interruption model

- PNG bytes are validated against adjacent metadata SHA-256 values.
- Metadata is written after the PNG and acts as the commit marker.
- Semantic evaluation commits one JSON record per image.
- DINO commits one compressed NPZ per batch and validates path/token coverage.
- Final tables require the exact complete grid.
- Temporary `.pending-*` files never count as committed work.
- A single-writer lock prevents concurrent mutation of one run root.

Local atomic replacement is not a guarantee that a cloud drive has synchronized
before shutdown. Preserve backups and wait for the storage provider to finish sync.
