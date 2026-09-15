# Validation record

## Empirical run supplied by the authors

- Linux workstation with NVIDIA RTX 6000 Ada Generation.
- 3,840 committed images per pipeline; 11,520 total.
- 11,520 automatic semantic evaluations and 1,440 DINO batches completed.
- SD 1.5 retained 19 model-native censored/all-black outputs; the other two
  pipelines retained none.
- Two independent raters completed all 1,536 selected images.
- 144 image-state disagreements (199 atom bits) were adjudicated.
- Pre-adjudication joint agreement: 0.969, Cohen's kappa 0.937.
- Pre-adjudication atomic agreement: 0.948, Cohen's kappa 0.891.

## Release checks

The release test suite covers:

- frozen-suite construction, disjoint development/main specifications, and balance;
- exact finite-grid decomposition and direct Hamming-distance identities;
- all-35-split held-out selection, tied minima, and ordering invariance;
- strict complete-grid and semantic-state validation;
- atomic writes, simulated interruption, corruption recovery, and writer locking;
- PixArt transformer/pipeline composition and non-finite scheduler provenance;
- evaluator and DINO checkpoint reuse after simulated interruption;
- two-rater blocking, disagreement-only adjudication, and main-paper gating;
- synthetic end-to-end analysis and generated tables;
- raw released label reconstruction and published numerical values.

Run:

```bash
python -m pytest -q
python -m paraseedbench.verify_release
```

The first command is software validation; the second is evidence validation.
Neither substitutes for GPU regeneration.

Release-build result: **46 tests passed** on the packaging host, followed by a
successful CPU evidence verification and editable/wheel build.

## Known limits

- Generated PNGs and model weights are not included.
- The complete original `run_manifest.json` still needs to be recovered from the
  workstation/Drive; the published audit recorded its expected hash.
- Cross-hardware or cross-library pixel identity is not asserted.
- No native-resolution SDXL sensitivity run was performed.
- The audit contains four scenes per category, so bootstrap intervals are
  descriptive stability summaries rather than calibrated population coverage.
- One image per wording–seed cell does not identify a noise-corrected latent
  interaction.
