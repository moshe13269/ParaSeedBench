# ICASSP-style technical review and revision record

Reviewed: the original ParaSeedBench manuscript and shipped Python implementation.
Scope: methodological correctness, originality/significance, validation, reproducibility,
clarity, and the ICASSP 2027 four-technical-page plus restricted fifth-page format.
This is an editorial assessment, not an actual ICASSP committee decision.

## Overall recommendation on the original draft: Reject

The idea is relevant to image/signal-processing evaluation, and distinguishing
correctness from mere similarity is valuable. However, the original file is a
protocol proposal, not an experimentally supported benchmark paper. It contains
no main results or completed human validation, and its novelty overlaps directly
with existing paraphrase and semantic-evaluation work. Several claims are not
implemented as described. Editing cannot replace missing empirical evidence.

| Criterion | Original assessment | Revised status |
| --- | --- | --- |
| Relevance | Appropriate for image generation/evaluation; not an acoustics contribution | Retained, with narrower measurement focus |
| Originality | Limited; paraphrases, compositional judges, and seed effects are established | Finite-grid interaction analysis sharpens the contribution; empirical value still unproven |
| Technical soundness | Useful definitions, but confounded disagreement interpretation and protocol mismatches | Added exact decomposition, estimand qualifications, stricter controls and validation |
| Experimental support | Insufficient: no results and no audited evaluator | Complete development grid and one-rater audit supplied; main results and resolved two-rater audit remain pending |
| Reproducibility | Partial: existence-only resume, unpinned models, incomplete grids, non-resumable embeddings | New versioned, hash-validated and checkpointed execution path |
| Clarity and format | Overclaims, incorrect template route, anonymous author block | Official spconf format, four technical pages plus reference page; real authors still required |

## Post-development-run review (v0.2.2)

The supplied RTX 6000 Ada development artifacts are complete: 16 scenes, two seeds,
five variants, three models, and 480 scored images. The 48 model-scene records have
unique keys; all finite metrics satisfy their stated bounds; the exact
$V_w+V_s+V_{ws}=V_{total}$ and disagreement identities hold to floating-point
precision; category-balanced point estimates reproduce from the case table; and no
images were censored or black.

The automatic judge reports mean joint accuracy of 0.344 for PixArt-Sigma, 0.133 for
SD 1.5, and 0.086 for SDXL. Corresponding empirical worst-wording accuracies are
0.156, 0.031, and 0.031. These are development diagnostics, not paper estimates.
PixArt-Sigma has higher estimated accuracy but also greater wording disagreement
than SDXL (paired development difference 0.099, 95% scene-bootstrap interval
[0.023, 0.195]). Low variation for SDXL cannot be interpreted favorably because
its estimated accuracy is also low.

The main scientific blocker is evaluator validity, not software completeness.
Exclusive paired-control discrimination is only 0.042 for PixArt-Sigma and zero for
the other two models; overall correctness-conditioned diversity is undefined because
at least one category has no eligible correct pair. Neither observation proves a
judge bug, because generated images are not ground truth and the paired control is
deliberately stringent. They do make a blinded human development audit necessary
before the main run. If human states materially change the model/category accuracy
or variation conclusions, revise the evaluator using development labels and repeat
development in a new output directory.

The live log also exposed environment-dependent preprocessing warnings. Version
0.2.3 explicitly disables PixArt caption cleaning, pins the OWLv2/DINO slow processor
path, records split/grid counts in analysis metadata, and improves publication labels.
Because code and frozen configuration are part of the run contract, v0.2.3 must use
a new development output and locked config. The earlier development run remains
useful diagnostic evidence but must not be mixed into the main run.

## Post-human-development-audit review (v0.2.3 data, v0.2.4 analysis)

The blinded development audit is complete for 192 images: two scenes per category,
four equivalent wordings, two seeds, and all three models. One annotator supplied a
valid atomic state for every image. This is a useful diagnostic, but it does not
provide inter-annotator reliability and has only eight independent scene clusters.

Against the human joint label, the automatic evaluator has 0.781 raw agreement,
0.484 recall, 0.923 specificity, balanced accuracy 0.703, and Cohen's kappa 0.449.
Raw agreement is inflated by negative cases. Count recall is 0.231 and spatial recall
is 0.385; both human-positive color-binding images are missed. Model-level human versus
automatic mean joint accuracies are 0.578/0.359 for PixArt-Sigma, 0.234/0.203 for
SD 1.5, and 0.156/0.063 for SDXL. The mean-accuracy ordering survives, but variation
conclusions do not: SDXL seed disagreement is 0.482 manually versus 0.260 automatically,
and the wording-disagreement ordering changes.

**Gate decision:** no-go for automatic-only primary semantic claims; conditional go
for the benchmark after a human-primary protocol patch. Version 0.2.4 retains automatic
scores as secondary diagnostics, reports atomic-bit and exact-state agreement, emits
paired machine--human metric differences, creates a blinded adjudication queue, and
blocks paper export until a main audit has at least four scenes per category, two
independent raters, and resolved atomic disagreements. Existing v0.2.3 development
images need only be rescored; the main run must be frozen under v0.2.4.

## Major findings, corrections, and residual requirements

### 1. Novelty is narrower than the original framing

MetaLogic already evaluates logically equivalent prompt pairs and object/count/position
inconsistency. SemVarBench already examines linguistic permutations. GenEval and TIFA
already provide compositional/faithfulness evaluations, while reliable-seed research
already demonstrates initialization-dependent composition. A prompt x seed loop alone
is not a sufficiently strong novelty claim. Sources: [MetaLogic](https://arxiv.org/abs/2510.00796),
[SemVarBench](https://arxiv.org/abs/2410.10291), [GenEval](https://arxiv.org/abs/2310.11513),
[TIFA](https://arxiv.org/abs/2303.11897), [All Seeds Are Not Equal](https://arxiv.org/abs/2411.18810).

Correction: explicitly acknowledge this overlap; focus on the interaction-aware,
correctness-aware finite-grid diagnostic. Standard orthogonal decomposition is not
presented as a newly invented statistical method. The eventual results must show
what this diagnostic reveals beyond accuracy and the two simple disagreements.
No comprehensive novelty guarantee is possible from this targeted literature review.

### 2. Two conditional disagreements do not isolate wording and seed effects

Original equations (3)-(4) average Hamming distance over different conditionings.
Both contain wording-seed interaction. For J wordings and S seeds:

\[
D_w=\frac{2J}{J-1}(V_w+V_{ws}),\qquad
D_s=\frac{2S}{S-1}(V_s+V_{ws}).
\]

The finite-grid squared variation partitions exactly into Vw + Vs + Vws because
row and column residual means vanish. This is a descriptive identity, not causal
identification or an unbiased random-effects estimator. The XOR example has zero
wording and seed main components but maximal paired disagreements.

Correction: implement the components and test the identity against direct distances
on random binary grids. Add constant-correct, constant-wrong, wording-only, seed-only,
and interaction-only analytical tests. Retain accuracy alongside all variability.

### 3. The v1 prompt suite is not uniformly meaning-equivalent

Concrete examples in build_suite.py: some variants request a photograph while others
do not; spatial wording introduces 'sits'; 'left of' becomes absolute 'on the left',
and 'above' becomes 'at the top'. These can legitimately alter composition. Pair
selection takes the first combinations, overrepresenting early nouns such as cat.

Correction: new versioned suite with common photographic style, purely relative
spatial language, and cyclic pair selection. Color/spatial prompts now require one
instance of each relevant object, resolving which instance is evaluated. This
changes the scientific task and requires fresh v2 generation. Independent human
review remains required; code/schema validation cannot certify semantic equivalence.

### 4. The claimed development split was not separate

The original smoke selection takes scenes from the same scenes_v1.jsonl used for the
main study. It is a valid installation smoke test but cannot simultaneously support
the claim that evaluator thresholds were selected on an independent development set.

Correction: 16 development scenes, with a four-scene smoke subset. Their base and
control specifications do not overlap main specifications. Additional development
nouns avoid count-control leakage, but create a calibration-domain difference that
is disclosed. Do not claim that previously selected thresholds were independently
validated; either keep them declared defaults or calibrate on the new development set.

### 5. Correctness-conditioned diversity is not an unconditional model ranking

Different generators pass on different scenes and seeds. Their eligible DINO pairs
therefore differ. DINO cosine distance does not isolate only unrequested visual
attributes, nor does larger distance imply better quality. The original prose did
not specify the same weighting as the implementation, and its upward arrow was too strong.

Correction: define the hierarchy (pair, eligible wording, eligible scene, equal
category weighting); export eligible pair/wording/scene counts; leave empty cases
undefined; remove 'higher is always better' claims. Human verification of eligibility
and comparisons on a shared eligible subset remain useful additional sensitivity tests.

### 6. Worst-wording accuracy and uncertainty need qualified interpretation

The minimum of noisy sample averages is downward biased. A scene bootstrap does not
remove this. Shared seeds and templates also limit broad population claims. v1
bootstraps globally rather than preserving the designed category balance.

Correction: name the metric empirical worst-wording accuracy; add two-way held-out
selected-worst sensitivity and all-wordings accuracy; bootstrap within categories
and pair model differences at scene level. State the finite-grid conditional target.
Secondary intervals are descriptive, not multiplicity-adjusted significance tests.
The held-out metric selects on four seeds and scores on four, so it is not an unbiased
estimate of the true minimum either. More seeds would be needed for stronger claims.

### 7. Automatic judge reliability is a central threat, not a footnote

The method measures the outputs of OWLv2/CLIP. Correlated detection errors can look
like generator instability or stable failure. Sampling isolated balanced pass/fail
images does not permit complete-grid human metric reconstruction; unweighted
pass/fail-stratified agreement is not population agreement. The original audit's
image paths also expose model identity. [GenEval 2](https://arxiv.org/abs/2512.16853)
documents why static judge agreement cannot simply be assumed.

Correction: FP32 judge inference, explicit pinned judge revisions, complete-grid
scene sampling independent of pass/fail, neutral copied image names, private key,
required atomic labels, and human-versus-machine metric reconstruction. The completed
development audit confirms that raw joint agreement can hide poor positive recall and
changed variation conclusions. The main human-primary audit now prespecifies four scenes
per category: 16 scene units and 1,536 images total per annotator across three models.
Two independent raters, atomic/exact-state agreement, and blinded adjudication are
required. Precision remains limited and ethical/institutional handling remains factual.

### 8. Counterfactuals were generated but not analyzed meaningfully

v1 removes counterfactual rows before analysis. Its prose discusses sensitivity
without computing targeted response. Object replacement is also non-exclusive: an
image with both original and replacement objects can satisfy both descriptions.

Correction: cross-evaluate p0 and cf0 images under both specifications for exclusive
count, color and spatial changes. Paired discrimination requires own-spec success
and alternate-spec failure on both images. Presence controls receive own accuracy
only. All controls stay outside the wording/seed decomposition.

### 9. Seed-inaccessible/API claims exceed the experiment

The code generates with explicit shared seeds and contains no API adapter or
independent API sampling study. A marginal JSD from this grid is not an experiment
on a seed-inaccessible service. Original smoothing enumerates only observed states,
despite more general Jeffreys-smoothing language.

Correction: remove operational/API validation claims. Implement smoothing over the
full binary state support. Retain JSD only as an exploratory sample-size-sensitive
diagnostic, not an equivalence test or primary ranking metric.

### 10. Scientific corruption can arise from partial/resumed runs

Original generation skips any existing PNG/JSON pair without validating its bytes,
prompt or config. run_info.json is overwritten on resume. Model commits are not
pinned. CSV scores are reused by path despite changed evaluator settings. The grid
checker derives its expected axes from observed rows, allowing an entire missing
seed, wording, model, or scene to go unnoticed. DINO is saved only at the end.

Correction: immutable model/evaluator commits; guarded run manifests and source/env
identity; relative image paths; SHA-256 image validation; atomic image/metadata,
per-image score, and per-batch embedding commits; frozen-config expected-grid
validation; one-writer locks. Both regeneration and scoring reuse are hash-aware.
Atomic local replacement cannot guarantee remote Drive synchronization or recovery
from storage loss. V2 resumes completed units, not an interrupted denoising step.

### 11. Execution and performance claims were too strong

PowerShell's ErrorActionPreference does not make every external Python nonzero exit
terminate a Windows PowerShell script. The old pipeline can continue after a failure.
Deleting a pipeline only inside unload_pipeline does not remove the caller's reference,
so another model can be loaded while the first still occupies GPU memory. Attention
slicing is always enabled in v1, which can penalize modern attention execution.

Correction: a single fail-fast Python orchestrator and exit-code-checked wrappers;
caller-side cleanup before loading another model; configurable slicing disabled in
new profiles. Explicit resident RTX 6000 Ada and offloaded RTX 3070 profiles are included.
No CUDA execution was performed in this editing environment. Previous hourly GPU
estimates were unmeasured planning estimates, not benchmark evidence.

### 12. Censoring must be measured rather than silently reclassified

The shared screenshot shows safety-triggered black outputs. It establishes censoring,
not that every trigger is a false positive. The earlier advice to treat every such
output as an unacceptable semantic failure and delete the run was too categorical.
For deployed-pipeline reliability, censoring is an outcome; generator-only evaluation
would require a separately specified filtering policy.

Correction: retain native filtering, save flags, report censor and black rates, count
them in end-to-end accuracy, and label uncensored accuracy conditional. Never reroll
only censored samples. A mix of disabled-filter and retained-filter outputs is not v2.

### 13. Format and presentation need actual conference compliance

The original PDF uses an article fallback with page numbers and an anonymous author
block, rather than the official conference style. Seven narrow table columns with
confidence intervals can become unreadably small when resized. PixArt-Sigma is
represented by an alpha-paper citation; DINO/OWLv2/CLIP citations are missing. The
earlier suggested error-bar caption is not supported by the original bar plot.

Correction: official spconf.sty, 10-point text, no page numbers, maximum four technical
pages plus reference-only fifth page, readable transposed numerical tables and
actual plotted confidence intervals. Add the correct method citations. Real authors
and affiliations remain unfilled because the complete author list was not supplied.
ICASSP 2027 is not blind reviewed. Consult the [official paper kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
for the author list, ORCID, font, margin and fifth-page requirements. Official style
download: https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/spconf.sty .

## What still blocks a defensible submission

1. Independent review and freezing of v2 prompt groups and evaluator settings.
2. Complete v0.2.4 main runs with recorded provenance.
3. A resolved two-rater, human-primary main audit with at least four scenes/category.
4. Measured findings showing diagnostic value beyond existing accuracy/disagreement
   baselines; a model-ranking reversal is not required and must not be manufactured.
5. Native-resolution sensitivity if making broader SDXL/model-quality rankings.
6. Factual author/affiliation/ethics metadata and final four-plus-one page verification.

**Revised assessment after the development audit:** Weak Reject, borderline. The audit
adds real evidence and identifies a consequential evaluator failure, strengthening the
paper's motivation. It is still not submission-ready because main results, independent
main labels, adjudication, prompt review, and author metadata are missing. A completed
human-primary main study could support Weak Accept if the interaction analysis reveals
information beyond ordinary accuracy and disagreement summaries.

## Migration decision

Keep all earlier output untouched. Rescore the completed v0.2.3 development audit with
v0.2.4; this post-processing step needs no GPU. Freeze and start the main experiment
only from the v0.2.4 source in a new output root. Re-running v1 analysis cannot fix
changed prompt meanings, judge instance ambiguity, or missing generation provenance.
If there is not enough time for the main human validation, report the work as a pilot
or defer submission; do not describe automatic-only results as confirmatory.

See README_v2.md for executable commands and VALIDATION.md for tests actually run.
