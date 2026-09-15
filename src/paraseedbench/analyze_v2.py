from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .io import stable_hash
from .protocol_v2 import jobs
from .schema import atom_keys
from .storage_v2 import atomic_path, save_json
from .metrics_v2 import decomposition, crossfit_worst, state_jsd, stratified_ci
from .metrics import mean_pairwise_cosine_distance

METRICS = ["mean_accuracy", "empirical_worst_accuracy", "crossfit_worst_accuracy", "all_wordings_accuracy",
           "wording_disagreement", "seed_disagreement", "variance_wording", "variance_seed", "variance_interaction",
           "variance_total", "semantic_jsd", "valid_diversity", "censored_rate", "black_rate", "uncensored_accuracy"]

MODEL_LABELS = {"pixart_sigma": r"PixArt-$\Sigma$", "sd15": "SD 1.5", "sdxl": "SDXL"}
CATEGORY_LABELS = {"color_binding": "color binding", "count": "count",
                   "existence": "existence", "spatial": "spatial"}


def validate_scores(scores, cfg, scenes):
    keys = ["model", "case_id", "variant_id", "seed"]
    expected = {(m["name"], s["case_id"], v["variant_id"], j["seed"]): (s, v, j) for m,s,v,j in jobs(cfg, scenes)}
    observed = set(scores[keys].itertuples(index=False, name=None))
    if scores.duplicated(keys).any() or observed != set(expected):
        raise ValueError(f"Grid is not complete: {len(set(expected)-observed)} missing, {len(observed-set(expected))} extra; no final table emitted")
    for row in scores.to_dict("records"):
        scene, variant, job = expected[tuple(row[k] for k in keys)]
        state = str(row["semantic_state"])
        if row["category"] != scene["category"] or row["kind"] != variant["kind"] or row["image_path"] != job["path"]:
            raise ValueError("Score identity disagrees with frozen protocol")
        if json.loads(row["atom_keys"]) != atom_keys(variant["requirements"]):
            raise ValueError("Mismatched atomic requirements")
        if len(state) != len(atom_keys(variant["requirements"])) or not set(state) <= {"0", "1"}:
            raise ValueError("Malformed semantic state")
        if row["all_correct"] != int(all(x == "1" for x in state)):
            raise ValueError("Joint correctness disagrees with atomic state")


def case_metrics(group, embeddings):
    group = group.sort_values(["variant_id", "seed"])
    variants, seeds = sorted(group.variant_id.unique()), sorted(group.seed.unique())
    if len(group) != len(variants)*len(seeds) or group.duplicated(["variant_id", "seed"]).any():
        raise ValueError("Incomplete case grid")
    states = group.semantic_state.astype(str).tolist()
    if len({len(x) for x in states}) != 1:
        raise ValueError("Inconsistent semantic-state lengths")
    q = np.array([[int(x) for x in state] for state in states]).reshape(len(variants), len(seeds), -1)
    a = q.prod(axis=-1)
    result = {k: str(group.iloc[0][k]) for k in ("model", "case_id", "category")}
    result.update(decomposition(q))
    result.update(mean_accuracy=float(a.mean()), empirical_worst_accuracy=float(a.mean(axis=1).min()),
                  crossfit_worst_accuracy=crossfit_worst(a), all_wordings_accuracy=float(a.min(axis=0).mean()),
                  semantic_jsd=state_jsd(q))
    values, pairs = [], 0
    for _, wording in group.groupby("variant_id"):
        paths = wording.loc[wording.all_correct == 1, "image_path"].tolist()
        if embeddings and any(p not in embeddings for p in paths):
            raise ValueError("Missing eligible embeddings")
        if len(paths) >= 2 and embeddings:
            values.append(mean_pairwise_cosine_distance(np.stack([embeddings[p] for p in paths])))
            pairs += len(paths)*(len(paths)-1)//2
    result["valid_diversity"] = float(np.mean(values)) if values else np.nan
    result["valid_diversity_pairs"] = pairs
    result["valid_diversity_wordings"] = len(values)
    censored = group.get("censored", pd.Series(0, index=group.index)).astype(bool)
    black = group.get("black_image", pd.Series(0, index=group.index)).astype(bool)
    result["censored_rate"], result["black_rate"] = float(censored.mean()), float(black.mean())
    valid = ~(censored | black)
    result["uncensored_accuracy"] = float(group.loc[valid, "all_correct"].mean()) if valid.any() else np.nan
    return result


def save_csv(path, data):
    with atomic_path(path) as tmp:
        data.to_csv(tmp, index=False)


def table_fragment(summary, models):
    labels = [("mean_accuracy", "Mean accuracy"), ("empirical_worst_accuracy", "Empirical worst"),
              ("crossfit_worst_accuracy", "Held-out selected worst"), ("all_wordings_accuracy", "All-wordings accuracy"),
              ("wording_disagreement", "Wording disagreement"), ("seed_disagreement", "Seed disagreement"),
              ("variance_wording", "$V_w$"), ("variance_seed", "$V_s$"), ("variance_interaction", "$V_{ws}$"),
              ("censored_rate", "Censoring rate"), ("valid_diversity", "Valid diversity")]
    # Vertically stacked panels avoid shrinking a seven-column table below 9 pt.
    lines = []
    for model in models:
        safe = MODEL_LABELS.get(model, model.replace("_", r"\_"))
        lines += [r"\textbf{" + safe + r"}\\", r"\begin{tabular}{@{}lr@{}}", r"\toprule",
                  r"Metric & Estimate [95\% CI] \\", r"\midrule"]
        for metric, label in labels:
            row = summary[(summary.model == model) & (summary.category == "all") & (summary.metric == metric)].iloc[0]
            value = f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]" if np.isfinite(row.estimate) else "undefined"
            lines.append(label + " & " + value + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}\par\medskip"]
    return "\n".join(lines)+"\n"


def main_table(summary, models):
    metrics = [("mean_accuracy", "Mean accuracy"), ("empirical_worst_accuracy", "Empirical worst accuracy"),
               ("crossfit_worst_accuracy", "Held-out selected-worst accuracy"),
               ("variance_wording", "$V_w$"), ("variance_seed", "$V_s$"), ("variance_interaction", "$V_{ws}$"),
               ("censored_rate", "Censoring rate")]
    lines = [r"\begin{tabular}{@{}l" + "c"*len(models) + r"@{}}", r"\toprule",
             "Metric & " + " & ".join(MODEL_LABELS.get(m, m.replace("_", r"\_")) for m in models) + r" \\", r"\midrule"]
    for metric,label in metrics:
        cells = []
        for model in models:
            row = summary[(summary.model == model)&(summary.category == "all")&(summary.metric == metric)].iloc[0]
            cells.append(f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]" if np.isfinite(row.estimate) else "undefined")
        lines.append(label+" & "+" & ".join(cells)+r" \\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n"


def analyze(cfg, scenes, scores, root=None, bootstrap=2000):
    validate_scores(scores, cfg, scenes)
    run_root = Path(cfg["output_dir"])
    root = Path(root) if root else run_root / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    with np.load(run_root / "dino_embeddings.npz", allow_pickle=False) as a:
        if len(a["paths"]) != len(set(a["paths"].tolist())) or set(a["paths"].tolist()) != set(scores.image_path):
            raise ValueError("Embedding coverage is not exactly the frozen grid")
        if not np.isfinite(a["embeddings"]).all():
            raise ValueError("Non-finite embeddings")
        embeddings = dict(zip(a["paths"].tolist(), a["embeddings"]))
    eq = scores[scores.kind == "equivalent"]
    cases = pd.DataFrame([case_metrics(g, embeddings) for _, g in eq.groupby(["model", "case_id"])])
    save_csv(root / "case_metrics.csv", cases)
    rng = np.random.default_rng(2027)
    rows = []
    for model, group in cases.groupby("model"):
        for category in ["all", *sorted(group.category.unique())]:
            part = group if category == "all" else group[group.category == category]
            for metric in METRICS:
                estimate, low, high = stratified_ci(part[metric], part.category, rng, bootstrap)
                rows.append(dict(model=model, category=category, metric=metric, estimate=estimate,
                                 ci_low=low, ci_high=high, n_scenes=len(part), n_finite=int(np.isfinite(part[metric]).sum())))
    summary = pd.DataFrame(rows)
    save_csv(root / "summary.csv", summary)
    differences = []
    for left, right in combinations(sorted(cases.model.unique()), 2):
        pair = cases[cases.model == left].merge(cases[cases.model == right], on=["case_id", "category"], suffixes=("_l", "_r"), validate="one_to_one")
        for metric in METRICS:
            delta, low, high = stratified_ci(pair[metric+"_l"]-pair[metric+"_r"], pair.category, rng, bootstrap)
            differences.append(dict(left=left, right=right, metric=metric, delta=delta, ci_low=low, ci_high=high))
    save_csv(root / "paired_model_differences.csv", pd.DataFrame(differences))
    control_rows = []
    for (model, case), group in scores.groupby(["model", "case_id"]):
        a = group[group.variant_id == "p0"].set_index("seed")
        b = group[group.variant_id == "cf0"].set_index("seed").reindex(a.index)
        exclusive = bool(group.exclusive_control.iloc[0])
        success = ((a.all_correct == 1) & (a.alternate_correct == 0) & (b.all_correct == 1) & (b.alternate_correct == 0)) if exclusive else None
        control_rows.append(dict(model=model, case_id=case, category=group.category.iloc[0],
             own_control_accuracy=float(b.all_correct.mean()), exclusive_control=exclusive,
             paired_discrimination=float(success.mean()) if exclusive else np.nan))
    save_csv(root / "counterfactual_controls.csv", pd.DataFrame(control_rows))
    timings = scores.groupby("model").agg(images=("image_path", "size"),
        generation_seconds=("elapsed_seconds", "sum"), median_seconds_per_image=("elapsed_seconds", "median"),
        peak_allocated_bytes=("peak_allocated_bytes", "max"), censored=("censored", "sum"), black=("black_image", "sum"))
    save_csv(root / "runtime.csv", timings.reset_index())
    table = table_fragment(summary, sorted(cases.model.unique()))
    with atomic_path(root / "results_table.tex") as tmp:
        tmp.write_text(table, encoding="utf-8")
    with atomic_path(root / "main_results_table.tex") as tmp:
        tmp.write_text(main_table(summary, sorted(cases.model.unique())), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 3.2))
    data = summary[(summary.metric == "empirical_worst_accuracy") & (summary.category != "all")]
    models = sorted(data.model.unique())
    for i, model in enumerate(models):
        g = data[data.model == model].sort_values("category")
        x = np.arange(len(g)) + (i-(len(models)-1)/2)*0.22
        ax.errorbar(x, g.estimate,
                    yerr=np.maximum(0, np.array([g.estimate-g.ci_low, g.ci_high-g.estimate])),
                    fmt="o", capsize=3, label=MODEL_LABELS.get(model, model))
    ax.set_xticks(np.arange(len(g)), [CATEGORY_LABELS.get(name, name) for name in g.category], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Empirical worst-wording accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "category_robust_accuracy.pdf")
    plt.close(fig)
    scene_splits = sorted({str(scene.get("split", "unspecified")) for scene in scenes})
    save_json(root / "analysis_info.json", {"protocol": cfg["protocol"], "bootstrap": bootstrap,
         "bootstrap_seed": 2027, "resampling": "within-category scene bootstrap; equal category weights",
         "inference": "conditional on tested wordings, seeds, evaluator; no multiplicity-adjusted tests",
         "semantic_label_source": "automatic OWLv2/CLIP evaluator",
         "semantic_result_role": "secondary diagnostic; human audit is primary",
         "scores_sha256": stable_hash(scores.fillna("").to_dict("records")),
         "dataset": str(cfg.get("dataset", "in-memory-test-fixture")),
         "dataset_split": scene_splits[0] if len(scene_splits) == 1 else scene_splits,
         "models": sorted(scores.model.unique().tolist()), "scene_count": len(scenes),
         "seed_count": len(cfg["seeds"]), "scored_images": len(scores),
         "images_per_model": {
             model: int(count) for model, count in scores.model.value_counts().sort_index().items()
         },
         "submission_ready": False})
    return summary
