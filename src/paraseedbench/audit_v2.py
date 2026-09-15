"""Blinded complete-grid human audit with explicit primary/secondary roles."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_v2 import MODEL_LABELS, METRICS, case_metrics, save_csv
from .io import stable_hash
from .metrics_v2 import stratified_ci
from .score_audit import agreement
from .storage_v2 import atomic_path, digest, read_json, save_json


PAPER_MIN_SCENES_PER_CATEGORY = 4
AUDIT_EXCLUDED_METRICS = {
    "valid_diversity", "censored_rate", "black_rate", "uncensored_accuracy"
}


def select_cases(scores, n, seed):
    if n < 1:
        raise ValueError("Select at least one scene per category")
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in scores[["case_id", "category"]].drop_duplicates().groupby(
        "category", sort=True
    ):
        if n > len(group):
            raise ValueError("Requested more scenes than available")
        selected.extend(rng.choice(sorted(group.case_id), size=n, replace=False).tolist())
    return selected


def _audit_html(rows, title, introduction):
    cards = []
    for row in rows:
        keys = json.loads(row["atom_keys"])
        cards.append(
            f'<article><h3>{html.escape(row["audit_id"])}</h3>'
            f'<img width="320" src="images/{html.escape(row["audit_id"])}.png">'
            f'<pre>{html.escape(chr(10).join(keys))}</pre></article>'
        )
    return (
        '<!doctype html><meta charset="utf-8">'
        f'<title>{html.escape(title)}</title>'
        '<style>body{font:16px sans-serif;max-width:760px;margin:30px auto}'
        'article{border-bottom:1px solid #ccc;padding:20px}'
        'pre{white-space:pre-wrap}</style>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(introduction)}</p>'
        + "".join(cards)
    )


def export(root, target, n, seed):
    if target.exists():
        raise FileExistsError("Do not overwrite human work. Choose a new audit directory.")
    scores = pd.read_csv(root / "semantic_scores.csv", dtype={"semantic_state": str})
    scores = scores[scores.kind == "equivalent"]
    selected = select_cases(scores, n, seed)
    sample = scores[scores.case_id.isin(selected)].sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    target.mkdir(parents=True)
    (target / "images").mkdir()
    sample["audit_id"] = [f"item_{i:05d}" for i in range(len(sample))]
    annotations = []
    for row in sample.to_dict("records"):
        filename = row["audit_id"] + ".png"
        shutil.copyfile(root / row["image_path"], target / "images" / filename)
        annotations.append({"audit_id": row["audit_id"], "human_atom_state": "", "notes": ""})
    template = pd.DataFrame(annotations)
    save_csv(target / "annotations.csv", template)
    save_csv(target / "annotations_rater_a.csv", template)
    save_csv(target / "annotations_rater_b.csv", template)
    # Machine labels and model identity are private until independent annotation ends.
    save_csv(target / "PRIVATE_machine_key.csv", sample)
    instructions = (
        "For each image, enter one 0/1 digit per requirement, in displayed order, "
        "in your own annotation CSV. 1 means satisfied and 0 means violated. Count only "
        "visible instances. For spatial relations use image coordinates, not object-facing "
        "direction. Color and relation atoms are false if a necessary object is absent. "
        "Ignore unrequested background content. Do not infer missing objects from the prompt. "
        "Record ambiguity in notes and resolve every state before scoring. Do not open the "
        "private key or another rater's labels before submitting your independent file."
    )
    with atomic_path(target / "index.html") as tmp:
        tmp.write_text(
            _audit_html(sample.to_dict("records"), "Blinded semantic audit", instructions),
            encoding="utf-8",
        )
    save_json(
        target / "selection.json",
        {
            "seed": seed,
            "scene_ids": selected,
            "scenes_per_category": n,
            "images": len(sample),
            "sampling": (
                "equal number of scenes/category; all wordings, seeds, and models; "
                "no automatic pass/fail stratification"
            ),
            "paper_minimum_scenes_per_category": PAPER_MIN_SCENES_PER_CATEGORY,
            "labeling": "two independent atomic annotations followed by blinded adjudication",
        },
    )
    print(
        f"Audit: {target}; {len(sample)} images. Give each annotator index.html, images/, "
        "and only their own rater CSV. Keep PRIVATE_machine_key.csv hidden."
    )


def checked_merge(human, key):
    required = {"audit_id", "human_atom_state"}
    if not required <= set(human.columns):
        raise ValueError(f"Human annotation file lacks columns: {sorted(required-set(human.columns))}")
    merged = key.merge(human, on="audit_id", validate="one_to_one", how="outer", indicator=True)
    if not (merged._merge == "both").all():
        raise ValueError("Audit IDs missing or extra")
    states = merged.human_atom_state.fillna("").astype(str).str.strip()
    for index, row in merged.iterrows():
        state = states.loc[index]
        if len(state) != len(json.loads(row.atom_keys)) or not set(state) <= {"0", "1"}:
            raise ValueError(
                f"Incomplete/invalid atomic annotation for {row.audit_id}; "
                "adjudicate all grids first"
            )
    merged["human_atom_state"] = states
    merged["human_all_correct"] = states.map(lambda x: int(all(c == "1" for c in x)))
    return merged.drop(columns="_merge")


def _validate_audit_key(key, selection):
    required = {
        "audit_id", "model", "case_id", "category", "variant_id", "kind", "seed",
        "semantic_state", "atom_keys", "all_correct",
    }
    if not required <= set(key.columns):
        raise ValueError(f"Private audit key lacks columns: {sorted(required-set(key.columns))}")
    if key.audit_id.duplicated().any() or key.duplicated(
        ["model", "case_id", "variant_id", "seed"]
    ).any():
        raise ValueError("Private audit key contains duplicate identities")
    if set(key.kind.astype(str)) != {"equivalent"}:
        raise ValueError("Human audit must contain equivalent wordings only")
    selected = set(map(str, selection.get("scene_ids", [])))
    if not selected or set(key.case_id.astype(str)) != selected:
        raise ValueError("Private audit key disagrees with frozen scene selection")
    if int(selection.get("images", -1)) != len(key):
        raise ValueError("Private audit key image count disagrees with selection")
    scene_categories = key[["case_id", "category"]].drop_duplicates()
    if scene_categories.case_id.duplicated().any():
        raise ValueError("A selected scene maps to multiple categories")
    expected_per_category = int(selection["scenes_per_category"])
    if set(scene_categories.groupby("category").size()) != {expected_per_category}:
        raise ValueError("Selected scene count is not balanced by category")
    models = set(key.model.astype(str))
    seeds = set(key.seed.astype(int))
    if len(models) < 1 or len(seeds) < 2:
        raise ValueError("Audit requires at least one model and two seeds")
    for (_, _), group in key.groupby(["model", "case_id"], sort=False):
        if set(group.seed.astype(int)) != seeds or group.variant_id.nunique() != 4:
            raise ValueError("Private audit key lacks a complete four-wording seed grid")
        if len(group) != 4*len(seeds):
            raise ValueError("Private audit key grid has missing or extra cells")


def _atomic_agreement(candidate_states, reference_states):
    candidate = [str(x) for x in candidate_states]
    reference = [str(x) for x in reference_states]
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("Atomic-state comparisons require equal non-empty collections")
    if any(len(a) != len(b) or not a for a, b in zip(candidate, reference)):
        raise ValueError("Atomic-state lengths disagree")
    candidate_bits = np.asarray([int(bit) for state in candidate for bit in state])
    reference_bits = np.asarray([int(bit) for state in reference for bit in state])
    bit = agreement(candidate_bits, reference_bits)
    return {
        "n_atoms": bit["n"],
        "atomic_tp": bit["tp"],
        "atomic_tn": bit["tn"],
        "atomic_fp": bit["fp"],
        "atomic_fn": bit["fn"],
        "atomic_accuracy": bit["accuracy"],
        "atomic_precision": bit["precision"],
        "atomic_recall": bit["recall"],
        "atomic_specificity": bit["specificity"],
        "atomic_balanced_accuracy": bit["balanced_accuracy"],
        "atomic_cohen_kappa": bit["cohen_kappa"],
        "atomic_exact_agreement": float(
            np.mean([a == b for a, b in zip(candidate, reference)])
        ),
        "mean_image_atomic_accuracy": float(
            np.mean([
                np.mean([x == y for x, y in zip(a, b)])
                for a, b in zip(candidate, reference)
            ])
        ),
    }


def comparison_report(
    frame,
    candidate_joint,
    reference_joint,
    candidate_state,
    reference_state,
    candidate_name,
    reference_name,
):
    candidate = frame[candidate_joint].to_numpy(int)
    reference = frame[reference_joint].to_numpy(int)
    result = agreement(candidate, reference)
    result.update(_atomic_agreement(frame[candidate_state], frame[reference_state]))
    result[f"{candidate_name}_joint_positive_rate"] = float(candidate.mean())
    result[f"{reference_name}_joint_positive_rate"] = float(reference.mean())
    return result


def grouped_comparisons(
    frame,
    candidate_joint,
    reference_joint,
    candidate_state,
    reference_state,
    candidate_name,
    reference_name,
):
    rows = []

    def append(scope, model, category, group):
        row = comparison_report(
            group, candidate_joint, reference_joint, candidate_state, reference_state,
            candidate_name, reference_name,
        )
        row.update(scope=scope, model=model, category=category)
        rows.append(row)

    for (model, category), group in frame.groupby(["model", "category"], sort=True):
        append("model_category", model, category, group)
    for model, group in frame.groupby("model", sort=True):
        append("model_all", model, "all", group)
    for category, group in frame.groupby("category", sort=True):
        append("all_models_category", "all", category, group)
    append("overall", "all", "all", frame)
    return pd.DataFrame(rows)


def _write_adjudication_material(target, disagreements):
    blank = target / "adjudication_blank.csv"
    if not blank.exists():
        save_csv(
            blank,
            pd.DataFrame({
                "audit_id": disagreements.audit_id,
                "human_atom_state": "",
                "notes": "",
            }),
        )
    intro = (
        "Independently decide the final atomic state for each disagreement. Use one binary "
        "digit per displayed requirement. Do not inspect automatic labels. Copy "
        "adjudication_blank.csv to human_adjudicated.csv and fill every listed row."
    )
    with atomic_path(target / "adjudication.html") as tmp:
        tmp.write_text(
            _audit_html(disagreements.to_dict("records"), "Blinded adjudication", intro),
            encoding="utf-8",
        )


def _apply_adjudication(first, second, adjudicated_path, key, target):
    disagreement_mask = first.human_atom_state.to_numpy() != second.human_atom_state.to_numpy()
    disagreements = first.loc[disagreement_mask, ["audit_id", "atom_keys"]].copy()
    if len(disagreements):
        _write_adjudication_material(target, disagreements)
    if not len(disagreements):
        return first.copy(), "identical_consensus", 0, True
    if not adjudicated_path:
        return first.copy(), "rater_a_provisional", len(disagreements), False
    adjudicated = pd.read_csv(adjudicated_path, dtype={"human_atom_state": str})
    supplied = set(adjudicated.audit_id.astype(str))
    needed = set(disagreements.audit_id.astype(str))
    all_ids = set(key.audit_id.astype(str))
    if supplied == all_ids:
        resolved = checked_merge(adjudicated, key)
        return resolved, "adjudicated_full", len(disagreements), True
    if supplied != needed:
        raise ValueError(
            "Adjudication IDs must be exactly the disagreement IDs in "
            "adjudication_blank.csv, or a complete audit file"
        )
    subset_key = key[key.audit_id.isin(needed)]
    resolved_subset = checked_merge(adjudicated, subset_key)
    resolved = first.copy()
    replacements = resolved_subset.set_index("audit_id").human_atom_state
    mask = resolved.audit_id.isin(needed)
    resolved.loc[mask, "human_atom_state"] = resolved.loc[mask, "audit_id"].map(replacements)
    resolved["human_all_correct"] = resolved.human_atom_state.map(
        lambda x: int(all(c == "1" for c in x))
    )
    return resolved, "adjudicated_disagreements", len(disagreements), True


def _case_frames(merged):
    machine_cases, human_cases = [], []
    for _, group in merged.groupby(["model", "case_id"], sort=True):
        machine_cases.append(case_metrics(group, {}))
        human = group.copy()
        human["semantic_state"] = human.human_atom_state
        human["all_correct"] = human.human_all_correct
        human_cases.append(case_metrics(human, {}))
    return pd.DataFrame(machine_cases), pd.DataFrame(human_cases)


def _metric_summary(machine_df, human_df, human_label):
    summaries = []
    rng = np.random.default_rng(2027)
    for label, frame in [("automatic_secondary", machine_df), (human_label, human_df)]:
        for model, group in frame.groupby("model", sort=True):
            for metric in METRICS:
                if metric in AUDIT_EXCLUDED_METRICS:
                    continue
                estimate, low, high = stratified_ci(group[metric], group.category, rng)
                summaries.append({
                    "labels": label,
                    "model": model,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_scenes": len(group),
                    "n_finite": int(np.isfinite(group[metric]).sum()),
                })
    return pd.DataFrame(summaries)


def _metric_differences(machine_df, human_df, human_label):
    pair = machine_df.merge(
        human_df, on=["model", "case_id", "category"], suffixes=("_automatic", "_human"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(2027)
    rows = []
    for model, group in pair.groupby("model", sort=True):
        for metric in METRICS:
            if metric in AUDIT_EXCLUDED_METRICS:
                continue
            automatic = group[f"{metric}_automatic"].astype(float)
            human = group[f"{metric}_human"].astype(float)
            delta, low, high = stratified_ci(automatic-human, group.category, rng)
            automatic_estimate = stratified_ci(
                automatic, group.category, np.random.default_rng(0), 100
            )[0]
            human_estimate = stratified_ci(
                human, group.category, np.random.default_rng(0), 100
            )[0]
            rows.append({
                "model": model,
                "metric": metric,
                "automatic_estimate": automatic_estimate,
                "human_estimate": human_estimate,
                "automatic_minus_human": delta,
                "ci_low": low,
                "ci_high": high,
                "human_label_source": human_label,
            })
    return pd.DataFrame(rows)


def _heldout_selected_worst_summary(human_df, human_label):
    """Paper sensitivity interval with a metric-local frozen RNG stream.

    Keeping this diagnostic separate makes its bootstrap reproducible even if
    other secondary metrics are added to ``METRICS`` or reordered later.
    Models are visited in lexical identifier order, as frozen for the paper.
    """
    rng = np.random.default_rng(2027)
    rows = []
    for model, group in human_df.groupby("model", sort=True):
        estimate, low, high = stratified_ci(
            group["crossfit_worst_accuracy"], group.category, rng
        )
        rows.append({
            "labels": human_label,
            "model": model,
            "metric": "heldout_selected_worst_accuracy",
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "n_scenes": len(group),
            "bootstrap": 2000,
            "bootstrap_seed": 2027,
            "split_rule": "all unordered balanced seed splits; reverse halves; average ties",
        })
    return pd.DataFrame(rows)


def _human_table(summary, models, human_label):
    metrics = [
        ("mean_accuracy", "Mean accuracy"),
        ("empirical_worst_accuracy", "Empirical worst accuracy"),
        ("all_wordings_accuracy", "All-wordings accuracy"),
        ("wording_disagreement", "Wording disagreement"),
        ("seed_disagreement", "Seed disagreement"),
        ("variance_interaction", "$V_{ws}$"),
    ]
    lines = [
        r"\begin{tabular}{@{}l" + "c"*len(models) + r"@{}}",
        r"\toprule",
        "Metric & " + " & ".join(MODEL_LABELS.get(m, m.replace("_", r"\_")) for m in models) + r" \\",
        r"\midrule",
    ]
    for metric, label in metrics:
        cells = []
        for model in models:
            row = summary[
                (summary.labels == human_label) & (summary.model == model)
                & (summary.metric == metric)
            ].iloc[0]
            cells.append(f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def _agreement_table(report, models):
    rows = report[report.scope == "model_all"].set_index("model")
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Model & Joint $\kappa$ & Joint recall & Exact state & Atomic accuracy \\",
        r"\midrule",
    ]
    for model in models:
        row = rows.loc[model]
        values = [row.cohen_kappa, row.recall, row.atomic_exact_agreement, row.atomic_accuracy]
        cells = [f"{value:.3f}" if np.isfinite(value) else "undefined" for value in values]
        lines.append(
            MODEL_LABELS.get(model, model.replace("_", r"\_"))
            + " & " + " & ".join(cells) + r" \\"
        )
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def _dataset_split(root, key):
    info_path = root / "analysis" / "analysis_info.json"
    if info_path.exists():
        return read_json(info_path).get("dataset_split", "unknown")
    prefixes = {str(case).split("_", 1)[0] for case in key.case_id.unique()}
    return next(iter(prefixes)) if len(prefixes) == 1 else "unknown"


def score(root, target, human_path, second_path=None, adjudicated_path=None):
    selection = read_json(target / "selection.json")
    key = pd.read_csv(target / "PRIVATE_machine_key.csv", dtype={"semantic_state": str})
    _validate_audit_key(key, selection)
    first = checked_merge(
        pd.read_csv(human_path, dtype={"human_atom_state": str}), key
    )
    second = None
    disagreements = 0
    resolved = False
    label_source = "rater_a_provisional"
    primary = first.copy()
    if adjudicated_path and not second_path:
        raise ValueError("--adjudicated-human requires --second-human")
    if second_path:
        second = checked_merge(
            pd.read_csv(second_path, dtype={"human_atom_state": str}), key
        )
        inter = first.merge(
            second[["audit_id", "human_all_correct", "human_atom_state"]],
            on="audit_id", suffixes=("_a", "_b"), validate="one_to_one",
        )
        inter_report = grouped_comparisons(
            inter,
            "human_all_correct_a", "human_all_correct_b",
            "human_atom_state_a", "human_atom_state_b",
            "rater_a", "rater_b",
        )
        save_csv(target / "inter_annotator_agreement.csv", inter_report)
        primary, label_source, disagreements, resolved = _apply_adjudication(
            first, second, adjudicated_path, key, target
        )

    human_label = "human_primary" if resolved else "human_provisional"
    machine_report = grouped_comparisons(
        primary,
        "all_correct", "human_all_correct", "semantic_state", "human_atom_state",
        "machine", "human",
    )
    save_csv(target / "machine_human_agreement.csv", machine_report)
    machine_df, human_df = _case_frames(primary)
    save_csv(target / "machine_audited_case_metrics.csv", machine_df)
    save_csv(target / "human_case_metrics.csv", human_df)
    summary = _metric_summary(machine_df, human_df, human_label)
    save_csv(target / "audit_metric_summary.csv", summary)
    heldout = _heldout_selected_worst_summary(human_df, human_label)
    save_csv(target / "heldout_selected_worst_summary.csv", heldout)
    differences = _metric_differences(machine_df, human_df, human_label)
    save_csv(target / "machine_human_metric_differences.csv", differences)

    models = list(dict.fromkeys(key.model.astype(str)))
    with atomic_path(target / "evaluator_agreement_table.tex") as tmp:
        tmp.write_text(_agreement_table(machine_report, models), encoding="utf-8")
    primary_table = target / "human_primary_results_table.tex"
    if resolved:
        with atomic_path(primary_table) as tmp:
            tmp.write_text(_human_table(summary, models, human_label), encoding="utf-8")
    else:
        with atomic_path(primary_table) as tmp:
            tmp.write_text(
                "% Human-primary table blocked pending a second rater and adjudication.\n",
                encoding="utf-8",
            )
    with atomic_path(target / "human_provisional_results_table.tex") as tmp:
        tmp.write_text(_human_table(summary, models, human_label), encoding="utf-8")

    split = _dataset_split(root, key)
    scenes_per_category = int(selection["scenes_per_category"])
    paper_ready = bool(
        resolved and split == "main"
        and scenes_per_category >= PAPER_MIN_SCENES_PER_CATEGORY
    )
    manifest_path = root / "run_manifest.json"
    manifest_hash = stable_hash(read_json(manifest_path)) if manifest_path.exists() else "missing"
    source_files = {"rater_a_sha256": digest(Path(human_path))}
    if second_path:
        source_files["rater_b_sha256"] = digest(Path(second_path))
    if adjudicated_path:
        source_files["adjudication_sha256"] = digest(Path(adjudicated_path))
    artifacts = {
        "private_key_sha256": digest(target / "PRIVATE_machine_key.csv"),
        "machine_human_agreement_sha256": digest(target / "machine_human_agreement.csv"),
        "audit_metric_summary_sha256": digest(target / "audit_metric_summary.csv"),
        "metric_differences_sha256": digest(target / "machine_human_metric_differences.csv"),
        "heldout_selected_worst_sha256": digest(target / "heldout_selected_worst_summary.csv"),
        "evaluator_table_sha256": digest(target / "evaluator_agreement_table.tex"),
        "human_table_sha256": digest(target / "human_primary_results_table.tex"),
    }
    if second_path:
        artifacts["inter_annotator_agreement_sha256"] = digest(
            target / "inter_annotator_agreement.csv"
        )
    save_json(
        target / "audit_info.json",
        {
            "schema": "paraseedbench-human-audit-2.1",
            "dataset_split": split,
            "run_manifest_sha256": manifest_hash,
            "selection_sha256": stable_hash(selection),
            "n_images": len(primary),
            "n_scenes": int(primary.case_id.nunique()),
            "n_models": int(primary.model.nunique()),
            "n_categories": int(primary.category.nunique()),
            "scenes_per_category": scenes_per_category,
            "minimum_paper_scenes_per_category": PAPER_MIN_SCENES_PER_CATEGORY,
            "rater_count": 2 if second_path else 1,
            "atomic_disagreements_between_raters": disagreements if second_path else None,
            "label_source": label_source,
            "human_primary_ready": resolved,
            "paper_primary_ready": paper_ready,
            "automatic_semantic_role": "secondary_diagnostic",
            "human_semantic_role": "primary" if resolved else "provisional",
            "source_files": source_files,
            "artifacts": artifacts,
        },
    )
    if paper_ready:
        print("Audit summaries written. Human-primary paper gate: READY.")
    elif not second_path:
        print(
            "Audit summaries written. Human-primary paper gate: BLOCKED; "
            "obtain an independent second annotation."
        )
    elif disagreements and not adjudicated_path:
        print(
            f"Audit summaries written. Human-primary paper gate: BLOCKED; {disagreements} "
            "atomic-state disagreements require adjudication. See adjudication.html and "
            "adjudication_blank.csv."
        )
    elif scenes_per_category < PAPER_MIN_SCENES_PER_CATEGORY:
        print(
            "Audit summaries written. Human labels are resolved, but the paper gate is "
            f"BLOCKED: use at least {PAPER_MIN_SCENES_PER_CATEGORY} scenes per category "
            "for the prespecified main audit."
        )
    else:
        print("Audit summaries written. This is not a main-split paper audit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Run root")
    parser.add_argument("--output", required=True, help="Audit directory")
    parser.add_argument("--scenes-per-category", type=int, default=PAPER_MIN_SCENES_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--human", default=None, help="Independent rater A annotation CSV")
    parser.add_argument("--second-human", default=None, help="Independent rater B annotation CSV")
    parser.add_argument(
        "--adjudicated-human", default=None,
        help="Completed adjudication CSV for disagreements, or a complete resolved audit CSV",
    )
    args = parser.parse_args()
    if args.human:
        score(
            Path(args.input), Path(args.output), args.human,
            args.second_human, args.adjudicated_human,
        )
    else:
        if args.second_human or args.adjudicated_human:
            raise ValueError("Human files can only be supplied together with --human")
        export(Path(args.input), Path(args.output), args.scenes_per_category, args.seed)


if __name__ == "__main__":
    main()
