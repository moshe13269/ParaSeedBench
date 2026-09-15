"""CPU-only verification of the published prompts, labels, and paper numbers."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_v2 import (
    _case_frames,
    _heldout_selected_worst_summary,
    checked_merge,
    grouped_comparisons,
)
from .io import load_yaml, read_jsonl, stable_hash


EXPECTED_MAIN_SEEDS = [11, 29, 47, 71, 101, 131, 173, 211]
EXPECTED_HUMAN = {
    "pixart_sigma": {
        "mean_accuracy": 0.634765625,
        "empirical_worst_accuracy": 0.484375,
        "heldout_selected_worst_accuracy": 0.5796316964285714,
        "all_wordings_accuracy": 0.4140625,
        "wording_disagreement": 0.1657986111111111,
        "seed_disagreement": 0.20414806547619044,
        "variance_wording": 0.010233561197916666,
        "variance_seed": 0.037373860677083336,
        "variance_interaction": 0.05194091796875,
    },
    "sd15": {
        "mean_accuracy": 0.357421875,
        "empirical_worst_accuracy": 0.234375,
        "heldout_selected_worst_accuracy": 0.3472284226190476,
        "all_wordings_accuracy": 0.15625,
        "wording_disagreement": 0.2709418402777778,
        "seed_disagreement": 0.3284505208333333,
        "variance_wording": 0.018437703450520832,
        "variance_seed": 0.0605316162109375,
        "variance_interaction": 0.08316548665364584,
    },
    "sdxl": {
        "mean_accuracy": 0.33203125,
        "empirical_worst_accuracy": 0.203125,
        "heldout_selected_worst_accuracy": 0.3146205357142857,
        "all_wordings_accuracy": 0.1328125,
        "wording_disagreement": 0.2645399305555555,
        "seed_disagreement": 0.3952752976190476,
        "variance_wording": 0.015848795572916664,
        "variance_seed": 0.08957926432291667,
        "variance_interaction": 0.08335367838541667,
    },
}
EXPECTED_RHO_INTERVALS = {
    "pixart_sigma": (0.5167136346726191, 0.6545786830357142),
    "sd15": (0.2482035900297619, 0.4319986979166666),
    "sdxl": (0.24457961309523812, 0.38274646577380933),
}
EXPECTED_M96_COMPONENTS = {
    "pixart_sigma": (0.01430511474609375, 0.045138041178385414, 0.060761345757378474),
    "sd15": (0.02292293972439236, 0.0584623548719618, 0.08027394612630208),
    "sdxl": (0.0152240329318576, 0.0731158786349826, 0.0719816419813368),
}


def _assert_close(actual, expected, label, atol=1e-12):
    if not np.isclose(float(actual), float(expected), rtol=0, atol=atol):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def _verify_checksums(root: Path) -> None:
    checksum_file = root / "paper_artifacts" / "SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(checksum_file)
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / "paper_artifacts" / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Checksum mismatch: {path}")


def _resolved_human_grid(audit: Path):
    key = pd.read_csv(
        audit / "audit_key_after_unblinding.csv", dtype={"semantic_state": str}
    )
    first = checked_merge(
        pd.read_csv(audit / "human_a.csv", dtype={"human_atom_state": str}), key
    )
    second = checked_merge(
        pd.read_csv(audit / "human_b.csv", dtype={"human_atom_state": str}), key
    )
    disagreement = first.human_atom_state.to_numpy() != second.human_atom_state.to_numpy()
    if int(disagreement.sum()) != 144:
        raise AssertionError("Expected 144 pre-adjudication image-state disagreements")
    adjudicated = pd.read_csv(
        audit / "human_adjudicated.csv", dtype={"human_atom_state": str}
    )
    needed = set(first.loc[disagreement, "audit_id"])
    if set(adjudicated.audit_id) != needed:
        raise AssertionError("Adjudication IDs are not exactly the disagreement IDs")
    resolved = first.copy()
    replacements = adjudicated.set_index("audit_id").human_atom_state
    mask = resolved.audit_id.isin(needed)
    resolved.loc[mask, "human_atom_state"] = resolved.loc[mask, "audit_id"].map(replacements)
    resolved["human_all_correct"] = resolved.human_atom_state.map(
        lambda state: int(all(bit == "1" for bit in state))
    )
    return key, first, second, resolved


def _verify_human_results(root: Path) -> None:
    audit = root / "paper_artifacts" / "human_audit"
    key, first, second, resolved = _resolved_human_grid(audit)
    if len(key) != 1536 or key.case_id.nunique() != 16 or key.model.nunique() != 3:
        raise AssertionError("Human audit is not the frozen 1,536-image complete-grid sample")
    if sorted(key.seed.unique().tolist()) != EXPECTED_MAIN_SEEDS:
        raise AssertionError("Human-audit seeds differ from the frozen main seeds")

    _, cases = _case_frames(resolved)
    heldout = _heldout_selected_worst_summary(cases, "human_primary").set_index("model")
    for model, expected in EXPECTED_HUMAN.items():
        group = cases[cases.model == model]
        if len(group) != 16:
            raise AssertionError(f"{model}: expected 16 audited scenes")
        for metric, value in expected.items():
            actual = (
                heldout.loc[model, "estimate"]
                if metric == "heldout_selected_worst_accuracy"
                else group[metric].mean()
            )
            _assert_close(actual, value, f"{model}/{metric}")
        low, high = EXPECTED_RHO_INTERVALS[model]
        _assert_close(heldout.loc[model, "ci_low"], low, f"{model}/R_ho low")
        _assert_close(heldout.loc[model, "ci_high"], high, f"{model}/R_ho high")

    inter = first.merge(
        second[["audit_id", "human_all_correct", "human_atom_state"]],
        on="audit_id", suffixes=("_a", "_b"), validate="one_to_one",
    )
    report = grouped_comparisons(
        inter, "human_all_correct_a", "human_all_correct_b",
        "human_atom_state_a", "human_atom_state_b", "rater_a", "rater_b",
    )
    overall = report[report.scope == "overall"].iloc[0]
    _assert_close(overall.accuracy, 0.96875, "inter-rater joint accuracy")
    _assert_close(overall.cohen_kappa, 0.9368839533820607, "inter-rater joint kappa")
    _assert_close(overall.atomic_accuracy, 0.9481770833333333, "inter-rater atomic accuracy")
    _assert_close(overall.atomic_cohen_kappa, 0.8906217794572311, "inter-rater atomic kappa")

    machine = pd.read_csv(audit / "machine_human_agreement.csv")
    expected_machine = {
        "pixart_sigma": (0.724609375, 0.4660828920510014, 0.8125),
        "sd15": (0.833984375, 0.6196735064844269, 0.80390625),
        "sdxl": (0.849609375, 0.6402000511079473, 0.8171875),
    }
    for model, expected in expected_machine.items():
        row = machine[(machine.scope == "model_all") & (machine.model == model)].iloc[0]
        for column, value in zip(("accuracy", "cohen_kappa", "atomic_accuracy"), expected):
            _assert_close(row[column], value, f"machine-human {model}/{column}")
    shifts = pd.read_csv(audit / "machine_human_metric_differences.csv")
    shifts = shifts[shifts.metric == "mean_accuracy"].set_index("model")
    for model, value in {
        "pixart_sigma": -0.193359375,
        "sd15": -0.080078125,
        "sdxl": -0.076171875,
    }.items():
        _assert_close(
            shifts.loc[model, "automatic_minus_human"], value,
            f"machine-minus-human mean accuracy/{model}",
        )


def _verify_automatic_results(root: Path) -> None:
    automatic = root / "paper_artifacts" / "automatic_main"
    summary = pd.read_csv(automatic / "summary.csv")
    for model, expected in EXPECTED_M96_COMPONENTS.items():
        for metric, value in zip(
            ("variance_wording", "variance_seed", "variance_interaction"), expected
        ):
            row = summary[
                (summary.model == model) & (summary.category == "all")
                & (summary.metric == metric)
            ]
            if len(row) != 1:
                raise AssertionError(f"Missing M96 summary row: {model}/{metric}")
            _assert_close(row.iloc[0].estimate, value, f"M96 {model}/{metric}")
    runtime = pd.read_csv(automatic / "runtime.csv").set_index("model")
    if not (runtime.images == 3840).all() or int(runtime.censored.sum()) != 19:
        raise AssertionError("Main automatic grid/runtime counts differ from the paper")
    controls = pd.read_csv(automatic / "counterfactual_controls.csv")
    eligible = controls[controls.exclusive_control.astype(bool)]
    expected_successes = {"pixart_sigma": 37, "sd15": 18, "sdxl": 5}
    for model, expected in expected_successes.items():
        group = eligible[eligible.model == model]
        if len(group) != 72:
            raise AssertionError(f"{model}: expected 72 exclusive control scenes")
        successes = int(round(float(group.paired_discrimination.sum() * 8)))
        if successes != expected:
            raise AssertionError(
                f"{model}: expected {expected}/576 paired discrimination successes, "
                f"found {successes}/576"
            )


def _verify_protocol(root: Path) -> None:
    cfg = load_yaml(root / "configs" / "main_v2.lock.yaml")
    if cfg["seeds"] != EXPECTED_MAIN_SEEDS:
        raise AssertionError("Locked main seed list changed")
    scenes = read_jsonl(root / cfg["dataset"])
    if len(scenes) != 96 or stable_hash(scenes) != cfg["frozen_dataset_sha256"]:
        raise AssertionError("Frozen main prompt suite changed")
    if sorted({scene["category"] for scene in scenes}) != [
        "color_binding", "count", "existence", "spatial"
    ]:
        raise AssertionError("Main categories changed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify checked-in ParaSeedBench evidence")
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parents[2]),
        help="Repository root",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    _verify_checksums(root)
    _verify_protocol(root)
    _verify_human_results(root)
    _verify_automatic_results(root)
    print("PASS: frozen protocol, checksums, audited labels, and paper values agree.")
    print("This verifies released evidence; it does not regenerate model images.")


if __name__ == "__main__":
    main()
