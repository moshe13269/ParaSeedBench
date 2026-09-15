from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    observed = float(np.mean(a == b))
    pa = float(np.mean(a))
    pb = float(np.mean(b))
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else np.nan


def agreement(machine: np.ndarray, human: np.ndarray) -> dict[str, float | int]:
    machine = np.asarray(machine, dtype=int)
    human = np.asarray(human, dtype=int)
    if machine.shape != human.shape or machine.ndim != 1 or len(machine) == 0:
        raise ValueError("Agreement inputs must be non-empty one-dimensional arrays of equal length")
    if not (np.isin(machine, [0, 1]).all() and np.isin(human, [0, 1]).all()):
        raise ValueError("Agreement inputs must be binary")
    tp = int(np.sum((machine == 1) & (human == 1)))
    tn = int(np.sum((machine == 0) & (human == 0)))
    fp = int(np.sum((machine == 1) & (human == 0)))
    fn = int(np.sum((machine == 0) & (human == 1)))
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n": len(machine), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(np.mean(machine == human)),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2
            if np.isfinite(recall) and np.isfinite(specificity) else np.nan,
        "cohen_kappa": kappa(machine, human),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score human audit against machine labels")
    parser.add_argument("--human", required=True)
    parser.add_argument("--machine-key", required=True)
    parser.add_argument("--second-human", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    human = pd.read_csv(args.human)
    key = pd.read_csv(args.machine_key)
    merged = human.merge(key, on="audit_id", validate="one_to_one")
    merged = merged.dropna(subset=["human_all_correct"])
    human_labels = merged["human_all_correct"].astype(int).to_numpy()
    report: dict[str, object] = {
        "machine_vs_human": agreement(merged["all_correct"].astype(int).to_numpy(), human_labels)
    }
    if args.second_human:
        second = pd.read_csv(args.second_human)[["audit_id", "human_all_correct"]].rename(columns={"human_all_correct": "human_2"})
        inter = merged.merge(second, on="audit_id", validate="one_to_one").dropna(subset=["human_2"])
        report["human_vs_human"] = agreement(
            inter["human_all_correct"].astype(int).to_numpy(), inter["human_2"].astype(int).to_numpy()
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
