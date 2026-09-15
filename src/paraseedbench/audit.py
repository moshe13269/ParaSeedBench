from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified, blinded human-audit sheet")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    data = pd.read_csv(args.scores, dtype={"semantic_state": str})
    rng = np.random.default_rng(args.seed)
    strata = ["model", "category", "all_correct"]
    groups = list(data.groupby(strata, dropna=False))
    base = max(1, args.n // max(1, len(groups)))
    selected_indices: list[int] = []
    for _, group in groups:
        take = min(len(group), base)
        selected_indices.extend(
            group.sample(take, random_state=int(rng.integers(0, 2**31 - 1))).index.tolist()
        )
    remaining_n = min(args.n - len(selected_indices), len(data) - len(selected_indices))
    if remaining_n > 0:
        remaining = data.drop(index=selected_indices)
        selected_indices.extend(
            remaining.sample(remaining_n, random_state=int(rng.integers(0, 2**31 - 1))).index.tolist()
        )
    sample = data.loc[selected_indices].copy() if selected_indices else data.head(0).copy()
    sample = sample.sample(frac=1, random_state=args.seed).head(args.n).reset_index(drop=True)
    sample.insert(0, "audit_id", [f"audit_{i:04d}" for i in range(len(sample))])
    output = sample[["audit_id", "image_path", "category", "atom_keys"]].copy()
    output["human_all_correct"] = ""
    output["human_atom_state"] = ""
    output["notes"] = ""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(target, index=False)
    key = sample[["audit_id", "model", "case_id", "variant_id", "seed", "all_correct", "semantic_state"]]
    key.to_csv(target.with_name(target.stem + "_machine_key.csv"), index=False)
    print(f"Wrote {len(output)} blinded audit rows to {target}")


if __name__ == "__main__":
    main()
