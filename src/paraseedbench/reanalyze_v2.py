"""Recompute derived tables from a completed, immutable v2 run.

This entry point intentionally does not reopen the generation contract.  It is
for analysis-code corrections or additional summaries after ``semantic_scores``
and DINO embeddings have already been committed.  Existing analysis products
are preserved by default in a new directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analyze_v2 import analyze
from .io import read_jsonl, stable_hash
from .storage_v2 import read_json


def _resolve_dataset(value: str, override: str | None) -> Path:
    if override:
        candidates = [Path(override)]
    else:
        requested = Path(value)
        repository_root = Path(__file__).resolve().parents[2]
        candidates = [requested]
        if not requested.is_absolute():
            candidates.append(repository_root / requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Frozen dataset was not found ({joined}). Supply --dataset explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute v2 analysis without regenerating or rescoring images"
    )
    parser.add_argument("--input", required=True, help="Completed v2 run directory")
    parser.add_argument("--dataset", help="Path to the matching frozen JSONL suite")
    parser.add_argument(
        "--output",
        help="Derived-output directory (default: INPUT/analysis_v030)",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.input).resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("protocol") != "paraseedbench-2.0":
        raise ValueError("The input is not a ParaSeedBench v2 run")
    cfg = dict(manifest["config"])
    cfg["output_dir"] = str(root)
    dataset_path = _resolve_dataset(str(cfg["dataset"]), args.dataset)
    scenes = read_jsonl(dataset_path)
    if stable_hash(scenes) != manifest.get("dataset_sha256"):
        raise ValueError("Dataset hash differs from the completed run manifest")

    scores_path = root / "semantic_scores.csv"
    embeddings_path = root / "dino_embeddings.npz"
    if not scores_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            "Re-analysis requires semantic_scores.csv and dino_embeddings.npz from "
            "the completed run"
        )
    scores = pd.read_csv(scores_path, dtype={"semantic_state": str})
    output = Path(args.output).resolve() if args.output else root / "analysis_v030"
    analyze(cfg, scenes, scores, root=output, bootstrap=args.bootstrap)
    print(f"Re-analysis complete: {output}")
    print("Generation, semantic-score, and embedding checkpoints were not modified.")


if __name__ == "__main__":
    main()
