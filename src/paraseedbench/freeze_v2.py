"""Resolve Hub commits, including judges, before either pilot or main execution."""
import argparse
from pathlib import Path

import yaml
from .io import load_yaml, read_jsonl, stable_hash
from .storage_v2 import atomic_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"Keep the existing frozen config for resume: {target}")
    from huggingface_hub import model_info
    cfg = load_yaml(args.config)
    for model in cfg["models"] + [cfg["evaluation"][k] for k in ("detector", "color_model", "embedding_model")]:
        model["revision"] = model_info(model["id"], revision=model.get("revision", "main")).sha
        if model.get("pipeline_id"):
            model["pipeline_revision"] = model_info(
                model["pipeline_id"], revision=model.get("pipeline_revision", "main")
            ).sha
    cfg["frozen_dataset_sha256"] = stable_hash(read_jsonl(cfg["dataset"]))
    with atomic_path(target) as tmp:
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Frozen: {target}. Reuse this file unchanged for all restarts.")


if __name__ == "__main__":
    main()
