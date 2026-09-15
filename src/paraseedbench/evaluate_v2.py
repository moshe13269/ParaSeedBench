from __future__ import annotations

import gc
from pathlib import Path

from .io import stable_hash
from .protocol_v2 import jobs
from .schema import atom_keys
from .storage_v2 import atomic_path, committed_image, read_json, save_json


def evaluate(cfg, scenes, run_id):
    import json
    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm
    from .evaluate import StructuredEvaluator

    root = Path(cfg["output_dir"])
    setting = cfg["evaluation"]
    score_id = stable_hash({"run": run_id, "evaluation": setting, "dtype": "float32"})
    rows, pending = [], []
    for model, scene, variant, base_job in jobs(cfg, scenes):
        job = dict(base_job, run_id=run_id)
        path = root / job["path"]
        if not committed_image(path, path.with_suffix(".json"), job):
            raise RuntimeError(f"Missing/corrupt generation: {path}. Resume generation first.")
        meta = read_json(path.with_suffix(".json"))
        checkpoint = root / "scores" / Path(job["path"]).relative_to("images").with_suffix(".json")
        token = {"score_id": score_id, "image_sha256": meta["image_sha256"]}
        try:
            cached = read_json(checkpoint)
            if cached["token"] == token:
                rows.append(cached["row"])
                continue
        except (OSError, ValueError, KeyError):
            pass
        pending.append((model, scene, variant, job, meta, checkpoint, token))
    print(f"Evaluation: {len(rows)} committed, {len(pending)} pending", flush=True)
    evaluator = None
    try:
        if pending:
            evaluator = StructuredEvaluator(setting["detector"]["id"], setting["color_model"]["id"], "cuda",
                setting["threshold"], setting["nms_threshold"], setting["relation_margin"],
                setting["detector"]["revision"], setting["color_model"]["revision"], fp32=True,
                use_fast=bool(setting["detector"].get("use_fast", False)))
        for model, scene, variant, job, meta, checkpoint, token in tqdm(pending, desc="semantic evaluation"):
            invalid = meta["censored"] or meta["black_image"]
            keys = atom_keys(job["requirements"])
            other_correct = None
            if invalid:
                atoms = {key: False for key in keys}
                predictions = {"not_evaluated": "censored_or_black"}
            else:
                with Image.open(root / job["path"]) as raw:
                    image = raw.convert("RGB")
                atoms, predictions = evaluator.evaluate(image, job["requirements"])
                if scene["exclusive_control"] and variant["variant_id"] in ("p0", "cf0"):
                    alt = next(v for v in scene["variants"] if v["variant_id"] == ("cf0" if variant["variant_id"] == "p0" else "p0"))
                    alternate, _ = evaluator.evaluate(image, alt["requirements"])
                    other_correct = int(all(alternate.values()))
            row = {"model": model["name"], "model_id": model["id"], "case_id": scene["case_id"],
                   "category": scene["category"], "variant_id": variant["variant_id"], "kind": variant["kind"],
                   "seed": job["seed"], "image_path": job["path"], "image_sha256": meta["image_sha256"],
                   "all_correct": int(all(atoms.values())), "atom_accuracy": sum(atoms.values()) / len(keys),
                   "semantic_state": "".join("1" if atoms[k] else "0" for k in keys),
                   "atom_keys": json.dumps(keys), "atom_results": json.dumps(atoms, sort_keys=True),
                   "predictions": json.dumps(predictions, sort_keys=True),
                   "censored": int(meta["censored"]), "black_image": int(meta["black_image"]),
                   "alternate_correct": other_correct, "exclusive_control": int(scene["exclusive_control"]),
                   "elapsed_seconds": meta["elapsed_seconds"], "peak_allocated_bytes": meta["peak_allocated_bytes"]}
            save_json(checkpoint, {"token": token, "row": row})
            rows.append(row)
    finally:
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
    data = pd.DataFrame(rows).sort_values(["model", "case_id", "variant_id", "seed"])
    with atomic_path(root / "semantic_scores.csv") as tmp:
        data.to_csv(tmp, index=False)
    save_json(root / "evaluation_info.json", {"score_id": score_id, "settings": setting,
              "dtype": "float32", "censoring_policy": "retain; end-to-end failure; report separately"})
    return data
