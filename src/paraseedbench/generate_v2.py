from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path

from .io import stable_hash
from .storage_v2 import (atomic_path, committed_image, digest, encode_nonfinite_floats,
                         guard_manifest, save_json, read_json)
from .protocol_v2 import jobs


def call_kwargs(model, job, cfg):
    from .determinism import fresh_generator
    requested = {"prompt": job["prompt"], "negative_prompt": cfg.get("negative_prompt", ""),
                 "num_inference_steps": model["steps"], "guidance_scale": model["guidance_scale"],
                 "height": model["height"], "width": model["width"],
                 "generator": fresh_generator(job["seed"]), "num_images_per_prompt": 1,
                 "output_type": "pil"}
    # PixArt otherwise changes caption preprocessing according to optional
    # BeautifulSoup availability. Make the intended behavior part of the
    # frozen run contract instead of accepting an environment-dependent fallback.
    if "clean_caption" in model:
        requested["clean_caption"] = bool(model["clean_caption"])
    return requested


def repeatability_gate(pipe, model, selected, cfg, run_id, root):
    import numpy as np
    import torch
    from .determinism import seed_runtime
    from .model_registry import inference_kwargs
    path = root / "repeatability_v2" / f"{model['name']}.json"
    if path.exists():
        report = read_json(path)
        if report.get("run_id") == run_id and report.get("passed") is True:
            return
    rows = []
    seen = set()
    for _, scene, variant, job in selected:
        if scene["category"] in seen or variant["variant_id"] != "p0":
            continue
        seen.add(scene["category"])
        hashes, flags = [], []
        for _ in range(2):
            seed_runtime(job["seed"], cfg["full_determinism"])
            with torch.inference_mode():
                output = pipe(**inference_kwargs(pipe, call_kwargs(model, job, cfg)))
            array = np.asarray(output.images[0].convert("RGB"))
            hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
            flags.append(bool(array.max() == 0))
            del output
        rows.append({"case_id": scene["case_id"], "seed": job["seed"], "pixel_hashes": hashes,
                     "black": flags, "equal": hashes[0] == hashes[1]})
    passed = bool(rows) and all(r["equal"] for r in rows) and any(not any(r["black"]) for r in rows)
    save_json(path, {"run_id": run_id, "passed": passed, "checks": rows,
                    "scope": "same-process output repeatability; not cross-device or raw uncensored determinism"})
    if not passed:
        raise RuntimeError(f"Repeatability gate failed: {path}. Do not proceed with a paired-noise claim.")


def generate(cfg, scenes, run_id):
    import numpy as np
    import torch
    from tqdm import tqdm
    from .determinism import seed_runtime, fresh_generator
    from .model_registry import load_pipeline, inference_kwargs

    root = Path(cfg["output_dir"])
    all_jobs = list(jobs(cfg, scenes))
    for model in cfg["models"]:
        selected = [row for row in all_jobs if row[0]["name"] == model["name"]]
        pending = []
        for row in selected:
            job = dict(row[3], run_id=run_id)
            path = root / job["path"]
            if not committed_image(path, path.with_suffix(".json"), job):
                pending.append((row, job))
        print(f"{model['name']}: {len(selected)-len(pending)} committed, {len(pending)} pending", flush=True)
        if not pending:
            continue
        pipe = load_pipeline(model)
        pipeline_info = {"class": type(pipe).__name__, "scheduler": type(pipe.scheduler).__name__,
                         "scheduler_config": encode_nonfinite_floats(dict(pipe.scheduler.config)),
                         "safety_checker_active": getattr(pipe, "safety_checker", None) is not None,
                         "model_id": model["id"], "revision": model["revision"],
                         "pipeline_id": model.get("pipeline_id"),
                         "pipeline_revision": model.get("pipeline_revision")}
        guard_manifest(root / "pipelines" / f"{model['name']}.json", pipeline_info)
        try:
            repeatability_gate(pipe, model, selected, cfg, run_id, root)
            for (_, scene, variant, _), job in tqdm(pending, desc=model["name"], unit="image"):
                seed_runtime(job["seed"], cfg["full_determinism"])
                requested = call_kwargs(model, job, cfg)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode():
                    result = pipe(**inference_kwargs(pipe, requested))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                im = result.images[0].convert("RGB")
                if im.size != (model["width"], model["height"]):
                    raise RuntimeError("Pipeline output resolution differs from frozen protocol")
                flags = getattr(result, "nsfw_content_detected", None)
                censored = bool(flags[0]) if flags is not None else False
                suspect_black = bool(np.asarray(im).max() == 0)
                path = root / job["path"]
                with atomic_path(path) as tmp:
                    im.save(tmp, format="PNG")
                metadata = {"job": job, "image_sha256": digest(path), "category": scene["category"],
                            "kind": variant["kind"], "model_id": model["id"],
                            "pipeline_id": model.get("pipeline_id"),
                            "censored": censored, "black_image": suspect_black,
                            "pipeline_sha256": stable_hash(pipeline_info),
                            "elapsed_seconds": elapsed,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                save_json(path.with_suffix(".json"), metadata)
                if censored or suspect_black:
                    print(f"Flagged output retained: {job['path']} (censored={censored}, black={suspect_black})", flush=True)
                del result, im
        finally:
            # Release the caller's reference before the next pipeline is loaded.
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
