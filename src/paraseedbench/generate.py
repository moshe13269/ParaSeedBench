from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from .determinism import prepare_environment


def select_scenes(scenes: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    selected = scenes
    categories = cfg.get("categories")
    if categories:
        selected = [s for s in selected if s["category"] in set(categories)]
    per_category = cfg.get("max_scenes_per_category")
    if per_category is not None:
        kept: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for scene in selected:
            category = scene["category"]
            if seen.get(category, 0) < int(per_category):
                kept.append(scene)
                seen[category] = seen.get(category, 0) + 1
        selected = kept
    max_scenes = cfg.get("max_scenes")
    if max_scenes is not None:
        selected = selected[: int(max_scenes)]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ParaSeedBench image grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from .io import load_yaml, read_jsonl, slugify, stable_hash, write_json

    cfg = load_yaml(args.config)
    full_determinism = bool(cfg.get("full_determinism", True))
    prepare_environment(full_determinism)

    import diffusers
    import torch
    import transformers
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install CUDA-enabled PyTorch before generation.")

    from .determinism import fresh_generator, seed_runtime
    from .model_registry import inference_kwargs, load_pipeline, unload_pipeline

    dataset_path = Path(cfg["dataset"])
    scenes = select_scenes(read_jsonl(dataset_path), cfg)
    output_root = Path(cfg.get("output_dir", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_path, output_root / "scenes_snapshot.jsonl")
    run_info = {
        "config": cfg,
        "config_sha256": stable_hash(cfg),
        "dataset_sha256": stable_hash(scenes),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(output_root / "run_info.json", run_info)

    include_cf = bool(cfg.get("include_counterfactuals", False))
    seeds = [int(seed) for seed in cfg["seeds"]]
    total_per_model = sum(
        1
        for scene in scenes
        for variant in scene["variants"]
        if include_cf or variant["kind"] == "equivalent"
        for _ in seeds
    )

    for model_cfg in cfg["models"]:
        model_slug = model_cfg.get("name") or slugify(model_cfg["id"])
        print(f"Loading {model_cfg['id']} as {model_slug}", flush=True)
        pipe = load_pipeline(model_cfg)
        progress = tqdm(total=total_per_model, desc=model_slug, unit="image")
        for scene in scenes:
            for variant in scene["variants"]:
                if variant["kind"] == "counterfactual" and not include_cf:
                    continue
                target_dir = output_root / "images" / model_slug / scene["case_id"] / variant["variant_id"]
                target_dir.mkdir(parents=True, exist_ok=True)
                for seed in seeds:
                    image_path = target_dir / f"seed_{seed}.png"
                    meta_path = image_path.with_suffix(".json")
                    if image_path.exists() and meta_path.exists() and not args.overwrite:
                        progress.update(1)
                        continue
                    seed_runtime(seed, full_determinism)
                    requested = {
                        "prompt": variant["prompt"],
                        "negative_prompt": cfg.get("negative_prompt") or None,
                        "num_inference_steps": int(model_cfg.get("steps", 30)),
                        "guidance_scale": float(model_cfg.get("guidance_scale", 7.5)),
                        "height": int(model_cfg.get("height", 512)),
                        "width": int(model_cfg.get("width", 512)),
                        "generator": fresh_generator(seed),
                        "num_images_per_prompt": 1,
                        "output_type": "pil",
                    }
                    requested = {k: v for k, v in requested.items() if v is not None}
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = pipe(**inference_kwargs(pipe, requested))
                    elapsed = time.perf_counter() - started
                    result.images[0].save(image_path, format="PNG", compress_level=6)
                    write_json(meta_path, {
                        "model_id": model_cfg["id"],
                        "model_name": model_slug,
                        "case_id": scene["case_id"],
                        "category": scene["category"],
                        "variant_id": variant["variant_id"],
                        "kind": variant["kind"],
                        "prompt": variant["prompt"],
                        "requirements": variant["requirements"],
                        "seed": seed,
                        "elapsed_seconds": elapsed,
                        "image_path": image_path.as_posix(),
                        "generation": {k: v for k, v in requested.items() if k not in {"generator", "prompt"}},
                        "config_sha256": run_info["config_sha256"],
                    })
                    progress.update(1)
        progress.close()
        unload_pipeline(pipe)

    print(f"Generation complete: {output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
