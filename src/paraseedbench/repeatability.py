from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from .determinism import prepare_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat one identical generation with freshly reset randomness")
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-id", default="existence_000")
    parser.add_argument("--variant-id", default="p0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    from .io import load_yaml, read_jsonl, slugify, write_json

    cfg = load_yaml(args.config)
    prepare_environment(bool(cfg.get("full_determinism", True)))

    import torch

    from .determinism import fresh_generator, seed_runtime
    from .model_registry import inference_kwargs, load_pipeline, unload_pipeline

    scene = next(item for item in read_jsonl(cfg["dataset"]) if item["case_id"] == args.case_id)
    variant = next(item for item in scene["variants"] if item["variant_id"] == args.variant_id)
    candidates = cfg["models"]
    model_cfg = next(
        (item for item in candidates if item.get("name") == args.model_name), candidates[0]
    )
    model_name = model_cfg.get("name") or slugify(model_cfg["id"])
    output_dir = Path(cfg.get("output_dir", "outputs")) / "repeatability" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(model_cfg)
    hashes = []
    elapsed = []
    for repeat in range(args.repeats):
        seed_runtime(args.seed, bool(cfg.get("full_determinism", True)))
        requested = {
            "prompt": variant["prompt"],
            "num_inference_steps": int(model_cfg.get("steps", 30)),
            "guidance_scale": float(model_cfg.get("guidance_scale", 7.5)),
            "height": int(model_cfg.get("height", 512)),
            "width": int(model_cfg.get("width", 512)),
            "generator": fresh_generator(args.seed),
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }
        started = time.perf_counter()
        with torch.inference_mode():
            image = pipe(**inference_kwargs(pipe, requested)).images[0]
        elapsed.append(time.perf_counter() - started)
        array = np.asarray(image.convert("RGB"))
        hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
        image.save(output_dir / f"repeat_{repeat}.png")
    unload_pipeline(pipe)
    report = {
        "model": model_cfg["id"],
        "case_id": args.case_id,
        "variant_id": args.variant_id,
        "seed": args.seed,
        "repeats": args.repeats,
        "pixel_sha256": hashes,
        "pixel_exact_reproducibility": len(set(hashes)) == 1,
        "elapsed_seconds": elapsed,
    }
    write_json(output_dir / "report.json", report)
    print(f"Pixel-exact reproducibility: {report['pixel_exact_reproducibility']}")
    print(f"Report: {(output_dir / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
