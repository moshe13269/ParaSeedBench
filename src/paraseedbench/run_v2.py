"""Cross-platform, fail-fast entry point. One writer per output root."""
from __future__ import annotations

import argparse
from pathlib import Path

from .determinism import prepare_environment
from .io import stable_hash
from .protocol_v2 import read_protocol, contract, jobs
from .storage_v2 import run_lock, guard_manifest, committed_image, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "generate", "evaluate", "embed", "analyze", "status"], default="all")
    parser.add_argument("--clear-stale-lock", action="store_true", help="ONLY after verifying that no old process is running; remove lock and exit")
    args = parser.parse_args()
    cfg, scenes = read_protocol(args.config)
    root = Path(cfg["output_dir"])
    if args.clear_stale_lock:
        (root / ".run.lock").unlink(missing_ok=True)
        print("Removed writer lock only. Existing data retained. Run again to resume.")
        return
    if args.stage == "status":
        manifest = read_json(root / "run_manifest.json")
        run_id = stable_hash(manifest)
        counts = {}
        for model, _, _, job in jobs(cfg, scenes):
            path = root / job["path"]
            state = counts.setdefault(model["name"], [0, 0])
            state[1] += 1
            state[0] += committed_image(path, path.with_suffix(".json"), dict(job, run_id=run_id))
        for model, (done, total) in counts.items():
            print(f"{model}: {done}/{total} validated images")
        print(f"Score checkpoints: {len(list((root/'scores').rglob('seed_*.json')))} (not validated by status)")
        print(f"Embedding chunks: {len(list((root/'embedding_chunks').glob('*.npz')))} (not validated by status)")
        return
    prepare_environment(cfg["full_determinism"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. No expensive CPU fallback is performed.")
    from .determinism import seed_runtime
    seed_runtime(0, cfg["full_determinism"])
    with run_lock(root):
        if not (root / "run_manifest.json").exists() and any((root / "images").rglob("*.png")):
            raise ValueError("Images exist without a v2 run manifest. Preserve this directory and start v2 in a NEW root.")
        manifest = contract(cfg, scenes)
        guard_manifest(root / "run_manifest.json", manifest)
        run_id = stable_hash(manifest)
        if args.stage in ("all", "generate"):
            from .generate_v2 import generate
            generate(cfg, scenes, run_id)
        scores = None
        if args.stage in ("all", "evaluate", "embed", "analyze"):
            # Revalidates image commits; reuses scores per image. Never trust a stale CSV.
            from .evaluate_v2 import evaluate
            scores = evaluate(cfg, scenes, run_id)
        if args.stage in ("all", "embed", "analyze"):
            from .embed_v2 import embed
            embed(cfg, scores, run_id)
        if args.stage in ("all", "analyze"):
            from .analyze_v2 import analyze
            analyze(cfg, scenes, scores)
            print(
                f"Automatic secondary results complete: {root/'analysis'}. "
                "Human-primary audit and author review still required."
            )
            if {str(scene.get("split", "")) for scene in scenes} == {"dev"}:
                print("Development output is diagnostic only. Audit generated images against human labels before freezing main.")


if __name__ == "__main__":
    main()
