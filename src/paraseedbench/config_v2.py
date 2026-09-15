"""Write a new hardware profile; never mutate a running configuration."""
import argparse
from pathlib import Path
import yaml

from .protocol_v2 import PROTOCOL
from .storage_v2 import atomic_path


def profile(hardware, split):
    cuda_resident = hardware == "rtx6000ada"
    models = [
        dict(name="sd15", id="stable-diffusion-v1-5/stable-diffusion-v1-5", steps=30, guidance_scale=7.5),
        dict(name="sdxl", id="stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", steps=30, guidance_scale=7.5),
        # The 512-MS repository contains only the denoising transformer.  The
        # complete 1024-MS repository supplies the compatible T5/VAE/scheduler
        # components; both Hub revisions are frozen before a run.
        dict(name="pixart_sigma", id="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
             pipeline_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
             steps=20, guidance_scale=4.5, clean_caption=False)]
    for model in models:
        model.update(height=512, width=512, use_safetensors=True, attention_slicing=False,
                     vae_slicing=False, vae_tiling=False, xformers=False,
                     offload="none" if cuda_resident else ("sequential" if model["name"] == "pixart_sigma" else "model"))
    return dict(protocol=PROTOCOL, dataset=f"data/scenes_v2_{split}.jsonl",
                output_dir=f"outputs/v2_{split}_v030_{hardware}",
                seeds=[11, 29, 47, 71, 101, 131, 173, 211] if split == "main" else [1009, 1013],
                full_determinism=True, negative_prompt="", models=models,
                evaluation=dict(detector=dict(id="google/owlv2-base-patch16-ensemble", use_fast=False),
                    color_model=dict(id="openai/clip-vit-base-patch32"),
                    embedding_model=dict(id="facebook/dinov2-small", use_fast=False),
                    threshold=0.10, nms_threshold=0.35, relation_margin=0.03, embedding_batch_size=8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hardware", choices=["rtx3070", "rtx6000ada"], default="rtx6000ada")
    p.add_argument("--split", choices=["main", "dev", "smoke"], default="main")
    p.add_argument("--output", required=True)
    p.add_argument("--output-dir", help="Override the run directory (use a new one after any contract change)")
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Config exists; reuse it to resume or choose a new name")
    cfg = profile(args.hardware, args.split)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    with atomic_path(target) as tmp:
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
