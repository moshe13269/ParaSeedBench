from __future__ import annotations

import importlib.metadata
import platform
import os
import re
from pathlib import Path

from .io import load_yaml, read_jsonl, stable_hash
from .schema import validate_scene
from .storage_v2 import digest

PROTOCOL = "paraseedbench-2.0"


def source_hash():
    return stable_hash({p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    import torch
    packages = {}
    for package in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "safetensors", "huggingface-hub", "scipy", "pandas"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "absent"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def read_protocol(config_path):
    cfg = load_yaml(config_path)
    if cfg.get("protocol") != PROTOCOL:
        raise ValueError("Use a v2 configuration. v1 outputs are exploratory and cannot be mixed with v2.")
    scenes = read_jsonl(cfg["dataset"])
    if not scenes or len({s["case_id"] for s in scenes}) != len(scenes):
        raise ValueError("Empty or duplicate scenes")
    for scene in scenes:
        validate_scene(scene)
        if len({v["variant_id"] for v in scene["variants"]}) != len(scene["variants"]):
            raise ValueError("Duplicate variants")
        if len([v for v in scene["variants"] if v["kind"] == "equivalent"]) != 4:
            raise ValueError("v2 requires four equivalent variants")
    seeds = cfg["seeds"]
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("At least two distinct seeds required")
    names = [m["name"] for m in cfg["models"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty/duplicate model names")
    for name in names + [s["case_id"] for s in scenes]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Unsafe identifier")
    if cfg.get("frozen_dataset_sha256") != stable_hash(scenes):
        raise ValueError("Dataset not frozen or changed. Run freeze_v2 into a NEW locked config.")
    for model in cfg["models"]:
        if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
            raise ValueError("Generation model revision must be an immutable Hub commit")
        if model.get("pipeline_id") and not re.fullmatch(
            r"[0-9a-f]{40}", model.get("pipeline_revision", "")
        ):
            raise ValueError("Generation pipeline revision must be an immutable Hub commit")
    for key in ("detector", "color_model", "embedding_model"):
        if not re.fullmatch(r"[0-9a-f]{40}", cfg["evaluation"][key].get("revision", "")):
            raise ValueError("Evaluator revisions must be immutable Hub commits")
    return cfg, scenes


def jobs(cfg, scenes):
    for model in cfg["models"]:
        for scene in scenes:
            for variant in scene["variants"]:
                for seed in cfg["seeds"]:
                    rel = Path("images") / model["name"] / scene["case_id"] / variant["variant_id"] / f"seed_{seed}.png"
                    job = {"model": model["name"], "case_id": scene["case_id"], "variant_id": variant["variant_id"],
                           "seed": seed, "prompt": variant["prompt"], "requirements": variant["requirements"],
                           "path": rel.as_posix()}
                    yield model, scene, variant, job


def contract(cfg, scenes):
    # Output path is location, not experimental identity: relocation is supported.
    identity = {k: v for k, v in cfg.items() if k != "output_dir"}
    return {"protocol": PROTOCOL, "config": identity, "dataset_sha256": stable_hash(scenes),
            "source_sha256": source_hash(), "environment": environment()}
