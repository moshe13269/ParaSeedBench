from __future__ import annotations

import gc
from typing import Any


def load_pipeline(model_cfg: dict[str, Any]):
    import torch
    from diffusers import DiffusionPipeline

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "use_safetensors": bool(model_cfg.get("use_safetensors", True)),
        "low_cpu_mem_usage": True,
    }
    for key in ("revision", "variant"):
        if model_cfg.get(key):
            kwargs[key] = model_cfg[key]
    if model_cfg.get("pipeline_id"):
        # PixArt-Sigma's 512-MS Hub repository is transformer-only (there is no
        # model_index.json).  Compose that transformer with the complete,
        # compatible 1024-MS pipeline components and pin both repositories.
        from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel

        transformer = PixArtTransformer2DModel.from_pretrained(
            model_cfg["id"], subfolder="transformer", **kwargs
        )
        pipeline_kwargs = dict(kwargs)
        pipeline_kwargs["revision"] = model_cfg["pipeline_revision"]
        pipe = PixArtSigmaPipeline.from_pretrained(
            model_cfg["pipeline_id"], transformer=transformer, **pipeline_kwargs
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(model_cfg["id"], **kwargs)

    offload = model_cfg.get("offload", "model")
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    elif offload == "none":
        pipe.to("cuda")
    else:
        raise ValueError(f"Unknown offload mode: {offload}")

    if model_cfg.get("attention_slicing", True) and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("auto")
    if model_cfg.get("vae_slicing", False) and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if model_cfg.get("vae_tiling", True) and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if model_cfg.get("xformers", False):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # optional optimization
            print(f"Warning: xFormers could not be enabled: {exc}", flush=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def unload_pipeline(pipe) -> None:
    import torch

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inference_kwargs(pipe, requested: dict[str, Any]) -> dict[str, Any]:
    """Keep only arguments accepted by a pipeline with an explicit signature."""
    import inspect

    signature = inspect.signature(pipe.__call__)
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if has_var_kwargs:
        return requested
    return {key: value for key, value in requested.items() if key in signature.parameters}
