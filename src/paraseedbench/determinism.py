from __future__ import annotations

import os
import random


def prepare_environment(full_determinism: bool) -> None:
    """Set environment flags before importing/initializing CUDA."""
    if full_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_runtime(seed: int, full_determinism: bool) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if full_determinism:
        torch.use_deterministic_algorithms(True)


def fresh_generator(seed: int):
    """A fresh CPU generator ensures the same initial noise for each wording."""
    import torch

    return torch.Generator(device="cpu").manual_seed(int(seed))
