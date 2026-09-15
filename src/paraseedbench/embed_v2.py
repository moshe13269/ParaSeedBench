from __future__ import annotations

import gc
import zipfile
from pathlib import Path
import numpy as np

from .io import stable_hash
from .storage_v2 import atomic_path


def load_chunk(path, token, expected_paths):
    try:
        with np.load(path, allow_pickle=False) as archive:
            vectors = archive["embeddings"]
            if str(archive["token"].item()) != token or archive["paths"].tolist() != expected_paths:
                return None
            if vectors.ndim != 2 or len(vectors) != len(expected_paths) or not np.isfinite(vectors).all():
                return None
            return vectors
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def embed(cfg, scores, run_id):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from PIL import Image
    from tqdm import tqdm

    root = Path(cfg["output_dir"])
    spec = cfg["evaluation"]["embedding_model"]
    # Include every output, even invalid outputs, so coverage checks are exact.
    paths = scores["image_path"].astype(str).tolist()
    image_hashes = scores["image_sha256"].tolist()
    batch_size = cfg["evaluation"].get("embedding_batch_size", 8)
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    chunks, model, processor = [], None, None
    try:
        for start in tqdm(range(0, len(paths), batch_size), desc="DINO batches"):
            batch_paths = paths[start:start + batch_size]
            token = stable_hash({"run_id": run_id, "model": spec, "dtype": "float32",
                                 "paths": batch_paths, "images": image_hashes[start:start + batch_size]})
            target = root / "embedding_chunks" / f"{start:07d}.npz"
            vectors = load_chunk(target, token, batch_paths)
            if vectors is None:
                if model is None:
                    processor = AutoImageProcessor.from_pretrained(
                        spec["id"], revision=spec["revision"],
                        use_fast=bool(spec.get("use_fast", False))
                    )
                    model = AutoModel.from_pretrained(
                        spec["id"], revision=spec["revision"], dtype=torch.float32
                    ).to("cuda").eval()
                images = []
                for path in batch_paths:
                    with Image.open(root / path) as im:
                        images.append(im.convert("RGB"))
                inputs = {k: v.to("cuda") for k, v in processor(images=images, return_tensors="pt").items()}
                with torch.inference_mode():
                    features = model(**inputs).last_hidden_state[:, 0, :].float()
                    vectors = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
                with atomic_path(target) as tmp:
                    np.savez_compressed(tmp, token=np.asarray(token), paths=np.asarray(batch_paths), embeddings=vectors)
            chunks.append(vectors)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with atomic_path(root / "dino_embeddings.npz") as tmp:
        np.savez_compressed(tmp, paths=np.asarray(paths), embeddings=np.concatenate(chunks))
