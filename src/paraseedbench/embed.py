from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings for valid-diversity analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModel

    root = Path(args.input)
    score_path = root / "semantic_scores.csv"
    if not score_path.exists():
        raise FileNotFoundError(f"Run semantic evaluation first: {score_path}")
    rows = pd.read_csv(score_path, dtype={"semantic_state": str})
    paths = rows["image_path"].astype(str).tolist()
    output = Path(args.output) if args.output else root / "dino_embeddings.npz"

    processor = AutoImageProcessor.from_pretrained(args.model)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), args.batch_size), desc="DINO embeddings", unit="batch"):
        batch_paths = paths[start : start + args.batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model(**inputs).last_hidden_state[:, 0, :].float()
            features = torch.nn.functional.normalize(features, dim=-1)
        chunks.append(features.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(chunks, axis=0)
    np.savez_compressed(output, paths=np.asarray(paths), embeddings=embeddings)
    output.with_suffix(".json").write_text(
        json.dumps({"model": args.model, "rows": len(paths), "dimension": int(embeddings.shape[1])}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {embeddings.shape} embeddings to {output}")


if __name__ == "__main__":
    main()

