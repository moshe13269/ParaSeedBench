from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a qualitative scene x wording x seed grid")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seeds", type=int, default=4)
    parser.add_argument("--thumb", type=int, default=256)
    args = parser.parse_args()

    data = pd.read_csv(args.scores, dtype={"semantic_state": str})
    data = data[
        (data["model"] == args.model)
        & (data["case_id"] == args.case_id)
        & (data["kind"] == "equivalent")
    ].copy()
    if data.empty:
        raise ValueError("No rows match the requested model and case")
    variants = sorted(data["variant_id"].unique())
    seeds = sorted(data["seed"].unique())[: args.max_seeds]
    header = 54
    canvas = Image.new("RGB", (args.thumb * len(variants), (args.thumb + header) * len(seeds)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, seed in enumerate(seeds):
        for column_index, variant in enumerate(variants):
            row = data[(data["seed"] == seed) & (data["variant_id"] == variant)].iloc[0]
            image = Image.open(row["image_path"]).convert("RGB")
            image.thumbnail((args.thumb, args.thumb))
            x = column_index * args.thumb
            y = row_index * (args.thumb + header)
            canvas.paste(image, (x + (args.thumb - image.width) // 2, y))
            caption = f"{variant}, seed={seed}, state={row['semantic_state']}, all={int(row['all_correct'])}"
            draw.multiline_text((x + 4, y + args.thumb + 3), caption, fill="black", font=font, spacing=2)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".pdf":
        canvas.save(target, "PDF", resolution=150)
    else:
        canvas.save(target)
    metadata = {
        "model": args.model, "case_id": args.case_id,
        "variants": variants, "seeds": seeds,
    }
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote example grid to {target}")


if __name__ == "__main__":
    main()
