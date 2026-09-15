from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any

from .io import stable_hash, write_json, write_jsonl
from .schema import validate_scene


OBJECTS = [
    ("cat", "cats"),
    ("dog", "dogs"),
    ("apple", "apples"),
    ("banana", "bananas"),
    ("bottle", "bottles"),
    ("cup", "cups"),
    ("chair", "chairs"),
    ("book", "books"),
    ("car", "cars"),
    ("bicycle", "bicycles"),
    ("ball", "balls"),
    ("clock", "clocks"),
]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
NUMBER_WORD = {2: "two", 3: "three", 4: "four"}


def article(noun: str) -> str:
    return "an" if noun[0].lower() in "aeiou" else "a"


def variants(prompts: list[str], requirements: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": f"p{index}",
            "kind": "equivalent",
            "prompt": prompt,
            "requirements": copy.deepcopy(requirements),
        }
        for index, prompt in enumerate(prompts)
    ]


def existence_scenes() -> list[dict[str, Any]]:
    scenes = []
    for index, ((a, _), (b, _)) in enumerate(list(itertools.combinations(OBJECTS, 2))[:24]):
        req = {"objects": [{"label": a, "min_count": 1}, {"label": b, "min_count": 1}]}
        prompts = [
            f"A photograph containing {article(a)} {a} and {article(b)} {b}.",
            f"Both {article(b)} {b} and {article(a)} {a} are visible in the image.",
            f"The scene includes {article(a)} {a}, together with {article(b)} {b}.",
            f"Alongside {article(b)} {b}, there is also {article(a)} {a}.",
        ]
        row = {"case_id": f"existence_{index:03d}", "category": "existence", "variants": variants(prompts, req)}
        replacement = next(name for name, _ in OBJECTS if name not in {a, b})
        row["variants"].append({
            "variant_id": "cf0",
            "kind": "counterfactual",
            "prompt": f"A photograph containing {article(a)} {a} and {article(replacement)} {replacement}.",
            "requirements": {"objects": [{"label": a, "min_count": 1}, {"label": replacement, "min_count": 1}]},
        })
        scenes.append(row)
    return scenes


def color_scenes() -> list[dict[str, Any]]:
    scenes = []
    pairs = list(itertools.combinations(OBJECTS[:8], 2))[:24]
    for index, ((a, _), (b, _)) in enumerate(pairs):
        c1 = COLORS[index % len(COLORS)]
        c2 = COLORS[(index + 2) % len(COLORS)]
        req = {
            "objects": [{"label": a, "min_count": 1}, {"label": b, "min_count": 1}],
            "colors": [{"object": a, "color": c1}, {"object": b, "color": c2}],
        }
        prompts = [
            f"A photograph of {article(c1)} {c1} {a} and {article(c2)} {c2} {b}.",
            f"The image contains {article(b)} {b} that is {c2} and {article(a)} {a} that is {c1}.",
            f"Shown together are {article(c1)} {c1} {a} and {article(c2)} {c2} {b}.",
            f"There is {article(c2)} {c2} {b}; the same scene also contains {article(c1)} {c1} {a}.",
        ]
        row = {"case_id": f"color_{index:03d}", "category": "color_binding", "variants": variants(prompts, req)}
        cf_req = copy.deepcopy(req)
        cf_req["colors"] = [{"object": a, "color": c2}, {"object": b, "color": c1}]
        row["variants"].append({
            "variant_id": "cf0",
            "kind": "counterfactual",
            "prompt": f"A photograph of {article(c2)} {c2} {a} and {article(c1)} {c1} {b}.",
            "requirements": cf_req,
        })
        scenes.append(row)
    return scenes


def count_scenes() -> list[dict[str, Any]]:
    scenes = []
    index = 0
    for singular, plural in OBJECTS[:8]:
        for count in (2, 3, 4):
            req = {"objects": [{"label": singular, "exact_count": count}]}
            word = NUMBER_WORD[count]
            prompts = [
                f"A photograph containing exactly {word} {plural}.",
                f"There are {word} {plural} in the image, no more and no fewer.",
                f"The image shows a group of precisely {word} {plural}.",
                f"Exactly {count} {plural} are visible in the scene.",
            ]
            row = {"case_id": f"count_{index:03d}", "category": "count", "variants": variants(prompts, req)}
            cf_count = 2 if count == 4 else count + 1
            row["variants"].append({
                "variant_id": "cf0",
                "kind": "counterfactual",
                "prompt": f"A photograph containing exactly {NUMBER_WORD[cf_count]} {plural}.",
                "requirements": {"objects": [{"label": singular, "exact_count": cf_count}]},
            })
            scenes.append(row)
            index += 1
    return scenes


def spatial_scenes() -> list[dict[str, Any]]:
    scenes = []
    pairs = list(itertools.combinations(OBJECTS, 2))[:12]
    index = 0
    for (a, _), (b, _) in pairs:
        for relation in ("left_of", "above"):
            if relation == "left_of":
                prompts = [
                    f"{article(a).capitalize()} {a} is to the left of {article(b)} {b}.",
                    f"{article(b).capitalize()} {b} is to the right of {article(a)} {a}.",
                    f"To the left of {article(b)} {b} sits {article(a)} {a}.",
                    f"The scene places {article(a)} {a} on the left and {article(b)} {b} on the right.",
                ]
                inverse = "right_of"
                cf_prompt = f"{article(a).capitalize()} {a} is to the right of {article(b)} {b}."
            else:
                prompts = [
                    f"{article(a).capitalize()} {a} is above {article(b)} {b}.",
                    f"{article(b).capitalize()} {b} is below {article(a)} {a}.",
                    f"Above {article(b)} {b} sits {article(a)} {a}.",
                    f"The scene places {article(a)} {a} at the top and {article(b)} {b} at the bottom.",
                ]
                inverse = "below"
                cf_prompt = f"{article(a).capitalize()} {a} is below {article(b)} {b}."
            req = {
                "objects": [{"label": a, "min_count": 1}, {"label": b, "min_count": 1}],
                "relations": [{"subject": a, "relation": relation, "object": b}],
            }
            row = {"case_id": f"spatial_{index:03d}", "category": "spatial", "variants": variants(prompts, req)}
            cf_req = copy.deepcopy(req)
            cf_req["relations"] = [{"subject": a, "relation": inverse, "object": b}]
            row["variants"].append({"variant_id": "cf0", "kind": "counterfactual", "prompt": cf_prompt, "requirements": cf_req})
            scenes.append(row)
            index += 1
    return scenes


def build() -> list[dict[str, Any]]:
    scenes = existence_scenes() + color_scenes() + count_scenes() + spatial_scenes()
    for scene in scenes:
        validate_scene(scene)
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/scenes_v1.jsonl")
    args = parser.parse_args()
    scenes = build()
    write_jsonl(args.output, scenes)
    manifest = {
        "schema_version": 1,
        "number_of_scenes": len(scenes),
        "equivalent_prompts_per_scene": 4,
        "categories": {name: sum(s["category"] == name for s in scenes) for name in sorted({s["category"] for s in scenes})},
        "sha256": stable_hash(scenes),
        "note": "Template-generated suite. Authors must manually audit every wording pair before a submission claim.",
    }
    write_json(Path(args.output).with_suffix(".manifest.json"), manifest)
    print(f"Wrote {len(scenes)} scenes to {args.output}")


if __name__ == "__main__":
    main()
