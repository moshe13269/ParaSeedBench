"""Fixed, category-balanced diagnostic suite; no generated-image-driven selection."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path

from .build_suite import OBJECTS, article, variants, NUMBER_WORD
from .io import stable_hash
from .schema import validate_scene
from .storage_v2 import atomic_path, save_json

DEV_OBJECTS = [("spoon", "spoons"), ("fork", "forks"), ("vase", "vases"), ("bus", "buses")]


def pair_indices(offsets):
    return [(i, (i + d) % len(OBJECTS)) for d in offsets for i in range(len(OBJECTS))]


def make_scene(category, index, a, b, count=2, axis="left_of", split="main"):
    aa, bb = f"{article(a)} {a}", f"{article(b)} {b}" if b else ""
    objects = [{"label": a, "min_count": 1}, {"label": b, "min_count": 1}]
    if category == "existence":
        clauses = [f"containing {aa} and {bb}", f"in which both {bb} and {aa} are visible",
                   f"showing {aa} together with {bb}", f"with {bb} as well as {aa}"]
        req = {"objects": objects}
        other = next(o for o, _ in OBJECTS if o not in (a, b))
        cf_clause = f"containing {aa} and {article(other)} {other}"
        cf = {"objects": [objects[0], {"label": other, "min_count": 1}]}
    elif category == "count":
        plural = dict(OBJECTS + DEV_OBJECTS)[a]
        word = NUMBER_WORD[count]
        clauses = [f"containing exactly {word} {plural}", f"with {word} {plural}, no more and no fewer",
                   f"showing precisely {word} {plural}", f"in which exactly {count} {plural} are visible"]
        req = {"objects": [{"label": a, "exact_count": count}]}
        cf_count = (count - 1) % 3 + 2
        cf = {"objects": [{"label": a, "exact_count": cf_count}]}
        cf_clause = f"containing exactly {NUMBER_WORD[cf_count]} {plural}"
    elif category == "color_binding":
        colors = ["red", "blue", "green", "yellow", "orange", "purple"]
        ca, cb = colors[index % 6], colors[(index + 2) % 6]
        req = {"objects": [{"label": a, "exact_count": 1}, {"label": b, "exact_count": 1}],
               "colors": [{"object": a, "color": ca}, {"object": b, "color": cb}]}
        clauses = [f"containing exactly one {a}, colored {ca}, and exactly one {b}, colored {cb}",
                   f"with exactly one {cb} {b} and exactly one {ca} {a}",
                   f"showing exactly one {a} and exactly one {b}; the {a} is {ca} and the {b} is {cb}",
                   f"in which there is exactly one {ca} {a}, along with exactly one {cb} {b}"]
        cf = copy.deepcopy(req)
        cf["colors"] = [{"object": a, "color": cb}, {"object": b, "color": ca}]
        cf_clause = f"with exactly one {cb} {a} and exactly one {ca} {b}"
    else:
        rel, inv = ("to the left of", "to the right of") if axis == "left_of" else ("above", "below")
        req = {"objects": [{"label": a, "exact_count": 1}, {"label": b, "exact_count": 1}],
               "relations": [{"subject": a, "relation": axis, "object": b}]}
        prefix = f"containing exactly one {a} and exactly one {b}; "
        clauses = [prefix + f"the {a} is {rel} the {b}", prefix + f"the {b} is {inv} the {a}",
                   prefix + f"relative to the {b}, the {a} is {rel.replace('to the ', 'on the ')}",
                   prefix + f"relative to the {a}, the {b} is {inv.replace('to the ', 'on the ')}"]
        # Vertical formulations need an explicit reference rather than dangling "above".
        if axis == "above":
            clauses[2:] = [prefix + f"the {a} is vertically higher than the {b}",
                           prefix + f"the {b} is vertically lower than the {a}"]
        cf = copy.deepcopy(req)
        cf["relations"][0]["relation"] = "right_of" if axis == "left_of" else "below"
        cf_clause = prefix + f"the {a} is {inv} the {b}"
    row = {"case_id": f"{split}_{category}_{index:03d}", "category": category, "split": split,
           "exclusive_control": category != "existence",
           "variants": variants([f"A photograph {p}." for p in clauses], req)}
    row["variants"].append({"variant_id": "cf0", "kind": "counterfactual",
                            "prompt": f"A photograph {cf_clause}.", "requirements": cf})
    validate_scene(row)
    return row


def build_v2(split="main"):
    if split == "smoke":
        dev = build_v2("dev")
        return [next(s for s in dev if s["category"] == cat) for cat in ("existence", "color_binding", "count", "spatial")]
    if split == "dev":
        return [make_scene(cat, i, DEV_OBJECTS[i][0], DEV_OBJECTS[(i+1)%4][0],
                           count=2+i%3, axis="left_of" if i%2 == 0 else "above", split="dev")
                for cat in ("existence", "color_binding", "count", "spatial") for i in range(4)]
    out = []
    pairs = pair_indices((1, 2) if split == "main" else (3,))
    if split != "main":
        pairs = pairs[:4]
    for category in ("existence", "color_binding", "spatial"):
        for idx, (i, j) in enumerate(pairs):
            out.append(make_scene(category, idx, OBJECTS[i][0], OBJECTS[j][0],
                                  axis="left_of" if idx % 2 == 0 else "above", split=split))
    for i, (name, _) in enumerate(OBJECTS):
        counts = (2 + i % 3, 2 + (i + 1) % 3) if split == "main" else (2 + (i + 2) % 3,)
        for count in counts:
            out.append(make_scene("count", len([s for s in out if s["category"] == "count"]),
                                  name, "", count=count, split=split))
        if split != "main" and i == 3:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data")
    args = parser.parse_args()
    import json
    for split in ("main", "dev", "smoke"):
        rows = build_v2(split)
        target = Path(args.directory) / f"scenes_v2_{split}.jsonl"
        if target.exists():
            existing = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
            if existing != rows:
                raise ValueError(f"Refusing to overwrite a different suite: {target}")
        with atomic_path(target) as tmp:
            tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        save_json(target.with_suffix(".manifest.json"), {"schema_version": 2, "split": split,
                  "scenes": len(rows), "categories": dict(Counter(r["category"] for r in rows)),
                  "sha256": stable_hash(rows), "human_prompt_review": "NOT PERFORMED"})
        print(f"{target}: {len(rows)} scenes")


if __name__ == "__main__":
    main()
