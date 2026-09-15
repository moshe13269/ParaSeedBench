from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_RELATIONS = {"left_of", "right_of", "above", "below"}
VALID_KINDS = {"equivalent", "counterfactual"}


@dataclass(frozen=True)
class Variant:
    variant_id: str
    prompt: str
    kind: str
    requirements: dict[str, Any]


def atom_keys(requirements: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for item in requirements.get("objects", []):
        if "exact_count" in item:
            keys.append(f"count:{item['label']}={item['exact_count']}")
        else:
            keys.append(f"exists:{item['label']}")
    for item in requirements.get("colors", []):
        keys.append(f"color:{item['object']}={item['color']}")
    for item in requirements.get("relations", []):
        keys.append(
            f"relation:{item['subject']}:{item['relation']}:{item['object']}"
        )
    return keys


def validate_scene(scene: dict[str, Any]) -> None:
    required = {"case_id", "category", "variants"}
    missing = required - scene.keys()
    if missing:
        raise ValueError(f"{scene.get('case_id', '<unknown>')}: missing {sorted(missing)}")
    variants = scene["variants"]
    equivalent = [v for v in variants if v["kind"] == "equivalent"]
    if len(equivalent) < 2:
        raise ValueError(f"{scene['case_id']}: need at least two equivalent prompts")
    base_atoms = atom_keys(equivalent[0]["requirements"])
    for variant in variants:
        if variant["kind"] not in VALID_KINDS:
            raise ValueError(f"{scene['case_id']}: invalid kind {variant['kind']}")
        if not variant["prompt"].strip():
            raise ValueError(f"{scene['case_id']}: empty prompt")
        for relation in variant["requirements"].get("relations", []):
            if relation["relation"] not in VALID_RELATIONS:
                raise ValueError(f"{scene['case_id']}: invalid relation")
        if variant["kind"] == "equivalent":
            if atom_keys(variant["requirements"]) != base_atoms:
                raise ValueError(f"{scene['case_id']}: equivalent requirements differ")

