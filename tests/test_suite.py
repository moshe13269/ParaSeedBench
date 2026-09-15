from paraseedbench.build_suite import build
from paraseedbench.schema import atom_keys, validate_scene


def test_suite_shape_and_categories():
    scenes = build()
    assert len(scenes) == 96
    counts = {category: sum(s["category"] == category for s in scenes) for category in {s["category"] for s in scenes}}
    assert counts == {"existence": 24, "color_binding": 24, "count": 24, "spatial": 24}


def test_equivalent_prompts_share_requirements():
    for scene in build():
        validate_scene(scene)
        equivalent = [v for v in scene["variants"] if v["kind"] == "equivalent"]
        assert len(equivalent) == 4
        assert len({tuple(atom_keys(v["requirements"])) for v in equivalent}) == 1
        assert len({v["prompt"] for v in equivalent}) == 4


def test_counterfactuals_change_requirements_where_present():
    for scene in build():
        base = atom_keys(scene["variants"][0]["requirements"])
        for variant in scene["variants"]:
            if variant["kind"] == "counterfactual":
                assert atom_keys(variant["requirements"]) != base

