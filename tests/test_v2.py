import json
import sys
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from paraseedbench.storage_v2 import (atomic_path, save_json, read_json, digest,
    encode_nonfinite_floats, guard_manifest, committed_image, run_lock)
from paraseedbench.suite_v2 import build_v2
from paraseedbench.metrics_v2 import decomposition, crossfit_worst, state_jsd, stratified_ci
from paraseedbench.analyze_v2 import validate_scores, case_metrics, analyze
from paraseedbench.protocol_v2 import jobs
from paraseedbench.schema import atom_keys
from paraseedbench.embed_v2 import load_chunk
from paraseedbench.audit_v2 import (
    select_cases, checked_merge, _atomic_agreement,
    _heldout_selected_worst_summary, score,
)
from paraseedbench.score_audit import agreement
from paraseedbench.config_v2 import profile


def test_atomic_interruption_preserves_previous_commit(tmp_path):
    path = tmp_path / "score.json"
    save_json(path, {"step": 1})
    with pytest.raises(RuntimeError):
        with atomic_path(path) as temporary:
            temporary.write_text("unfinished")
            raise RuntimeError("simulated interruption")
    assert read_json(path) == {"step": 1}
    assert not list(tmp_path.glob(".pending-*"))


def test_atomic_success_and_manifest_rejection(tmp_path):
    path = tmp_path / "manifest.json"
    guard_manifest(path, {"version": 1})
    guard_manifest(path, {"version": 1})
    with pytest.raises(ValueError):
        guard_manifest(path, {"version": 2})
    assert read_json(path)["version"] == 1


def test_scheduler_nonfinite_provenance_is_explicit_and_strict(tmp_path):
    raw = {"lambda_min_clipped": -float("inf"), "nested": [float("inf"), float("nan")]}
    with pytest.raises(ValueError):
        save_json(tmp_path / "raw.json", raw)
    encoded = encode_nonfinite_floats(raw)
    save_json(tmp_path / "encoded.json", encoded)
    assert read_json(tmp_path / "encoded.json") == {
        "lambda_min_clipped": {"__paraseedbench_nonfinite_float__": "-Infinity"},
        "nested": [
            {"__paraseedbench_nonfinite_float__": "Infinity"},
            {"__paraseedbench_nonfinite_float__": "NaN"},
        ],
    }


def test_image_commit_recovery(tmp_path):
    path = tmp_path / "image.png"
    meta = path.with_suffix(".json")
    job = {"seed": 1}
    assert not committed_image(path, meta, job)
    Image.new("RGB", (8, 8), "blue").save(path)
    assert not committed_image(path, meta, job)
    save_json(meta, {"job": job, "image_sha256": digest(path)})
    assert committed_image(path, meta, job)
    path.write_bytes(b"truncated PNG")
    assert not committed_image(path, meta, job)
    with pytest.raises(ValueError):
        committed_image(path, meta, {"seed": 2})


def test_single_writer_lock(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path):
                pass
    assert not (tmp_path / ".run.lock").exists()


def test_pixart_profile_declares_complete_pipeline_repository():
    cfg = profile("rtx6000ada", "smoke")
    model = next(m for m in cfg["models"]
                 if m["name"] == "pixart_sigma")
    assert model["id"].endswith("PixArt-Sigma-XL-2-512-MS")
    assert model["pipeline_id"].endswith("PixArt-Sigma-XL-2-1024-MS")
    assert model["vae_slicing"] is False
    assert model["clean_caption"] is False
    assert cfg["evaluation"]["detector"]["use_fast"] is False
    assert cfg["evaluation"]["embedding_model"]["use_fast"] is False


def test_pixart_transformer_only_repository_is_composed(monkeypatch):
    from paraseedbench import model_registry

    calls = []
    transformer_object = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16"))

    class GenericPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("generic loader must not be used for a transformer-only repository")

    class Transformer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("transformer", model_id, kwargs))
            return transformer_object

    class LoadedPipeline:
        def __init__(self):
            self.moved_to = None
            self.vae_slicing_enabled = False

        def to(self, device):
            self.moved_to = device
            return self

        def enable_vae_slicing(self):
            self.vae_slicing_enabled = True

        def set_progress_bar_config(self, **kwargs):
            pass

    loaded = LoadedPipeline()

    class PixArtPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("pipeline", model_id, kwargs))
            return loaded

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        DiffusionPipeline=GenericPipeline,
        PixArtSigmaPipeline=PixArtPipeline,
        PixArtTransformer2DModel=Transformer,
    ))
    cfg = dict(
        id="PixArt-alpha/PixArt-Sigma-XL-2-512-MS", revision="a" * 40,
        pipeline_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        pipeline_revision="b" * 40, offload="none", use_safetensors=True,
        attention_slicing=False, vae_slicing=False, vae_tiling=False,
    )
    result = model_registry.load_pipeline(cfg)
    assert result is loaded and loaded.moved_to == "cuda"
    assert loaded.vae_slicing_enabled is False
    assert calls[0][0] == "transformer"
    assert calls[0][2]["subfolder"] == "transformer"
    assert calls[0][2]["revision"] == "a" * 40
    assert calls[1][0] == "pipeline"
    assert calls[1][2]["transformer"] is transformer_object
    assert calls[1][2]["revision"] == "b" * 40


def test_pixart_caption_cleaning_is_explicitly_disabled(monkeypatch):
    from paraseedbench import determinism, generate_v2

    marker = object()
    monkeypatch.setattr(determinism, "fresh_generator", lambda seed: (marker, seed))
    model = dict(steps=20, guidance_scale=4.5, height=512, width=512,
                 clean_caption=False)
    requested = generate_v2.call_kwargs(model, {"prompt": "test", "seed": 17},
                                         {"negative_prompt": ""})
    assert requested["clean_caption"] is False
    assert requested["generator"] == (marker, 17)


def test_v2_suite_balanced_disjoint_and_style():
    main, dev, smoke = build_v2(), build_v2("dev"), build_v2("smoke")
    assert (len(main), len(dev), len(smoke)) == (96, 16, 4)
    assert set(s["case_id"] for s in main).isdisjoint(s["case_id"] for s in dev)
    def spec(s):
        return json.dumps(s["variants"][0]["requirements"], sort_keys=True)
    assert {spec(s) for s in main}.isdisjoint(spec(s) for s in dev)
    main_all = {json.dumps(v["requirements"],sort_keys=True) for s in main for v in s["variants"]}
    dev_all = {json.dumps(v["requirements"],sort_keys=True) for s in dev for v in s["variants"]}
    assert main_all.isdisjoint(dev_all)
    for category in {s["category"] for s in main}:
        assert sum(s["category"] == category for s in main) == 24
    for scene in main+dev:
        assert len({v["prompt"] for v in scene["variants"]}) == 5
        assert all(v["prompt"].startswith("A photograph ") for v in scene["variants"])
        if scene["category"] in ("spatial", "color_binding"):
            assert all(o.get("exact_count") == 1 for o in scene["variants"][0]["requirements"]["objects"])


@pytest.mark.parametrize("kind", ["wording", "seed", "interaction", "wrong", "correct"])
def test_analytical_components(kind):
    grids = {"wording": [[0,0],[1,1]], "seed": [[0,1],[0,1]], "interaction": [[0,1],[1,0]],
             "wrong": [[0,0],[0,0]], "correct": [[1,1],[1,1]]}
    result = decomposition(np.asarray(grids[kind])[...,None])
    expected = {"wording": "variance_wording", "seed": "variance_seed", "interaction": "variance_interaction"}
    assert result["variance_total"] == (0.25 if kind in expected else 0)
    for name in expected.values():
        assert result[name] == (0.25 if expected.get(kind) == name else 0)


def test_decomposition_matches_direct_distances():
    q = np.random.default_rng(42).integers(0, 2, (4,8,5))
    r = decomposition(q)
    dw = np.mean([np.mean(q[j] != q[k]) for j,k in combinations(range(4),2)])
    ds = np.mean([np.mean(q[:,s] != q[:,t]) for s,t in combinations(range(8),2)])
    assert r["wording_disagreement"] == pytest.approx(dw)
    assert r["seed_disagreement"] == pytest.approx(ds)
    assert r["variance_total"] == pytest.approx(sum(r[k] for k in ("variance_wording","variance_seed","variance_interaction")))


def test_crossfit_not_same_sample_minimum():
    assert crossfit_worst([[0,1],[1,0]]) == 1


def test_crossfit_uses_all_balanced_splits_and_averages_ties():
    # The first wording succeeds on the final four seeds and the second on the
    # first four. Across all C(8,4)=70 directed halves, the held-out selected
    # score is 44/70 = 22/35. An even/odd split alone would incorrectly give .5.
    grid = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ])
    assert crossfit_worst(grid) == pytest.approx(22 / 35)


def test_crossfit_is_invariant_to_wording_and_seed_order():
    rng = np.random.default_rng(2031)
    grid = rng.integers(0, 2, size=(4, 8))
    expected = crossfit_worst(grid)
    assert crossfit_worst(grid[[2, 0, 3, 1]]) == pytest.approx(expected)
    for _ in range(10):
        assert crossfit_worst(grid[:, rng.permutation(8)]) == pytest.approx(expected)


def test_crossfit_rejects_odd_seed_count():
    with pytest.raises(ValueError, match="even seed count"):
        crossfit_worst(np.ones((4, 3)))


def test_heldout_summary_has_metric_local_reproducibility():
    rows = []
    for model in ("z_model", "a_model"):
        for category in ("a", "b"):
            for value in (0.25, 0.75):
                rows.append({
                    "model": model,
                    "category": category,
                    "crossfit_worst_accuracy": value,
                })
    frame = pd.DataFrame(rows)
    first = _heldout_selected_worst_summary(frame, "human_primary")
    second = _heldout_selected_worst_summary(
        frame.sample(frac=1, random_state=19), "human_primary"
    )
    pd.testing.assert_frame_equal(first, second)


def test_jsd_uses_full_state_support():
    from scipy.spatial.distance import jensenshannon
    q = np.array([[[0,0],[0,0]], [[1,1],[1,1]]])
    assert state_jsd(q) == pytest.approx(jensenshannon([2.5,.5,.5,.5],[.5,.5,.5,2.5],base=2)**2)
    assert state_jsd(np.zeros((4,8,2))) == 0


def test_category_balance_and_missing_diversity():
    e, lo, hi = stratified_ci([0,0,0,1], ["a","a","a","b"], np.random.default_rng(1),100)
    assert (e,lo,hi) == (.5,.5,.5)
    assert np.isnan(stratified_ci([0,np.nan],["a","b"],np.random.default_rng(1),100)[0])


def synthetic_scores():
    scenes = build_v2("smoke")
    cfg = {"protocol":"test", "models":[{"name":"toy"},{"name":"toy2"}], "seeds":[11,29]}
    rows = []
    for model,scene,variant,job in jobs(cfg,scenes):
        keys = atom_keys(variant["requirements"])
        rows.append(dict(model=model["name"],case_id=scene["case_id"],category=scene["category"],variant_id=variant["variant_id"],
            kind=variant["kind"],seed=job["seed"],image_path=job["path"],semantic_state="1"*len(keys),
            all_correct=1,atom_keys=json.dumps(keys),censored=0,black_image=0,exclusive_control=scene["exclusive_control"],
            alternate_correct=0 if scene["exclusive_control"] else np.nan,elapsed_seconds=1.,peak_allocated_bytes=1024))
    return cfg,scenes,pd.DataFrame(rows)


@pytest.mark.parametrize("missing", ["scene","seed","variant","model","one_cell"])
def test_entire_missing_axis_is_rejected(missing):
    cfg,scenes,scores = synthetic_scores()
    if missing == "one_cell":
        scores = scores.iloc[1:]
    else:
        col = {"scene":"case_id","seed":"seed","variant":"variant_id","model":"model"}[missing]
        scores = scores[scores[col] != scores[col].iloc[0]]
    with pytest.raises(ValueError):
        validate_scores(scores,cfg,scenes)


def test_duplicate_and_wrong_state_rejected():
    cfg,scenes,scores = synthetic_scores()
    validate_scores(scores,cfg,scenes)
    with pytest.raises(ValueError):
        validate_scores(pd.concat([scores,scores.iloc[:1]]),cfg,scenes)
    scores.loc[0,"semantic_state"] = "bad"
    with pytest.raises(ValueError):
        validate_scores(scores,cfg,scenes)


def test_embedding_chunk_resume_and_corruption(tmp_path):
    p = tmp_path / "chunk.npz"
    np.savez_compressed(p,token=np.asarray("abc"),paths=np.array(["a"]),embeddings=np.ones((1,3)))
    assert load_chunk(p,"abc",["a"]).shape == (1,3)
    assert load_chunk(p,"different",["a"]) is None
    assert load_chunk(p,"abc",["b"]) is None
    p.write_bytes(b"interrupted")
    assert load_chunk(p,"abc",["a"]) is None


def test_audit_selection_does_not_depend_on_scores():
    _,_,scores = synthetic_scores()
    selection = select_cases(scores,1,2027)
    scores.all_correct = 0
    assert selection == select_cases(scores,1,2027)
    assert len(selection) == 4


def test_audit_requires_complete_valid_atomic_states():
    key = pd.DataFrame([dict(audit_id="a",atom_keys='["one","two"]')])
    human = pd.DataFrame([dict(audit_id="a",human_atom_state="01")])
    assert checked_merge(human,key).human_all_correct.iloc[0] == 0
    human.human_atom_state = ""
    with pytest.raises(ValueError):
        checked_merge(human,key)


def test_agreement_reports_class_balance_and_atomic_states():
    report = agreement(np.array([1, 0, 0, 0]), np.array([1, 1, 0, 0]))
    assert report["recall"] == .5
    assert report["specificity"] == 1
    assert report["balanced_accuracy"] == .75
    atomic = _atomic_agreement(["10", "01"], ["11", "01"])
    assert atomic["n_atoms"] == 4
    assert atomic["atomic_accuracy"] == .75
    assert atomic["atomic_exact_agreement"] == .5
    assert atomic["mean_image_atomic_accuracy"] == .75


def test_two_rater_audit_blocks_until_disagreements_are_adjudicated(tmp_path):
    _, scenes, scores = synthetic_scores()
    sample = scores[scores.kind == "equivalent"].reset_index(drop=True).copy()
    sample["audit_id"] = [f"item_{i:05d}" for i in range(len(sample))]
    root, target = tmp_path / "run", tmp_path / "run" / "audit"
    (root / "analysis").mkdir(parents=True)
    target.mkdir()
    sample.to_csv(target / "PRIVATE_machine_key.csv", index=False)
    save_json(target / "selection.json", {
        "seed": 2027, "scene_ids": [s["case_id"] for s in scenes],
        "scenes_per_category": 1, "images": len(sample),
    })
    save_json(root / "analysis" / "analysis_info.json", {"dataset_split": "dev"})
    save_json(root / "run_manifest.json", {"protocol": "test"})
    first = pd.DataFrame({
        "audit_id": sample.audit_id,
        "human_atom_state": sample.semantic_state,
        "notes": "",
    })
    second = first.copy()
    state = second.loc[0, "human_atom_state"]
    second.loc[0, "human_atom_state"] = ("0" if state[0] == "1" else "1") + state[1:]
    first_path, second_path = target / "human_a.csv", target / "human_b.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)
    score(root, target, first_path, second_path)
    blocked = read_json(target / "audit_info.json")
    assert blocked["rater_count"] == 2
    assert blocked["atomic_disagreements_between_raters"] == 1
    assert not blocked["human_primary_ready"]
    assert len(pd.read_csv(target / "adjudication_blank.csv")) == 1
    adjudicated = first.iloc[[0]].copy()
    adjudicated_path = target / "human_adjudicated.csv"
    adjudicated.to_csv(adjudicated_path, index=False)
    score(root, target, first_path, second_path, adjudicated_path)
    resolved = read_json(target / "audit_info.json")
    assert resolved["human_primary_ready"]
    assert not resolved["paper_primary_ready"]
    report = pd.read_csv(target / "machine_human_agreement.csv")
    overall = report[(report.model == "all") & (report.category == "all")].iloc[0]
    assert overall.atomic_exact_agreement == 1


def test_synthetic_analysis_end_to_end(tmp_path):
    cfg,scenes,scores = synthetic_scores()
    cfg["output_dir"] = str(tmp_path)
    np.savez_compressed(tmp_path/"dino_embeddings.npz",paths=scores.image_path.to_numpy(str),embeddings=np.tile([1.,0.,0.],(len(scores),1)))
    result = analyze(cfg,scenes,scores,bootstrap=100)
    assert (result[result.metric == "mean_accuracy"].estimate == 1).all()
    assert (tmp_path/"analysis"/"category_robust_accuracy.pdf").exists()
    info = read_json(tmp_path/"analysis"/"analysis_info.json")
    assert not info["submission_ready"]
    assert info["dataset_split"] == "dev"
    assert info["scene_count"] == 4
    assert info["scored_images"] == 80
    assert info["images_per_model"] == {"toy": 40, "toy2": 40}


def fake_torch(monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
         max_memory_allocated=lambda: 1024,max_memory_reserved=lambda: 1024,empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=cuda,inference_mode=contextlib.nullcontext))


def test_generation_interrupted_then_only_pending_jobs_run(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from paraseedbench import generate_v2 as gen, model_registry, determinism
    fake_torch(monkeypatch)
    cfg = {"models":[dict(name="toy",id="toy",revision="a"*40,width=8,height=8)],
           "seeds":[1,2],"output_dir":str(tmp_path),"full_determinism":True}
    scenes = build_v2("smoke")[:1]
    class Pipe:
        scheduler = SimpleNamespace(config={})
        calls = 0
        stop_at = 2
        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls == self.stop_at:
                raise RuntimeError("simulated runtime termination")
            return SimpleNamespace(images=[Image.new("RGB",(8,8),"red")],nsfw_content_detected=[False])
    pipe = Pipe()
    monkeypatch.setattr(model_registry,"load_pipeline",lambda _:pipe)
    monkeypatch.setattr(determinism,"seed_runtime",lambda *args:None)
    monkeypatch.setattr(gen,"call_kwargs",lambda *args:{})
    monkeypatch.setattr(gen,"repeatability_gate",lambda *args:None)
    with pytest.raises(RuntimeError):
        gen.generate(cfg,scenes,"run")
    assert len(list((tmp_path/"images").rglob("*.json"))) == 1
    pipe.stop_at = -1
    before = pipe.calls
    gen.generate(cfg,scenes,"run")
    assert pipe.calls-before == 9
    before = pipe.calls
    gen.generate(cfg,scenes,"run")
    assert pipe.calls == before
    # Corrupt one already-committed PNG: only that image is repaired.
    next((tmp_path/"images").rglob("*.png")).write_bytes(b"corrupt")
    gen.generate(cfg,scenes,"run")
    assert pipe.calls == before+1


def test_evaluation_interruption_reuses_individual_scores(tmp_path,monkeypatch):
    from paraseedbench import evaluate_v2 as ev2, evaluate as ev1
    fake_torch(monkeypatch)
    cfg = {"models":[dict(name="toy",id="toy")],"seeds":[1,2],"output_dir":str(tmp_path),
           "evaluation":{"detector":{"id":"toy","revision":"x"},"color_model":{"id":"toy","revision":"x"},
                         "threshold":.1,"nms_threshold":.3,"relation_margin":.03}}
    scenes = build_v2("smoke")[:1]
    for _,scene,variant,job in jobs(cfg,scenes):
        path = tmp_path/job["path"]
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.new("RGB",(8,8),"blue").save(path)
        save_json(path.with_suffix(".json"),dict(job=dict(job,run_id="run"),image_sha256=digest(path),
             censored=False,black_image=False,elapsed_seconds=1.,peak_allocated_bytes=1))
    class FakeEvaluator:
        calls=0
        stop_at=2
        def __init__(self,*args,**kwargs):pass
        def evaluate(self,image,req):
            FakeEvaluator.calls += 1
            if FakeEvaluator.calls == FakeEvaluator.stop_at:
                raise RuntimeError("simulated termination")
            return {k:True for k in atom_keys(req)},{}
    monkeypatch.setattr(ev1,"StructuredEvaluator",FakeEvaluator)
    with pytest.raises(RuntimeError):
        ev2.evaluate(cfg,scenes,"run")
    assert len(list((tmp_path/"scores").rglob("*.json"))) == 1
    FakeEvaluator.stop_at = -1
    before = FakeEvaluator.calls
    assert len(ev2.evaluate(cfg,scenes,"run")) == 10
    assert FakeEvaluator.calls-before == 9
    before = FakeEvaluator.calls
    ev2.evaluate(cfg,scenes,"run")
    assert FakeEvaluator.calls == before
