from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .io import write_json
from .schema import atom_keys


COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "gray"]


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= threshold]
    return keep


class StructuredEvaluator:
    def __init__(
        self,
        detector_id: str,
        color_model_id: str,
        device: str,
        threshold: float,
        nms_threshold: float,
        relation_margin: float,
        detector_revision: str | None = None,
        color_revision: str | None = None,
        fp32: bool = False,
        use_fast: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.relation_margin = relation_margin
        dtype = torch.float16 if device.startswith("cuda") and not fp32 else torch.float32
        self.dtype = dtype
        self.detector_processor = AutoProcessor.from_pretrained(
            detector_id, revision=detector_revision, use_fast=use_fast
        )
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_id, dtype=dtype, revision=detector_revision
        ).to(device).eval()
        self.color_processor = CLIPProcessor.from_pretrained(color_model_id, revision=color_revision)
        self.color_model = CLIPModel.from_pretrained(
            color_model_id, dtype=dtype, revision=color_revision
        ).to(device).eval()

    def detect(self, image: Image.Image, labels: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not labels:
            return {}
        queries = [f"a photo of a {label}" for label in labels]
        inputs = self.detector_processor(text=[queries], images=image, return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.detector(**inputs)
        target_sizes = self.torch.tensor([[image.height, image.width]], device=self.device)
        try:
            result = self.detector_processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.threshold,
                text_labels=[queries],
            )[0]
        except (TypeError, AttributeError):
            try:
                result = self.detector_processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=self.threshold
                )[0]
            except (AttributeError, TypeError):
                result = self.detector_processor.post_process_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=self.threshold
                )[0]
        boxes = result["boxes"].detach().float().cpu().numpy()
        scores = result["scores"].detach().float().cpu().numpy()
        label_ids = result.get("labels")
        if label_ids is not None and hasattr(label_ids, "detach"):
            label_ids = label_ids.detach().cpu().numpy().astype(int)
        elif result.get("text_labels") is not None or label_ids is not None:
            query_to_index = {query: index for index, query in enumerate(queries)}
            try:
                label_ids = np.asarray([query_to_index[label] for label in result.get("text_labels", label_ids)], dtype=int)
            except KeyError as exc:
                raise RuntimeError(f"Unexpected detector text label: {exc}") from exc
        else:
            raise RuntimeError("Detector post-processing returned neither labels nor text_labels")
        found: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
        for label_index, label in enumerate(labels):
            indices = np.flatnonzero(label_ids == label_index)
            kept_local = _nms(boxes[indices], scores[indices], self.nms_threshold)
            for local_index in kept_local:
                global_index = int(indices[local_index])
                found[label].append({
                    "box": boxes[global_index].round(2).tolist(),
                    "score": float(scores[global_index]),
                })
        return found

    def classify_color(self, image: Image.Image, box: list[float], object_name: str) -> dict[str, Any]:
        x1, y1, x2, y2 = box
        pad_x = 0.03 * max(1.0, x2 - x1)
        pad_y = 0.03 * max(1.0, y2 - y1)
        crop = image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(image.width, x2 + pad_x), min(image.height, y2 + pad_y)))
        texts = [f"a photo of a {color} {object_name}" for color in COLORS]
        inputs = self.color_processor(text=texts, images=crop, return_tensors="pt", padding=True)
        inputs = {key: value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.color_model(**inputs).logits_per_image[0].float().cpu().numpy()
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = probabilities.argsort()[::-1]
        return {
            "label": COLORS[int(order[0])],
            "confidence": float(probabilities[order[0]]),
            "margin": float(probabilities[order[0]] - probabilities[order[1]]),
        }

    def evaluate(self, image: Image.Image, requirements: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
        labels = sorted({item["label"] for item in requirements.get("objects", [])})
        detections = self.detect(image, labels)
        atoms: dict[str, bool] = {}
        predictions: dict[str, Any] = {"detections": detections, "colors": {}, "relations": {}}

        for item in requirements.get("objects", []):
            count = len(detections.get(item["label"], []))
            if "exact_count" in item:
                key = f"count:{item['label']}={item['exact_count']}"
                atoms[key] = count == int(item["exact_count"])
            else:
                key = f"exists:{item['label']}"
                atoms[key] = count >= int(item.get("min_count", 1))

        for item in requirements.get("colors", []):
            key = f"color:{item['object']}={item['color']}"
            candidates = detections.get(item["object"], [])
            if not candidates:
                atoms[key] = False
                predictions["colors"][item["object"]] = {"label": None, "confidence": 0.0, "margin": 0.0}
            else:
                prediction = self.classify_color(image, candidates[0]["box"], item["object"])
                predictions["colors"][item["object"]] = prediction
                atoms[key] = prediction["label"] == item["color"]

        for item in requirements.get("relations", []):
            key = f"relation:{item['subject']}:{item['relation']}:{item['object']}"
            subject = detections.get(item["subject"], [])
            obj = detections.get(item["object"], [])
            if not subject or not obj:
                atoms[key] = False
                predictions["relations"][key] = None
                continue
            sb = subject[0]["box"]
            ob = obj[0]["box"]
            sx, sy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
            ox, oy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
            dx, dy = (sx - ox) / image.width, (sy - oy) / image.height
            relation = item["relation"]
            passed = {
                "left_of": dx < -self.relation_margin,
                "right_of": dx > self.relation_margin,
                "above": dy < -self.relation_margin,
                "below": dy > self.relation_margin,
            }[relation]
            atoms[key] = bool(passed)
            predictions["relations"][key] = {"dx": dx, "dy": dy}
        return atoms, predictions


def metadata_files(root: Path) -> list[Path]:
    return sorted(path for path in (root / "images").rglob("seed_*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured object/color/count/spatial evaluation")
    parser.add_argument("--input", required=True, help="Generation output directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--color-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--nms-threshold", type=float, default=0.35)
    parser.add_argument("--relation-margin", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.input)
    output = Path(args.output) if args.output else root / "semantic_scores.csv"
    existing: dict[str, dict[str, Any]] = {}
    if output.exists() and not args.overwrite:
        for row in pd.read_csv(output, dtype={"semantic_state": str}).to_dict("records"):
            existing[str(row["image_path"])] = row

    evaluator = StructuredEvaluator(
        args.detector, args.color_model, args.device, args.threshold,
        args.nms_threshold, args.relation_margin,
    )
    rows = list(existing.values())
    files = metadata_files(root)
    pending_since_checkpoint = 0
    for meta_path in tqdm(files, desc="semantic evaluation", unit="image"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        image_path = str(meta["image_path"])
        if image_path in existing:
            continue
        image = Image.open(image_path).convert("RGB")
        atoms, predictions = evaluator.evaluate(image, meta["requirements"])
        ordered_keys = atom_keys(meta["requirements"])
        state = "".join("1" if atoms[key] else "0" for key in ordered_keys)
        rows.append({
            "model": meta["model_name"],
            "model_id": meta["model_id"],
            "case_id": meta["case_id"],
            "category": meta["category"],
            "variant_id": meta["variant_id"],
            "kind": meta["kind"],
            "seed": int(meta["seed"]),
            "image_path": image_path,
            "all_correct": int(all(atoms.values())),
            "atom_accuracy": float(np.mean(list(atoms.values()))),
            "atom_keys": json.dumps(ordered_keys),
            "semantic_state": state,
            "atom_results": json.dumps(atoms, sort_keys=True),
            "predictions": json.dumps(predictions, sort_keys=True),
        })
        pending_since_checkpoint += 1
        if pending_since_checkpoint >= 25:
            pd.DataFrame(rows).sort_values(["model", "case_id", "variant_id", "seed"]).to_csv(output, index=False)
            pending_since_checkpoint = 0

    pd.DataFrame(rows).sort_values(["model", "case_id", "variant_id", "seed"]).to_csv(output, index=False)

    settings = {
        "detector": args.detector,
        "color_model": args.color_model,
        "threshold": args.threshold,
        "nms_threshold": args.nms_threshold,
        "relation_margin": args.relation_margin,
        "number_of_images": len(rows),
    }
    write_json(output.with_suffix(".settings.json"), settings)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
