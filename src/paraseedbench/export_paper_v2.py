"""Copy validated main-run and resolved human-audit outputs into the manuscript."""
import argparse
import json
from pathlib import Path
from .io import stable_hash
from .storage_v2 import read_json, atomic_path, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Run root after successful analysis")
    p.add_argument("--audit", required=True, help="Resolved main human-audit directory")
    p.add_argument("--manuscript", default="manuscript")
    args = p.parse_args()
    root, audit, paper = Path(args.input), Path(args.audit), Path(args.manuscript)
    info = read_json(root / "analysis" / "analysis_info.json")
    manifest = read_json(root / "run_manifest.json")
    if info.get("dataset_split") != "main" or "main" not in Path(manifest["config"]["dataset"]).stem:
        raise ValueError("Do not insert smoke/dev scores as main paper results")
    if info["protocol"] != "paraseedbench-2.0":
        raise ValueError("Not v2 results")
    audit_info = read_json(audit / "audit_info.json")
    if audit_info.get("dataset_split") != "main":
        raise ValueError("Paper audit must come from the main split")
    if not audit_info.get("paper_primary_ready"):
        raise ValueError(
            "Human-primary gate is not ready: require two independent raters, resolved "
            "atomic disagreements, and at least four audited scenes per category"
        )
    if audit_info.get("run_manifest_sha256") != stable_hash(manifest):
        raise ValueError("Human audit does not belong to this exact main run")
    protected = {
        "private_key_sha256": audit / "PRIVATE_machine_key.csv",
        "machine_human_agreement_sha256": audit / "machine_human_agreement.csv",
        "audit_metric_summary_sha256": audit / "audit_metric_summary.csv",
        "metric_differences_sha256": audit / "machine_human_metric_differences.csv",
        "evaluator_table_sha256": audit / "evaluator_agreement_table.tex",
        "human_table_sha256": audit / "human_primary_results_table.tex",
    }
    recorded = audit_info.get("artifacts", {})
    for label, path in protected.items():
        if not path.exists() or recorded.get(label) != digest(path):
            raise ValueError(f"Human-audit artifact missing or changed: {path}")
    copies = {
        root / "analysis" / "main_results_table.tex":
            paper / "generated" / "automatic_secondary_results_v2.tex",
        root / "analysis" / "category_robust_accuracy.pdf":
            paper / "generated" / "automatic_secondary_category_v2.pdf",
        audit / "human_primary_results_table.tex":
            paper / "generated" / "human_primary_results_v2.tex",
        audit / "evaluator_agreement_table.tex":
            paper / "generated" / "evaluator_agreement_v2.tex",
    }
    for source, destination in copies.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(destination) as tmp:
            tmp.write_bytes(source.read_bytes())
    with atomic_path(paper / "generated" / "paper_results_info.json") as tmp:
        tmp.write_text(
            json.dumps({
                "run_analysis": info,
                "human_audit": audit_info,
                "automatic_semantic_role": "secondary_diagnostic",
                "human_semantic_role": "primary",
            }, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(
        "Human-primary and automatic-secondary artifacts copied. Recompile main_v2.tex, "
        "replace pending-results prose with measured findings, and recheck <=4 technical pages."
    )


if __name__ == "__main__":
    main()
