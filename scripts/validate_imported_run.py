"""Validate an original workstation run manifest after copying it into the release."""
from __future__ import annotations

import json
from pathlib import Path

from paraseedbench.io import stable_hash


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "paper_artifacts" / "provenance"
MANIFEST = PROVENANCE / "original_run_manifest.json"
AUDIT_INFO = PROVENANCE / "audit_info_original_v024.json"


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Copy outputs/v2_main_v024_rtx6000ada/run_manifest.json to {MANIFEST}"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_INFO.read_text(encoding="utf-8"))
    if stable_hash(manifest) != audit["run_manifest_sha256"]:
        raise ValueError("Imported manifest does not match the completed human audit")
    if manifest.get("protocol") != "paraseedbench-2.0":
        raise ValueError("Imported manifest is not protocol v2")
    if manifest.get("dataset_sha256") != "027941aebeabf0d544fd0c400abd0e93aa4b966cdcb09ba2fd3e9e475d098bfc":
        raise ValueError("Imported manifest does not reference the published main prompt suite")
    print("PASS: original run manifest matches the audit and published dataset.")
    print(json.dumps(manifest.get("environment", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
