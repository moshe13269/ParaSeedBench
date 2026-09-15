"""Crash-aware local checkpoints. Atomic replacement is not a cloud-sync guarantee."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def atomic_path(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", suffix=target.suffix, dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(target, value):
    with atomic_path(target) as temporary:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_nonfinite_floats(value):
    """Make provenance JSON-standard without changing the runtime object.

    Some third-party scheduler configurations legitimately contain +/-inf.
    JSON has no portable representation for those values, so provenance stores
    an explicit tagged value while inference continues to use the untouched
    scheduler configuration.
    """
    if isinstance(value, float) and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return {"__paraseedbench_nonfinite_float__": label}
    if isinstance(value, dict):
        return {key: encode_nonfinite_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_nonfinite_floats(item) for item in value]
    return value


def guard_manifest(path, expected):
    path = Path(path)
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(f"Run/config/environment mismatch: {path}. Use a NEW output directory; do not mix protocols.")
    else:
        save_json(path, expected)


def committed_image(image_path, metadata_path, expected):
    """Metadata is the commit marker, written last; validate bytes, not just existence."""
    from PIL import Image
    try:
        metadata = read_json(metadata_path)
    except (OSError, ValueError):
        return False
    if metadata.get("job") != expected:
        raise ValueError(f"Different generation job at {metadata_path}; choose a NEW output directory")
    try:
        if digest(image_path) != metadata.get("image_sha256"):
            return False
        with Image.open(image_path) as im:
            im.verify()
        return True
    except (OSError, ValueError):
        return False


@contextmanager
def run_lock(root):
    """Single writer per run. No automatic stale-lock stealing after a crash."""
    lock = Path(root) / ".run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Writer lock exists: {lock}. Confirm the old process is stopped, then use run_v2 --clear-stale-lock.") from exc
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)
