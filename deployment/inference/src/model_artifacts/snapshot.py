"""Portable MLflow provenance and checksums for offline inference artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import SelectedModel

MANIFEST = "model-source.json"


def inventory(directory: Path) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Model snapshots cannot contain symbolic links: {path}")
        if path.is_file() and path != directory / MANIFEST:
            with path.open("rb") as source:
                files[path.relative_to(directory).as_posix()] = hashlib.file_digest(
                    source, "sha256"
                ).hexdigest()
    if not files:
        raise ValueError("Model snapshot must contain artifacts")
    return files


def write_snapshot(directory: Path, model: SelectedModel) -> dict[str, Any]:
    """Record the selected version without querying or modifying the registry."""
    manifest = {
        "format": 1,
        "model": {
            "name": model.name,
            "version": model.version,
            "uri": model.artifact_uri,
            "tracking_uri": model.tracking_uri,
            "registry_uri": model.registry_uri,
        },
        "files": inventory(directory),
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_snapshot(directory: Path) -> dict[str, Any]:
    """Verify a fixed resolved directory before exposing it to an inference worker."""
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest.get("format") != 1:
        raise ValueError("Unsupported model snapshot format")
    model = manifest["model"]
    if not all(
        isinstance(model.get(field), str) and model[field].strip()
        for field in ("name", "version", "uri", "tracking_uri", "registry_uri")
    ):
        raise ValueError("Model snapshot requires complete MLflow provenance")
    files = manifest["files"]
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or name == MANIFEST:
            raise ValueError("Invalid model snapshot artifact path")
    if inventory(directory) != files:
        raise ValueError("Model snapshot artifacts do not match recorded checksums")
    return manifest
