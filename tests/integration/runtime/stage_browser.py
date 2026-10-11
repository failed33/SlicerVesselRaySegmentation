"""Extract and verify the compiled runtime delivered in an extension wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def stage(wheel: Path, destination: Path) -> dict[str, object]:
    with ZipFile(wheel) as archive:
        manifests = [
            name for name in archive.namelist() if name.endswith("/wasm-runtime/manifest.json")
        ]
        if len(manifests) != 1:
            raise ValueError("Expected one packaged WASM runtime manifest")
        prefix = manifests[0].removesuffix("manifest.json")
        manifest_bytes = archive.read(manifests[0])
        manifest = json.loads(manifest_bytes)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").write_bytes(manifest_bytes)
        for relative, digest in manifest["files"].items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid runtime asset path: {relative}")
            data = archive.read(prefix + relative)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Packaged runtime hash differs: {relative}")
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
    return {"wheel_prefix": prefix, "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.wheel, args.destination)))
