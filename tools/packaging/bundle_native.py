"""Package one CMake-installed native asset set with identity, hashes and notices."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    source = Path(__file__).resolve().parents[2] / "deployment/inference"
    pins = dict(
        re.findall(
            r"set\((DAWN_COMMIT|ORT_COMMIT) (\w+)\)", (source / "native/pins.cmake").read_text()
        )
    )
    # Accept only the exact source identities used by the build recipe.
    for name, pin in (("dawn", "DAWN_COMMIT"), ("ort", "ORT_COMMIT")):
        actual = subprocess.check_output(
            ["git", "-C", str(build / "sources" / name), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != pins[pin]:
            raise ValueError(f"{name} source identity mismatch: {actual}")
    suffix = {"linux": ".so", "darwin": ".dylib", "win32": ".dll"}[sys.platform]
    prefix = "" if sys.platform == "win32" else "lib"
    names = [
        "webgpu.py",
        f"{prefix}webgpu_dawn{suffix}",
        f"{prefix}onnxruntime_providers_webgpu{suffix}",
    ]
    identity = f"dawn-ort-{sys.platform}-{platform.machine().lower()}"
    staging = output / identity
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in names:
        path = build / "assets" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing native asset: {path}")
        shutil.copyfile(path, staging / name)
    # Preserve upstream license paths, including third-party notices in fetched
    # sources and binary archives. Build output itself is never included.
    for directory in (build / "sources", build / "_deps", build / "deps"):
        if not directory.exists():
            continue
        for pattern in ("*LICENSE*", "*NOTICE*", "*ThirdPartyNotices*"):
            for path in directory.rglob(pattern):
                if path.is_file() and ".git" not in path.parts:
                    target = staging / "licenses" / directory.name / path.relative_to(directory)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
    shutil.copyfile(source / "LICENSE.txt", staging / "LICENSE.txt")
    for name in ("dawn", "ort"):
        if not (staging / "licenses/sources" / name / "LICENSE").is_file():
            raise ValueError(f"Missing upstream license: {name}")
    manifest = {
        "platform": sys.platform,
        "architecture": platform.machine().lower(),
        "python": "3.12",
        "onnxruntime": "1.30.0",
        **pins,
        "files": {
            str(path.relative_to(staging)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = shutil.make_archive(str(output / identity), "zip", staging)
    print(json.dumps({"archive": archive, "files": len(manifest["files"]), "pins": pins}))


if __name__ == "__main__":
    main()
