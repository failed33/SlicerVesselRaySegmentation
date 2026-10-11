"""Fetch published SlicerWeb wheels and their exact matching application source."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path


def prepare(output: Path) -> dict[str, str]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            "runtime-latest",
            "--repo",
            "lassoan/slicerweb-app",
            "--pattern",
            "wheels.zip",
            "--dir",
            str(output),
            "--clobber",
        ],
        check=True,
    )
    wheels = output / "wheels"
    with zipfile.ZipFile(output / "wheels.zip") as archive:
        archive.extractall(wheels)
    info = json.loads((wheels / "build-info.json").read_text())
    revision = info["slicerweb"]["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or info["slicerweb"]["modified"]:
        raise ValueError("Published runtime must name an unmodified SlicerWeb commit")
    source = output / "SlicerWeb"
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/lassoan/SlicerWeb.git",
            str(source),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", revision], check=True)
    with zipfile.ZipFile(next(wheels.glob("slicer_core-*.whl"))) as archive:
        versions = {
            name.split("/")[2].removeprefix("Slicer-")
            for name in archive.namelist()
            if name.startswith("slicer_home/lib/Slicer-")
        }
    if len(versions) != 1:
        raise ValueError("Expected one Slicer version in the runtime wheel")
    # Do not load upstream bundled extensions or download samples for this demo.
    (wheels / "application.json").write_text(
        json.dumps({"features": {"developerMode": "enabledByDefault"}})
    )
    return {
        "source": str(source),
        "wheels": str(wheels),
        "slicer_version": versions.pop(),
        "revision": revision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    values = prepare(arguments.output)
    print(json.dumps(values))
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as destination:
            for key, value in values.items():
                destination.write(f"{key}={value}\n")
