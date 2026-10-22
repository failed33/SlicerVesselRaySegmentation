"""Download and verify the current stable or nightly Linux Slicer application."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path


def download(channel: str, output: Path) -> dict[str, str]:
    stability = {"stable": "release", "nightly": "nightly"}[channel]
    base = "https://download.slicer.org"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "slicer.tar.gz"
    attempts = 3
    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(
            f"{base}/find?os=linux&stability={stability}", timeout=60
        ) as response:
            metadata = json.load(response)
        (output / "download.json").write_text(json.dumps(metadata, indent=2) + "\n")
        digest = hashlib.sha512()
        with urllib.request.urlopen(base + metadata["download_url"], timeout=120) as response:
            with archive.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
        expected_size = int(metadata["size"])
        actual_size = archive.stat().st_size
        expected_sha = str(metadata["sha512"]).lower()
        actual_sha = digest.hexdigest()
        if actual_size == expected_size and actual_sha == expected_sha:
            break
        archive.unlink(missing_ok=True)
        if attempt == attempts:
            raise ValueError(
                "Slicer download size or SHA-512 verification failed "
                f"(expected size={expected_size}, sha512={expected_sha}; "
                f"actual size={actual_size}, sha512={actual_sha})"
            )
    with tarfile.open(archive) as package:
        package.extractall(output / "application", filter="data")
    executables = list((output / "application").glob("*/Slicer"))
    if len(executables) != 1:
        raise ValueError("Expected exactly one Slicer launcher in the application archive")
    return {
        "slicer_executable": str(executables[0].resolve()),
        "slicer_version": ".".join(metadata["version"].split(".")[:2]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channel", choices=("stable", "nightly"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    values = download(arguments.channel, arguments.output)
    print(json.dumps(values))
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as destination:
            for key, value in values.items():
                destination.write(f"{key}={value}\n")
