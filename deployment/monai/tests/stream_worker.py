"""Produce a preview before a delayed final file, without a model dependency."""

import argparse
import json
import os
import time
from pathlib import Path

import SimpleITK as sitk

from monai_service.worker_io import PreviewPublisher


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("input", "output", "metadata", "preview-directory"):
        parser.add_argument("--" + name)
    arguments = parser.parse_args()
    mode = os.environ.get("STREAM_TEST_MODE", "success")
    if started := os.environ.get("STREAM_TEST_STARTED"):
        Path(started).write_text(str(os.getpid()))
    image = sitk.ReadImage(arguments.input)
    preview = sitk.Cast(image > 10, sitk.sitkUInt8)
    publisher = PreviewPublisher(Path(arguments.preview_directory), interval_seconds=0)
    publisher.publish(
        {"completed": 1, "total": 2, "elapsed_seconds": 0.1},
        lambda path: sitk.WriteImage(preview, str(path)),
    )
    if mode == "invalid-preview":
        record = Path(arguments.preview_directory) / "ready" / "progress.json"
        record.write_text("invalid-json")
    time.sleep(100 if mode == "cancel" else 1)
    if mode == "error":
        raise SystemExit(7)
    sitk.WriteImage(sitk.Cast(image > 0, sitk.sitkUInt8), arguments.output)
    Path(arguments.metadata).write_text(json.dumps({"worker": "stream-fixture"}))
    if completed := os.environ.get("STREAM_TEST_COMPLETED"):
        Path(completed).touch()


if __name__ == "__main__":
    main()
