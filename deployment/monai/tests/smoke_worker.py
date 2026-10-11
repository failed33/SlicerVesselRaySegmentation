"""Deterministic test worker. Its output is not clinical model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import SimpleITK as sitk


def write_test_mask(input_path: Path, output_path: Path, metadata_path: Path) -> None:
    """Write a binary positive-intensity mask with the input's geometry."""
    image = sitk.ReadImage(str(input_path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Test worker requires a scalar 3D image")
    mask = sitk.Cast(image > 0, sitk.sitkUInt8)
    sitk.WriteImage(mask, str(output_path))
    metadata_path.write_text(json.dumps({"worker": "smoke", "test_only": True}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    arguments = parser.parse_args()
    write_test_mask(arguments.input, arguments.output, arguments.metadata)


if __name__ == "__main__":
    main()
