"""The test worker produces only the documented deterministic test mask."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from tests.smoke_worker import write_test_mask


def test_test_mask_preserves_geometry_and_threshold(tmp_path: Path) -> None:
    voxels = np.array([[[-2, 0, 1], [23, -1, 2]], [[0, 5, -3], [4, 0, 6]]], dtype=np.int16)
    image = sitk.GetImageFromArray(voxels)
    image.SetOrigin((14, -7, 2))
    image.SetSpacing((0.7, 1.2, 2.5))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    input_path, output_path = tmp_path / "input.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(input_path))

    metadata_path = tmp_path / "metadata.json"
    write_test_mask(input_path, output_path, metadata_path)

    mask = sitk.ReadImage(str(output_path))
    assert np.array_equal(sitk.GetArrayFromImage(mask), voxels > 0)
    assert mask.GetPixelID() == sitk.sitkUInt8
    assert mask.GetSize() == image.GetSize()
    assert mask.GetOrigin() == image.GetOrigin()
    assert mask.GetSpacing() == image.GetSpacing()
    assert mask.GetDirection() == image.GetDirection()
    assert json.loads(metadata_path.read_text()) == {"worker": "smoke", "test_only": True}


def test_ordinary_file_cli_runs_without_service_imports(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "input.nii.gz", tmp_path / "mask.nii.gz"
    sitk.WriteImage(sitk.Image([2, 3, 4], sitk.sitkInt16) + 10, str(input_path))
    worker_path = Path(__file__).with_name("smoke_worker.py")
    result = subprocess.run(
        [
            sys.executable,
            str(worker_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--metadata",
            str(tmp_path / "metadata.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert np.all(sitk.GetArrayFromImage(sitk.ReadImage(str(output_path))) == 1)


def test_two_dimensional_input_is_rejected(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "input.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(sitk.Image([2, 3], sitk.sitkInt16), str(input_path))
    with pytest.raises(ValueError, match="scalar 3D"):
        write_test_mask(input_path, output_path, tmp_path / "metadata.json")
    assert not output_path.exists()
