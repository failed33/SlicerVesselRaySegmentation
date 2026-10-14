"""Captured physical grids and Slicer native image-file serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GEOMETRY_ATOL = 1e-6
ORIGIN_ATOL = 1e-4  # NIfTI stores its physical origin as float32.
IMAGE_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha")
_RAS_TO_LPS = np.diag(np.asarray([-1.0, -1.0, 1.0], dtype=np.float64))


@dataclass(frozen=True)
class ImageGeometry:
    shape_zyx: tuple[int, ...]
    spacing_xyz: tuple[float, ...]
    origin_lps: tuple[float, ...]
    direction_lps: tuple[float, ...]

    @classmethod
    def from_ijk_to_ras(cls, shape: tuple[int, ...], ijk_to_ras: np.ndarray) -> ImageGeometry:
        matrix = np.array(ijk_to_ras, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Invalid CT geometry")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=GEOMETRY_ATOL, rtol=0):
            raise ValueError("Invalid CT geometry")
        spacing = np.linalg.norm(matrix[:3, :3], axis=0)
        if np.any(spacing <= 0):
            raise ValueError("CT spacing must be positive")
        direction = matrix[:3, :3] / spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=GEOMETRY_ATOL, rtol=0):
            raise ValueError("Sheared CT geometry is not supported; resample to an orthogonal grid")
        return cls(
            shape_zyx=tuple(int(value) for value in shape),
            spacing_xyz=tuple(float(value) for value in spacing),
            origin_lps=tuple(float(value) for value in _RAS_TO_LPS @ matrix[:3, 3]),
            direction_lps=tuple(float(value) for value in (_RAS_TO_LPS @ direction).flat),
        )

    def ijk_to_ras(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = _RAS_TO_LPS @ np.asarray(self.direction_lps).reshape(3, 3)
        matrix[:3, :3] *= np.asarray(self.spacing_xyz)
        matrix[:3, 3] = _RAS_TO_LPS @ np.asarray(self.origin_lps)
        return matrix


@dataclass(frozen=True)
class VolumeSnapshot:
    array: np.ndarray = field(repr=False)
    geometry: ImageGeometry
    source_id: str
    source_name: str

    @classmethod
    def capture(
        cls, array: np.ndarray, ijk_to_ras: np.ndarray, source: tuple[str, str]
    ) -> VolumeSnapshot:
        """Copy scalar CT voxels and their untransformed physical grid on the UI thread."""
        if array.ndim != 3 or not all(array.shape) or array.dtype.kind not in "iuf":
            raise ValueError("MONAI inference requires a nonempty scalar 3D CT")
        geometry = ImageGeometry.from_ijk_to_ras(array.shape, ijk_to_ras)
        copied = np.array(array, copy=True, order="C")
        copied.flags.writeable = False
        return cls(copied, geometry, *source)

    def write_image(self, path: Path) -> None:
        import slicer
        import vtk

        node = slicer.vtkMRMLScalarVolumeNode()
        matrix = vtk.vtkMatrix4x4()
        slicer.util.updateVTKMatrixFromArray(matrix, self.geometry.ijk_to_ras())
        node.SetIJKToRASMatrix(matrix)
        slicer.util.updateVolumeFromArray(node, self.array)
        storage = slicer.vtkMRMLVolumeArchetypeStorageNode()
        storage.SetFileName(str(path))
        storage.UseCompressionOn()
        if not storage.WriteData(node):
            raise ValueError("Could not serialize the submitted CT")

    def matches(self, array: np.ndarray, ijk_to_ras: np.ndarray, source: tuple[str, str]) -> bool:
        """Compare the live input without copying an unchanged volume again."""
        return (
            source == (self.source_id, self.source_name)
            and self.geometry == ImageGeometry.from_ijk_to_ras(array.shape, ijk_to_ras)
            and self.array.dtype == array.dtype
            and np.array_equal(self.array, array)
        )

    def read_mask(self, path: Path) -> np.ndarray:
        """Read labels only when their physical grid matches this submitted image."""
        import slicer
        import vtk

        node = slicer.vtkMRMLLabelMapVolumeNode()
        storage = slicer.vtkMRMLVolumeArchetypeStorageNode()
        storage.SetFileName(str(path))
        if not storage.ReadData(node):
            raise ValueError("Could not read the returned segmentation")
        image = node.GetImageData()
        if image is None or image.GetNumberOfScalarComponents() != 1:
            raise ValueError("Returned segmentation must contain scalar labels")
        mask = slicer.util.arrayFromVolume(node)
        matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(matrix)
        returned = ImageGeometry.from_ijk_to_ras(mask.shape, slicer.util.arrayFromVTKMatrix(matrix))
        if returned.shape_zyx != self.geometry.shape_zyx:
            raise ValueError("Returned segmentation dimensions do not match the submitted CT")
        for actual, expected, tolerance in (
            (returned.spacing_xyz, self.geometry.spacing_xyz, GEOMETRY_ATOL),
            (returned.origin_lps, self.geometry.origin_lps, ORIGIN_ATOL),
            (returned.direction_lps, self.geometry.direction_lps, GEOMETRY_ATOL),
        ):
            if not np.allclose(actual, expected, atol=tolerance, rtol=0):
                raise ValueError(
                    "Returned segmentation physical geometry differs from the submitted CT"
                )
        if mask.ndim != 3 or mask.dtype.kind not in "iu" or np.any(mask < 0):
            raise ValueError("Returned segmentation must contain nonnegative integer labels")
        return np.array(mask, copy=True)
