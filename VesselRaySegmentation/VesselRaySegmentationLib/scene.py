"""MRML snapshots, editable segmentation copies and native editing panels."""

from __future__ import annotations

from typing import Any

import numpy as np

from .monai.images import VolumeSnapshot


def capture_volume(volume: Any) -> VolumeSnapshot:
    import slicer
    import vtk

    if volume is None or not volume.IsA("vtkMRMLScalarVolumeNode"):
        raise ValueError("Select the input CT volume")
    if volume.GetParentTransformNode():
        raise ValueError("Harden or resample the CT transform before inference")
    matrix = vtk.vtkMatrix4x4()
    volume.GetIJKToRASMatrix(matrix)
    return VolumeSnapshot.capture(
        slicer.util.arrayFromVolume(volume),
        slicer.util.arrayFromVTKMatrix(matrix),
        (volume.GetID(), volume.GetName()),
    )


def source_is_unchanged(snapshot: VolumeSnapshot) -> bool:
    """Check the live source only on the UI thread before accepting its response."""
    import slicer
    import vtk

    node = slicer.mrmlScene.GetNodeByID(snapshot.source_id)
    if node is None or node.GetParentTransformNode() or node.GetImageData() is None:
        return False
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return snapshot.matches(
        slicer.util.arrayFromVolume(node),
        slicer.util.arrayFromVTKMatrix(matrix),
        (node.GetID(), snapshot.source_name),
    )


def import_result(result: Any) -> Any:
    """Materialize only a finished, geometry-validated response on the UI thread."""
    import slicer
    import vtk

    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    segmentation = None
    try:
        matrix = vtk.vtkMatrix4x4()
        slicer.util.updateVTKMatrixFromArray(matrix, result.request.snapshot.geometry.ijk_to_ras())
        labelmap.SetIJKToRASMatrix(matrix)
        slicer.util.updateVolumeFromArray(labelmap, result.mask)
        segmentation = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", result.request.output_name
        )
        segmentation.CreateDefaultDisplayNodes()
        if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmap, segmentation
        ):
            raise ValueError("Could not import the returned segmentation")
        segmentation.SetAttribute("VesselRaySegmentation.Provenance", result.provenance())
        return segmentation
    except Exception:
        if segmentation is not None:
            slicer.mrmlScene.RemoveNode(segmentation)
        raise
    finally:
        slicer.mrmlScene.RemoveNode(labelmap)


def segment_image(segmentation: Any, segment_id: str) -> Any:
    import slicer

    if segmentation is None or not segment_id:
        raise ValueError("Select the ARG vessel segment")
    image = slicer.vtkOrientedImageData()
    if not segmentation.GetBinaryLabelmapRepresentation(segment_id, image):
        raise ValueError("The selected segment has no binary labelmap")
    return image


def filter_components(image: Any, minimum_mm3: float, largest_count: int) -> tuple[Any, dict]:
    """Rank native face-connected islands by mass and preserve image geometry."""
    import slicer
    import vtk

    if not np.isfinite(minimum_mm3) or minimum_mm3 < 0 or largest_count < 0:
        raise ValueError("Component volume and count must be nonnegative")
    connectivity = vtk.vtkImageConnectivityFilter()
    connectivity.SetInputData(image)
    connectivity.SetScalarRange(1, 1)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.SetLabelModeToSizeRank()
    connectivity.SetLabelScalarTypeToInt()
    connectivity.Update()
    values = connectivity.GetExtractedRegionSizes()
    sizes = sorted(
        (int(values.GetValue(i)) for i in range(values.GetNumberOfValues())), reverse=True
    )
    voxel_mm3 = float(np.prod(image.GetSpacing()))
    count = sum(size * voxel_mm3 >= minimum_mm3 for size in sizes)
    if largest_count:
        count = min(count, largest_count)
    if count == 0:
        raise ValueError("No connected components meet the selected filter")
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(connectivity.GetOutputPort())
    threshold.ThresholdBetween(1, count)
    threshold.SetInValue(1)
    threshold.SetOutValue(0)
    threshold.SetOutputScalarTypeToUnsignedChar()
    threshold.Update()
    matrix = vtk.vtkMatrix4x4()
    image.GetImageToWorldMatrix(matrix)
    selected = slicer.vtkOrientedImageData()
    selected.DeepCopy(threshold.GetOutput())
    selected.SetGeometryFromImageToWorldMatrix(matrix)
    return selected, {
        "source_components": len(sizes),
        "selected_components": count,
        "selected_volume_mm3": sum(sizes[:count]) * voxel_mm3,
    }


def prepare_cleanup(
    segmentation: Any, segment_id: str, minimum_mm3: float = 0, largest_count: int = 1
) -> tuple[Any, dict]:
    """Create a separate editable result; keep the original ARG prediction intact."""
    import slicer

    if segmentation is not None and segmentation.GetParentTransformNode():
        raise ValueError("Harden the ARG segmentation transform before filtering physical volumes")
    image, summary = filter_components(
        segment_image(segmentation, segment_id), minimum_mm3, largest_count
    )
    output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ARG cleaned")
    try:
        name = segmentation.GetSegmentation().GetSegment(segment_id).GetName()
        output.AddSegmentFromBinaryLabelmapRepresentation(image, name, [0.0, 0.8, 1.0])
        output.CreateDefaultDisplayNodes()
        output.SetAttribute("VesselRaySegmentation.SourceARG", segmentation.GetID())
        return output, summary
    except Exception:
        slicer.mrmlScene.RemoveNode(output)
        raise


def open_segment_editor(segmentation: Any, volume: Any, segment_id: str) -> None:
    import sys

    import slicer

    if segmentation is None or not segment_id:
        raise ValueError("Create or select a cleaned ARG segmentation first")
    if sys.platform == "emscripten":
        from slicerweb import segment_editor

        from .browser import open_panel

        open_panel("SegmentEditor")
        segment_editor.editor().setup(segmentation.GetID(), volume.GetID() if volume else None)
        segment_editor.editor().selectSegment(segment_id)
        return
    if slicer.util.mainWindow() is not None:
        slicer.util.selectModule("SegmentEditor")
    widget = slicer.modules.segmenteditor.widgetRepresentation().self()
    widget.editor.setSegmentationNode(segmentation)
    widget.editor.setSourceVolumeNode(volume)
    widget.editor.setCurrentSegmentID(segment_id)
