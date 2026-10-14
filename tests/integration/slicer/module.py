"""Run the installed module's identical tests in desktop Slicer and SlicerWeb."""

from __future__ import annotations

import asyncio
import io
import sys
import time
import unittest
from pathlib import Path

import slicer


def test_installed_module(package_root: str) -> dict[str, object]:
    import VesselRaySegmentation as module

    loaded = Path(module.__file__).resolve()
    if not loaded.is_relative_to(Path(package_root).resolve()):
        raise AssertionError(f"Module loaded outside the installed package: {loaded}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.VesselRaySegmentationTest)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print(log.getvalue(), flush=True)
    if result.testsRun != 7 or not result.wasSuccessful():
        raise AssertionError(log.getvalue())
    return {
        "status": "passed",
        "tests_run": result.testsRun,
        "module_path": str(loaded),
        "slicer_version": str(slicer.app.applicationVersion),
        "automatic_centerlines": test_automatic_centerlines(),
    }


def test_automatic_centerlines() -> dict:
    """Require actual tube coverage when the host has SlicerVMTK installed."""
    import numpy as np
    from VesselRaySegmentationLib import VesselRaySegmentationLogic
    from VesselRaySegmentationLib.centerlines import collect_polylines
    from VesselRaySegmentationLib.monai.client import ConnectionSettings
    from VesselRaySegmentationLib.monai.images import VolumeSnapshot
    from VesselRaySegmentationLib.monai.submission import InferenceRequest, InferenceResult

    if slicer.app.moduleManager().module("ExtractCenterline") is None:
        return {
            "status": "unavailable",
            "dependency": "SlicerVMTK",
            "alternative": "native editable Markups curves",
        }
    z, y, x = np.indices((48, 48, 48))
    mask = (((x - 24) ** 2 + (y - 24) ** 2 <= 9) & (z >= 8) & (z <= 40)).astype(np.uint8)
    snapshot = VolumeSnapshot.capture(mask, np.eye(4), ("tube-fixture", "Tube fixture"))
    request = InferenceRequest(snapshot, ConnectionSettings(), "fixture", {}, "image")
    segmentation = VesselRaySegmentationLogic.import_result(InferenceResult(mask, request, {}))
    segment_id = segmentation.GetSegmentation().GetNthSegmentID(0)
    curve, summary = VesselRaySegmentationLogic.extract_centerlines(segmentation, segment_id)
    hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    lines = collect_polylines(hierarchy.GetItemByDataNode(curve), np.eye(4))
    coverage = float(np.ptp(np.concatenate([np.asarray(line)[:, 0] for line in lines])))
    if coverage < 25 or summary["failed_components"]:
        raise AssertionError(f"VMTK did not cover the straight tube: {coverage} mm, {summary}")
    # Leave the widget test's single manually created curve for the external click check.
    hierarchy.RemoveItem(hierarchy.GetItemByDataNode(curve))
    slicer.mrmlScene.RemoveNode(segmentation)
    return {
        "status": "passed",
        "coverage_mm": coverage,
        "curve_count": summary["curve_count"],
        "method": "VMTK approximate network",
    }


async def _settled(panel) -> None:
    deadline = asyncio.get_running_loop().time() + 45
    while panel.tasks.busy:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("Hosted workflow stalled: " + str(panel.status.text))
        if sys.platform != "emscripten":
            slicer.app.processEvents()
        await asyncio.sleep(0.01)


def _workflow_scene(panel):
    import numpy as np
    import vtk

    slicer.mrmlScene.Clear(0)
    array = np.full((8, 9, 10), -1000, dtype=np.int16)
    array[1:4, 1:4, 1:4] = 400
    array[6, 7, 8] = 400
    volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Workflow CT")
    slicer.util.updateVolumeFromArray(volume, array)
    matrix = vtk.vtkMatrix4x4()
    matrix.SetElement(0, 0, 0.5)
    matrix.SetElement(2, 2, 2.0)
    matrix.SetElement(0, 3, 10.0)
    matrix.SetElement(1, 3, -20.0)
    volume.SetIJKToRASMatrix(matrix)
    panel.volume.setCurrentNode(volume)
    return volume


def _segment_mask(node, volume):
    return slicer.util.arrayFromSegmentBinaryLabelmap(
        node, node.GetSegmentation().GetNthSegmentID(0), volume
    )


def _workflow_cleanup(panel, volume):
    import numpy as np

    raw = panel.arg_segment.currentNode()
    if raw is None:
        raise AssertionError("ARG result missing: " + str(panel.status.text))
    expected = np.array(_segment_mask(raw, volume), copy=True)
    assert int(expected.sum()) == 28
    panel.prepare_cleanup()
    cleaned = panel.cleaned_segment.currentNode()
    assert cleaned is not None and cleaned is not raw
    mask = np.array(_segment_mask(cleaned, volume), copy=True)
    assert int(mask.sum()) == 27
    panel.open_editor()
    mask[1, 1, 1] = 0  # Simulate a manual edit through Slicer's native segment API.
    segment_id = cleaned.GetSegmentation().GetNthSegmentID(0)
    slicer.util.updateSegmentBinaryLabelmapFromArray(mask, cleaned, segment_id, volume)
    np.testing.assert_array_equal(_segment_mask(raw, volume), expected)
    assert int(_segment_mask(cleaned, volume).sum()) == 26
    return raw, cleaned


def _workflow_curve(panel):
    import vtk

    panel.create_curve()
    curve = panel.centerline.currentNode()
    curve.AddControlPoint(vtk.vtkVector3d(11.0, -18.0, 2.0))
    curve.AddControlPoint(vtk.vtkVector3d(11.0, -18.0, 6.0))
    return curve


async def test_hosted_workflow(endpoint: str, token: str, wait_for_request) -> dict:
    """Exercise real native Label HTTP and the installed production widget on either host."""
    import qt
    import vtk

    module = slicer.app.moduleManager().module("VesselRaySegmentation")
    representation = module.widgetRepresentation()
    widget = representation.self() if hasattr(representation, "self") else representation
    panel = widget.workflow
    settings = qt.QSettings()
    previous = settings.value("VesselRaySegmentation/endpoint", None)
    try:
        volume = _workflow_scene(panel)
        panel.endpoint.text = endpoint
        panel.token.text = token
        panel.refresh()
        await _settled(panel)
        assert [model.name for model in panel.arg_models] == ["arg-fixture"]
        assert [model.name for model in panel.model_o_models] == ["model-o-fixture"]
        panel.run_arg()
        await _settled(panel)
        raw, cleaned = _workflow_cleanup(panel, volume)
        curve = _workflow_curve(panel)
        panel.run_model_o()
        await _settled(panel)
        first = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")[-1]
        assert first not in (raw, cleaned), str(panel.status.text)
        assert int(_segment_mask(first, volume).sum()) == 3
        curve.SetNthControlPointPositionWorld(1, vtk.vtkVector3d(11.0, -18.0, 4.0))
        panel.run_model_o()
        await _settled(panel)
        second = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")[-1]
        assert second is not first, str(panel.status.text)
        assert int(_segment_mask(second, volume).sum()) == 2
        panel.run_model_o()
        # The harness observes the real server's start receipt, rather than guessing a delay.
        await wait_for_request()
        closed_at = time.time()
        slicer.mrmlScene.Clear(0)
        await _settled(panel)
        assert slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode") == 0
        assert panel.logic.snapshot is None
        return {
            "status": "passed",
            "arg_voxels": 28,
            "cleaned_voxels": 26,
            "model_o_voxels": [3, 2],
            "scene_close_discards_result": True,
            "centerline_mode": "native editable Markups",
            "trained_model": False,
            "scene_closed_at": closed_at,
        }
    finally:
        panel.logic.cleanup()
        panel.tasks.release_session()
        panel.token.text = ""
        if previous is None:
            settings.remove("VesselRaySegmentation/endpoint")
        else:
            settings.setValue("VesselRaySegmentation/endpoint", previous)
