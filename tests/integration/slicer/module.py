"""Run the installed module's identical tests in desktop Slicer and SlicerWeb."""

from __future__ import annotations

import asyncio
import io
import json
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
    hierarchy.RemoveItem(hierarchy.GetItemByDataNode(curve))
    slicer.mrmlScene.RemoveNode(segmentation)
    return {
        "status": "passed",
        "coverage_mm": coverage,
        "curve_count": summary["curve_count"],
        "method": "VMTK approximate network",
    }


def _panel():
    module = slicer.app.moduleManager().module("VesselRaySegmentation")
    representation = module.widgetRepresentation()
    widget = representation.self() if hasattr(representation, "self") else representation
    return widget.workflow


def request_state() -> str:
    """Report the production panel's request state to the browser's real button clicks."""
    panel = _panel()
    feedback = panel.feedback
    return json.dumps(
        {
            "models": len(panel.models),
            "requests": feedback.request,
            "pending": panel.pending,
            "busy": not feedback.progress.isHidden(),
            "unlocked": all(control.enabled for control in panel.controls),
            "status": str(feedback.status.text),
        }
    )


async def _pump_until(panel, ready, failure: str) -> None:
    deadline = asyncio.get_running_loop().time() + 45
    while not ready():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(failure + ": " + str(panel.feedback.status.text))
        if sys.platform != "emscripten":
            slicer.app.processEvents()
        await asyncio.sleep(0.01)


async def _settled(panel) -> None:
    await _pump_until(panel, lambda: not panel.pending, "Hosted workflow stalled")


def _assert_idle(panel, outcome: str) -> None:
    """Request-only feedback disappears, its timers stop and every workflow control unlocks."""
    feedback = panel.feedback
    status = str(feedback.status.text)
    assert outcome in status, status
    assert not panel.pending and not feedback.clock.active, status
    assert not feedback.check_timer.active, status
    assert feedback.progress.isHidden() and feedback.detail.isHidden(), status
    assert not feedback.discard_button.enabled, status
    assert all(control.enabled for control in panel.controls), status
    assert panel.refresh_button.text == "Refresh models", status
    assert panel.arg_button.text == "Run ARG inference", status
    assert panel.model_o_button.text == "Run ModelO inference", status


def _record_progress(panel) -> list:
    """Observe the real updates delivered to the production feedback."""
    updates = []
    show = panel.feedback.show

    def record(update):
        updates.append(update)
        show(update)

    panel.feedback.show = record
    return updates


async def _assert_waiting(panel, updates) -> None:
    """While the server holds the request, the probe controller checks the real endpoint."""
    from VesselRaySegmentationLib.monai.progress import RequestPhase

    feedback = panel.feedback
    await _pump_until(
        panel,
        lambda: (
            updates
            and updates[-1].phase is RequestPhase.WAITING
            and feedback.last_check is not None
            and not feedback.detail.isHidden()
        ),
        "Held request showed no endpoint check",
    )
    status = str(feedback.status.text)
    assert panel.pending and feedback.check_timer.active, status
    # A busy indicator, not fabricated model progress.
    assert not feedback.progress.isHidden() and feedback.progress.maximum == 0, status
    assert feedback.discard_button.enabled and not panel.model_o_button.enabled, status
    assert panel.model_o_button.text == "Running ModelO inference...", status
    assert feedback.last_check[0], str(feedback.detail.text)
    # The check ran on the probe controller; the request controller still owns the request.
    assert panel.tasks.busy, status


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
        raise AssertionError("ARG result missing: " + str(panel.feedback.status.text))
    expected = np.array(_segment_mask(raw, volume), copy=True)
    assert int(expected.sum()) == 28
    panel.prepare_cleanup()
    cleaned = panel.cleaned_segment.currentNode()
    assert cleaned is not None and cleaned is not raw
    mask = np.array(_segment_mask(cleaned, volume), copy=True)
    assert int(mask.sum()) == 27
    mask[1, 1, 1] = 0  # Simulate a manual edit through Slicer's native segment API.
    segment_id = cleaned.GetSegmentation().GetNthSegmentID(0)
    slicer.util.updateSegmentBinaryLabelmapFromArray(mask, cleaned, segment_id, volume)
    np.testing.assert_array_equal(_segment_mask(raw, volume), expected)
    assert int(_segment_mask(cleaned, volume).sum()) == 26
    return raw, cleaned


def _workflow_curve(panel):
    import vtk

    curve = panel.logic.create_centerline()
    panel.centerline.setCurrentNode(curve)
    curve.AddControlPoint(vtk.vtkVector3d(11.0, -18.0, 2.0))
    curve.AddControlPoint(vtk.vtkVector3d(11.0, -18.0, 6.0))
    return curve


async def test_hosted_workflow(
    endpoint: str, token: str, wait_for_request, release_request
) -> dict:
    """Exercise real native Label HTTP and the installed production widget on either host."""
    import qt
    import vtk

    panel = _panel()
    settings = qt.QSettings()
    previous = settings.value("VesselRaySegmentation/endpoint", None)
    updates = _record_progress(panel)
    try:
        volume = _workflow_scene(panel)
        panel.endpoint.text = endpoint
        panel.token.text = token
        panel.refresh()
        await _settled(panel)
        _assert_idle(panel, "Endpoint ready")
        from VesselRaySegmentationLib.monai.progress import RequestPhase

        assert [model.name for model in panel.arg_models] == ["arg-fixture"]
        assert [model.name for model in panel.model_o_models] == ["model-o-fixture"]
        panel.run_arg()
        # The click is acknowledged before the deferred scene capture runs.
        assert panel.pending and not panel.feedback.progress.isHidden()
        assert panel.arg_button.text == "Running ARG inference..."
        assert not panel.arg_button.enabled and panel.feedback.discard_button.enabled
        await _settled(panel)
        _assert_idle(panel, "Finished in")
        phases = [update.phase for update in updates]
        # The first request uploads the input once, then reports the endpoint session.
        assert phases.index(RequestPhase.UPLOADING) < phases.index(RequestPhase.SESSION_READY)
        assert phases.index(RequestPhase.WAITING) < phases.index(RequestPhase.READING), phases
        uploads = [update for update in updates if update.total_bytes]
        if sys.platform != "emscripten":
            # Browser fetch cannot observe upload bytes; desktop must report them.
            assert uploads and uploads[-1].completed_bytes == uploads[-1].total_bytes, uploads
        raw, cleaned = _workflow_cleanup(panel, volume)
        curve = _workflow_curve(panel)
        updates.clear()
        panel.run_model_o()
        await _settled(panel)
        _assert_idle(panel, "Finished in")
        phases = [update.phase for update in updates]
        # Later requests reuse the uploaded input instead of sending it again.
        assert RequestPhase.SESSION_READY in phases and RequestPhase.UPLOADING not in phases, phases
        first = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")[-1]
        assert first not in (raw, cleaned), str(panel.feedback.status.text)
        assert int(_segment_mask(first, volume).sum()) == 3
        curve.SetNthControlPointPositionWorld(1, vtk.vtkVector3d(11.0, -18.0, 4.0))
        panel.run_model_o()
        await _settled(panel)
        second = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")[-1]
        assert second is not first, str(panel.feedback.status.text)
        assert int(_segment_mask(second, volume).sum()) == 2
        _assert_idle(panel, "Finished in")
        # Check sooner than the production interval so the held request sees a check.
        panel.feedback.check_timer.setInterval(200)
        updates.clear()
        panel.run_model_o()
        # Pump Qt so the deferred submission is sent, then check the held request's feedback.
        await _assert_waiting(panel, updates)
        # The harness observes the real server's start receipt, rather than guessing a delay.
        await wait_for_request()
        closed_at = time.time()
        slicer.mrmlScene.Clear(0)
        feedback = panel.feedback
        assert panel.pending and not feedback.progress.isHidden(), str(feedback.status.text)
        assert "Scene closed" in str(feedback.status.text) and not feedback.discard_button.enabled
        assert not feedback.check_timer.active, str(feedback.status.text)
        await release_request()
        await _settled(panel)
        _assert_idle(panel, "Result discarded")
        assert slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLSegmentationNode") == 0
        assert panel.logic.snapshot is None
        return {
            "status": "passed",
            "arg_voxels": 28,
            "cleaned_voxels": 26,
            "model_o_voxels": [3, 2],
            "upload_progress_updates": len(uploads),
            "request_feedback": "passed",
            "scene_close_discards_result": True,
            "centerline_mode": "native editable Markups",
            "trained_model": False,
            "scene_closed_at": closed_at,
        }
    finally:
        del panel.feedback.show
        panel.feedback.check_timer.setInterval(15000)
        panel.logic.cleanup()
        panel.tasks.release_session()
        panel.token.text = ""
        if previous is None:
            settings.remove("VesselRaySegmentation/endpoint")
        else:
            settings.setValue("VesselRaySegmentation/endpoint", previous)
