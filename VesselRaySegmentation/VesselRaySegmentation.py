"""Endpoint → ARG → editable cleanup → centerlines → ModelO."""

import numpy as np
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from VesselRaySegmentationLib import VesselRaySegmentationLogic
from VesselRaySegmentationLib.monai.client import ConnectionSettings, ModelDescriptor
from VesselRaySegmentationLib.monai.images import VolumeSnapshot
from VesselRaySegmentationLib.monai.submission import InferenceRequest, InferenceResult
from VesselRaySegmentationLib.widgets import WorkflowPanel


class VesselRaySegmentation(ScriptedLoadableModule):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.parent.title = "Vessel Ray Segmentation"
        self.parent.categories = ["Vascular Modeling"]
        self.parent.dependencies = ["Markups", "Segmentations", "SegmentEditor"]
        self.parent.contributors = ["Moritz Kirschner"]
        self.parent.helpText = "Configure a MONAI Label endpoint, run initial segmentation, filter and trim a copy of the vessel mask, generate or edit native Markups centerlines, then run a refinement segmentation model. Requires SlicerVMTK for centerline generation."
        self.parent.acknowledgementText = "Inspiration from nnunet-interactive"


class VesselRaySegmentationWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        super().setup()
        self.logic = VesselRaySegmentationLogic()
        self.workflow = WorkflowPanel(self.layout, self.logic)

    def cleanup(self) -> None:
        self.workflow.cleanup()


class VesselRaySegmentationTest(ScriptedLoadableModuleTest):
    """Identical real MRML and widget checks in desktop Slicer and SlicerWeb."""

    def setUp(self) -> None:
        slicer.mrmlScene.Clear(0)

    def runTest(self) -> None:
        for name in (
            "test_component_cleanup_preserves_prediction",
            "test_volume_file_geometry",
            "test_centerline_snapshot_keeps_world_geometry",
            "test_model_profiles",
            "test_widget_has_production_workflow",
            "test_missing_vmtk_is_explicit",
            "test_editor_uses_selected_segment",
        ):
            self.setUp()
            getattr(self, name)()

    def _prediction(self):
        array = np.zeros((8, 9, 10), dtype=np.int16)
        matrix = np.eye(4)
        angle = np.pi / 5
        matrix[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        ) * [0.5, 1.0, 2.0]
        matrix[:3, 3] = [10.0, -20.0, 30.0]
        snapshot = VolumeSnapshot.capture(array, matrix, ("fixture", "CT fixture"))
        request = InferenceRequest(snapshot, ConnectionSettings(), "arg_fixture", {}, "image")
        mask = np.zeros(array.shape, dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        mask[6, 7, 8] = 1
        result = InferenceResult(mask, request, {"test_only": True})
        return VesselRaySegmentationLogic.import_result(result), result

    def test_component_cleanup_preserves_prediction(self) -> None:
        original, result = self._prediction()
        segment_id = original.GetSegmentation().GetNthSegmentID(0)
        reference = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        slicer.util.updateVolumeFromArray(reference, result.request.snapshot.array)
        matrix = vtk.vtkMatrix4x4()
        slicer.util.updateVTKMatrixFromArray(matrix, result.request.snapshot.geometry.ijk_to_ras())
        reference.SetIJKToRASMatrix(matrix)
        cleaned, summary = VesselRaySegmentationLogic.prepare_cleanup(original, segment_id, 2.0, 0)
        self.assertEqual(summary["source_components"], 2)
        self.assertEqual(summary["selected_components"], 1)
        self.assertEqual(summary["selected_volume_mm3"], 27.0)
        original_array = slicer.util.arrayFromSegmentBinaryLabelmap(original, segment_id, reference)
        cleaned_id = cleaned.GetSegmentation().GetNthSegmentID(0)
        cleaned_array = slicer.util.arrayFromSegmentBinaryLabelmap(cleaned, cleaned_id, reference)
        np.testing.assert_array_equal(original_array, result.mask)
        self.assertEqual(int(cleaned_array.sum()), 27)
        self.assertNotEqual(original.GetID(), cleaned.GetID())
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        original.SetAndObserveTransformNodeID(transform.GetID())
        with self.assertRaisesRegex(ValueError, "Harden the ARG"):
            VesselRaySegmentationLogic.prepare_cleanup(original, segment_id, 2.0, 0)

    def test_volume_file_geometry(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from VesselRaySegmentationLib.scene import capture_volume, source_is_unchanged

        _prediction, result = self._prediction()
        snapshot = VolumeSnapshot.capture(
            result.mask,
            result.request.snapshot.geometry.ijk_to_ras(),
            ("mask", "mask fixture"),
        )
        with TemporaryDirectory() as temporary:
            for suffix in (".nrrd", ".nii.gz"):
                path = Path(temporary) / ("mask" + suffix)
                snapshot.write_image(path)
                np.testing.assert_array_equal(result.request.snapshot.read_mask(path), result.mask)
            shifted = np.array(snapshot.geometry.ijk_to_ras(), copy=True)
            shifted[0, 3] += 1
            wrong = VolumeSnapshot.capture(result.mask, shifted, ("wrong", "wrong geometry"))
            wrong.write_image(path)
            with self.assertRaisesRegex(ValueError, "physical geometry"):
                result.request.snapshot.read_mask(path)
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Editable CT")
        slicer.util.updateVolumeFromArray(volume, np.zeros((3, 4, 5), np.int16))
        captured = capture_volume(volume)
        self.assertTrue(source_is_unchanged(captured))
        slicer.util.arrayFromVolume(volume)[0, 0, 0] = 1
        self.assertFalse(source_is_unchanged(captured))

    def test_centerline_snapshot_keeps_world_geometry(self) -> None:
        from VesselRaySegmentationLib.centerlines import collect_polylines

        root = VesselRaySegmentationLogic.create_centerline()
        curve = VesselRaySegmentationLogic.create_centerline()
        curve.AddControlPoint(vtk.vtkVector3d(1.25, 2.5, 3.75))
        curve.AddControlPoint(vtk.vtkVector3d(5.25, 2.5, 3.75))
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 3, 10)
        transform.SetMatrixTransformToParent(matrix)
        curve.SetAndObserveTransformNodeID(transform.GetID())
        hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        root_id = hierarchy.GetItemByDataNode(root)
        hierarchy.SetItemParent(hierarchy.GetItemByDataNode(curve), root_id)
        polylines = collect_polylines(root_id, np.eye(4))
        self.assertEqual(len(polylines), 1)
        np.testing.assert_allclose(polylines[0][0], [3.75, 2.5, 11.25])
        np.testing.assert_allclose(polylines[0][-1], [3.75, 2.5, 15.25])

    def test_model_profiles(self) -> None:
        models = {
            name: ModelDescriptor(name, name, "", (), profile)
            for name, profile in [
                ("arg", "image"),
                ("model_o", "polylines"),
                ("unknown", "other"),
            ]
        }
        arg, model_o = VesselRaySegmentationLogic.select_models(models)
        self.assertEqual([model.name for model in arg], ["arg"])
        self.assertEqual([model.name for model in model_o], ["model_o"])

    def test_widget_has_production_workflow(self) -> None:
        module = slicer.app.moduleManager().module("VesselRaySegmentation")
        self.assertIsNotNone(module, "The installed production module must be discovered")
        representation = module.widgetRepresentation()
        widget = representation.self() if hasattr(representation, "self") else representation
        self.assertIsNone(getattr(module, "_setupError", None))
        self.assertIsInstance(widget.logic, VesselRaySegmentationLogic)
        panel = widget.workflow
        self.assertEqual(panel.arg_button.text, "Run ARG inference")
        self.assertEqual(panel.model_o_button.text, "Run ModelO inference")
        panel.create_curve_button.click()
        curve = panel.centerline.currentNode()
        self.assertIsNotNone(curve)
        self.assertEqual(curve.GetNumberOfControlPoints(), 0)

    def test_missing_vmtk_is_explicit(self) -> None:
        if slicer.app.moduleManager().module("ExtractCenterline") is None:
            with self.assertRaisesRegex(RuntimeError, "SlicerVMTK"):
                VesselRaySegmentationLogic.extract_centerlines(None, "")

    def test_editor_uses_selected_segment(self) -> None:
        import sys

        from VesselRaySegmentationLib.scene import open_segment_editor

        segmentation, _result = self._prediction()
        selected = segmentation.GetSegmentation().AddEmptySegment("edited", "Selected vessel")
        open_segment_editor(segmentation, None, selected)
        if sys.platform == "emscripten":
            from slicerweb import segment_editor

            actual = segment_editor.editor().logic.GetCurrentSegmentID()
        else:
            actual = (
                slicer.modules.segmenteditor.widgetRepresentation().self().editor.currentSegmentID()
            )
        self.assertEqual(actual, selected)
