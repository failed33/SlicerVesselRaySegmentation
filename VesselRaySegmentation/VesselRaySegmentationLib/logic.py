"""Coordinate the production workflow using native scene objects."""

from __future__ import annotations

import sys

from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from . import centerlines, scene
from .monai.client import ConnectionSettings, ModelDescriptor
from .monai.sessions import VolumeSession
from .monai.submission import InferenceRequest


class VesselRaySegmentationLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot = None
        self.session = None

    @staticmethod
    def select_models(models: dict[str, ModelDescriptor]) -> tuple[list, list]:
        """Use endpoint metadata; unknown model profiles are unsupported."""
        return (
            [model for model in models.values() if model.input_profile == "image"],
            [model for model in models.values() if model.input_profile == "polylines"],
        )

    def _snapshot(self, volume):
        import slicer
        import vtk

        if volume is None or volume.GetParentTransformNode():
            return scene.capture_volume(volume)
        matrix = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(matrix)
        if self.snapshot and self.snapshot.matches(
            slicer.util.arrayFromVolume(volume),
            slicer.util.arrayFromVTKMatrix(matrix),
            (volume.GetID(), volume.GetName()),
        ):
            return self.snapshot
        self.snapshot = scene.capture_volume(volume)
        return self.snapshot

    def request(
        self, volume, model: ModelDescriptor, connection: ConnectionSettings, centerline_node=None
    ) -> InferenceRequest:
        import slicer

        snapshot = self._snapshot(volume)
        parameters = {}
        if model.input_profile == "polylines":
            if centerline_node is None:
                raise ValueError("Select the edited centerline curves for ModelO")
            hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            parameters["polylines_zyx"] = centerlines.collect_polylines(
                hierarchy.GetItemByDataNode(centerline_node), snapshot.geometry.ijk_to_ras()
            )
        elif model.input_profile != "image":
            raise ValueError(f"Unsupported model input profile: {model.input_profile}")
        if sys.platform != "emscripten":
            if self.session and (
                self.session.snapshot is not snapshot or self.session.connection != connection
            ):
                self.session.close_in_background()
                self.session = None
            self.session = self.session or VolumeSession(connection, snapshot)
        return InferenceRequest(
            snapshot,
            connection,
            model.name,
            parameters,
            model.input_profile,
            "ARG prediction" if model.input_profile == "image" else "ModelO segmentation",
            session=self.session,
        )

    @staticmethod
    def import_result(result):
        return scene.import_result(result)

    @staticmethod
    def prepare_cleanup(
        segmentation, segment_id: str, minimum_mm3: float = 0, largest_count: int = 1
    ):
        return scene.prepare_cleanup(segmentation, segment_id, minimum_mm3, largest_count)

    @staticmethod
    def extract_centerlines(segmentation, segment_id: str):
        import slicer

        if slicer.app.moduleManager().module("ExtractCenterline") is None:
            raise RuntimeError(
                "Automatic centerlines require the SlicerVMTK extension. "
                "Install it, or create and edit Markups curves manually."
            )
        if segmentation is None or not segment_id:
            raise ValueError("Select the cleaned ARG vessel segment")
        if segmentation.GetParentTransformNode():
            raise ValueError("Harden the cleaned segmentation transform before extraction")
        extraction = centerlines.extract_networks(segmentation, segment_id)
        if not extraction.summary["curve_count"]:
            raise ValueError("VMTK produced no usable centerlines; inspect the cleaned mask")
        root = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Vessel centerlines")
        centerlines.add_network_curves(extraction.network, root)
        return root, extraction.summary

    @staticmethod
    def create_centerline():
        """Create an editable, empty native curve; the user supplies its geometry."""
        import slicer

        curve = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Vessel centerline")
        curve.CreateDefaultDisplayNodes()
        curve.SetCurveTypeToLinear()
        return curve

    def cleanup(self) -> None:
        if self.session:
            self.session.close_in_background()
            self.session = None
        self.snapshot = None
