"""Small MRML and widget integration demo for desktop Slicer and SlicerWeb."""

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class VesselRaySegmentationDemo(ScriptedLoadableModule):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.parent.title = "Vessel Ray Segmentation Demo"
        self.parent.categories = ["Vascular Modeling"]
        self.parent.dependencies = ["Markups"]
        self.parent.contributors = ["Moritz Kirschner"]
        self.parent.helpText = (
            "Create a synthetic curve to check extension installation and testing."
        )
        self.parent.acknowledgementText = ""


class VesselRaySegmentationDemoWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        super().setup()
        self.logic = VesselRaySegmentationDemoLogic()
        self.createButton = qt.QPushButton("Create demo curve")
        self.createButton.objectName = "CreateDemoCurveButton"
        self.createButton.toolTip = "Add a synthetic three-point curve to the current scene."
        self.layout.addWidget(self.createButton)
        self.createButton.connect("clicked()", self.onCreateCurve)
        self.layout.addStretch(1)

    def onCreateCurve(self) -> None:
        self.logic.createDemoCurve()

    def cleanup(self) -> None:
        self.createButton.disconnect("clicked()", self.onCreateCurve)


class VesselRaySegmentationDemoLogic(ScriptedLoadableModuleLogic):
    def createDemoCurve(self):
        """Create editable native Markups without downloading data or model weights."""
        curve = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Demo vessel curve")
        curve.SetCurveTypeToLinear()
        for point in ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 5.0, 0.0)):
            curve.AddControlPoint(vtk.vtkVector3d(*point))
        curve.CreateDefaultDisplayNodes()
        return curve


class VesselRaySegmentationDemoTest(ScriptedLoadableModuleTest):
    def setUp(self) -> None:
        slicer.mrmlScene.Clear(0)

    def runTest(self) -> None:
        self.setUp()
        self.test_curve_geometry()
        self.setUp()
        self.test_widget_creates_curve()

    def test_curve_geometry(self) -> None:
        curve = VesselRaySegmentationDemoLogic().createDemoCurve()
        self.assertEqual(curve.GetNumberOfControlPoints(), 3)
        self.assertIsNotNone(curve.GetDisplayNode())
        endpoint = [0.0, 0.0, 0.0]
        curve.GetNthControlPointPositionWorld(2, endpoint)
        self.assertEqual(endpoint, [20.0, 5.0, 0.0])
        self.assertAlmostEqual(curve.GetCurveLengthWorld(), 10.0 + 125.0**0.5, places=5)

    def test_widget_creates_curve(self) -> None:
        module = slicer.app.moduleManager().module("VesselRaySegmentationDemo")
        self.assertIsNotNone(module, "The installed module must be discovered by Slicer")
        representation = module.widgetRepresentation()
        # Desktop exposes the Python widget through self(); SlicerWeb returns it directly.
        widget = representation.self() if hasattr(representation, "self") else representation
        self.assertIsInstance(widget.logic, VesselRaySegmentationDemoLogic)
        widget.createButton.click()
        self.assertEqual(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLMarkupsCurveNode"), 1)
        curve = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsCurveNode")
        self.assertEqual(curve.GetNumberOfControlPoints(), 3)
