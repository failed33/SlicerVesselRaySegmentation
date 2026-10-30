"""The endpoint, ARG, cleanup, centerline and ModelO workflow panel."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import qt
import slicer

from . import scene
from .monai.client import ConnectionSettings
from .monai.progress import RequestProgress
from .tasks import TaskController

# Give Qt and the browser one display frame to paint an acknowledgement before
# main-thread scene capture or result import blocks the event loop.
ACKNOWLEDGE_DELAY_MS = 50


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"


def _size(count: float) -> str:
    for unit in ("B", "kB", "MB"):
        if count < 1000:
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1000
    return f"{count:.1f} GB"


class WorkflowPanel:
    def __init__(self, layout, logic) -> None:
        self.logic = logic
        self.tasks = TaskController()
        self.pending = False
        self.started = 0.0
        self.endpoint_check: tuple[bool, str] | None = None
        self.running_button: tuple[Any, str] | None = None
        self.models = {}
        self.arg_models = []
        self.model_o_models = []
        self.controls = []
        self._setup_endpoint(layout)
        self._setup_feedback(layout)
        self._setup_arg(layout)
        self._setup_cleanup(layout)
        self._setup_centerlines(layout)
        self._setup_model_o(layout)
        layout.addStretch(1)
        self.scene_observer = slicer.mrmlScene.AddObserver(
            slicer.vtkMRMLScene.StartCloseEvent, self._scene_closing
        )

    def _setup_feedback(self, parent) -> None:
        # Sits beneath the endpoint, above every step that sends a request. The status
        # line stays in place; request-only rows appear beneath it.
        self.feedback = qt.QWidget()
        layout = qt.QVBoxLayout(self.feedback)
        layout.setContentsMargins(0, 0, 0, 0)
        parent.addWidget(self.feedback)
        self.status = qt.QLabel("Configure the endpoint, then refresh its available models.")
        self.status.setTextFormat(qt.Qt.PlainText)
        self.status.wordWrap = True
        layout.addWidget(self.status)
        self.activity = qt.QProgressBar()
        self.activity.visible = False
        self.activity.toolTip = (
            "Shows that a request is active. Upload progress is measured; "
            "the server does not report model progress."
        )
        layout.addWidget(self.activity)
        self.liveness = qt.QLabel()
        self.liveness.setTextFormat(qt.Qt.PlainText)
        self.liveness.wordWrap = True
        self.liveness.visible = False
        layout.addWidget(self.liveness)
        self.clock = qt.QTimer()
        self.clock.setInterval(1000)
        self.clock.connect("timeout()", self._show_liveness)
        self.cancel_button = qt.QPushButton("Discard pending result")
        self.cancel_button.enabled = False
        self.cancel_button.connect("clicked()", self._discard)
        layout.addWidget(self.cancel_button)

    def _scene_closing(self, _caller, _event) -> None:
        self.tasks.cancel()
        self.tasks.release_session()
        self.logic.cleanup()
        if self.pending:
            self.cancel_button.enabled = False
            self._phase("Scene closed. The pending result will be discarded.")

    def _group(self, layout, title):
        group = qt.QGroupBox(title)
        fields = qt.QFormLayout(group)
        layout.addWidget(group)
        return fields

    def _button(self, fields, text, callback, name):
        button = qt.QPushButton(text)
        button.objectName = name
        button.connect("clicked()", callback)
        fields.addRow(button)
        self.controls.append(button)
        return button

    def _node(self, fields, label, kinds):
        widget = slicer.qMRMLNodeComboBox()
        widget.nodeTypes = kinds
        widget.noneEnabled = True
        widget.addEnabled = False
        widget.removeEnabled = False
        widget.setMRMLScene(slicer.mrmlScene)
        fields.addRow(label, widget)
        self.controls.append(widget)
        return widget

    def _setup_endpoint(self, layout) -> None:
        fields = self._group(layout, "1. Endpoint configuration")
        settings = qt.QSettings()
        self.endpoint = qt.QLineEdit(str(settings.value("VesselRaySegmentation/endpoint", "")))
        self.endpoint.placeholderText = "https://your-monai-label-server"
        fields.addRow("MONAI Label URL:", self.endpoint)
        self.token = qt.QLineEdit()
        if self.tasks.browser:
            from .browser import mask_token_input

            self.token_observer = mask_token_input(self.token)
        else:
            self.token.setEchoMode(qt.QLineEdit.Password)
        fields.addRow("Bearer token:", self.token)
        self.timeout = qt.QSpinBox()
        self.timeout.minimum = 1
        self.timeout.maximum = 86400
        self.timeout.value = 900
        fields.addRow("Timeout in seconds:", self.timeout)
        self.refresh_button = self._button(
            fields, "Refresh models", self.refresh, "RefreshModelsButton"
        )
        self.controls.extend([self.endpoint, self.token, self.timeout])

    def _setup_arg(self, layout) -> None:
        fields = self._group(layout, "2. ARG inference")
        self.volume = self._node(fields, "Input CT:", ["vtkMRMLScalarVolumeNode"])
        self.arg_model = qt.QComboBox()
        fields.addRow("ARG model:", self.arg_model)
        self.controls.append(self.arg_model)
        self.arg_button = self._button(fields, "Run ARG inference", self.run_arg, "RunARGButton")

    def _setup_cleanup(self, layout) -> None:
        fields = self._group(layout, "3. Component filtering and manual trimming")
        self.arg_segment = slicer.qMRMLSegmentSelectorWidget()
        self.arg_segment.setMRMLScene(slicer.mrmlScene)
        self.arg_segment.noneEnabled = True
        fields.addRow("ARG vessel segment:", self.arg_segment)
        self.minimum_mass = qt.QDoubleSpinBox()
        self.minimum_mass.minimum = 0
        self.minimum_mass.maximum = 1000000000
        self.minimum_mass.value = 0
        fields.addRow("Minimum volume in mm³:", self.minimum_mass)
        self.component_count = qt.QSpinBox()
        self.component_count.minimum = 0
        self.component_count.maximum = 1000000
        self.component_count.value = 1
        self.component_count.toolTip = (
            "Keep this many largest components. Zero keeps all above the volume limit."
        )
        fields.addRow("Largest components:", self.component_count)
        self._button(
            fields, "Create editable cleanup copy", self.prepare_cleanup, "PrepareCleanupButton"
        )
        self.cleaned_segment = slicer.qMRMLSegmentSelectorWidget()
        self.cleaned_segment.setMRMLScene(slicer.mrmlScene)
        self.cleaned_segment.noneEnabled = True
        fields.addRow("Cleaned vessel segment:", self.cleaned_segment)
        self._button(fields, "Open Segment Editor for trimming", self.open_editor, "TrimARGButton")
        self.controls.extend(
            [self.arg_segment, self.cleaned_segment, self.minimum_mass, self.component_count]
        )

    def _setup_centerlines(self, layout) -> None:
        fields = self._group(layout, "4. Centerline generation and editing")
        self._button(
            fields, "Generate VMTK network curves", self.extract, "GenerateCenterlinesButton"
        )
        self._button(
            fields, "Open Extract Centerline", self.open_extraction, "OpenCenterlinesButton"
        )
        self.create_curve_button = self._button(
            fields, "Create centerline curve", self.create_curve, "CreateCenterlineButton"
        )
        self.centerline = self._node(
            fields, "Edited centerlines:", ["vtkMRMLMarkupsCurveNode", "vtkMRMLMarkupsLineNode"]
        )
        self.centerline.toolTip = "Select the root curve to include all nested branch curves."
        self._button(fields, "Open Markups for editing", self.open_markups, "EditCenterlinesButton")
        note = qt.QLabel(
            "Network curves are approximate. Inspect coverage and edit the curves before ModelO. "
            "Automatic extraction requires SlicerVMTK; manually placed curves work on both hosts."
        )
        note.wordWrap = True
        fields.addRow(note)

    def _setup_model_o(self, layout) -> None:
        fields = self._group(layout, "5. ModelO inference")
        self.model_o_model = qt.QComboBox()
        fields.addRow("ModelO model:", self.model_o_model)
        self.controls.append(self.model_o_model)
        self.model_o_button = self._button(
            fields, "Run ModelO inference", self.run_model_o, "RunModelOButton"
        )

    def connection(self):
        return ConnectionSettings(
            str(self.endpoint.text).strip(), float(self.timeout.value), str(self.token.text)
        )

    def _begin(self, text: str, button, running: str) -> None:
        self.pending = True
        for control in self.controls:
            control.enabled = False
        self.cancel_button.enabled = True
        self.running_button = (button, button.text)
        button.text = running
        self.started = time.monotonic()
        self.endpoint_check = None
        self.liveness.visible = False
        self.activity.visible = True
        self.clock.start()
        self._phase(text)
        self._reveal_feedback()

    def _reveal_feedback(self) -> None:
        """Scroll the module panel so a request started further down shows its feedback."""
        if self.tasks.browser:
            # SlicerWeb's QScrollArea has no ensureWidgetVisible; its panel scrolls the page.
            from js import Object
            from pyodide.ffi import to_js

            options = to_js({"block": "nearest"}, dict_converter=Object.fromEntries)
            self.feedback.element().scrollIntoView(options)
            return
        area = self.feedback.parentWidget()
        while area is not None and not isinstance(area, qt.QScrollArea):
            area = area.parentWidget()
        if area is not None:
            area.ensureWidgetVisible(self.feedback)

    def _end(self, text: str) -> None:
        self.pending = False
        self.clock.stop()
        for control in self.controls:
            control.enabled = True
        self.cancel_button.enabled = False
        if self.running_button is not None:
            button, label = self.running_button
            button.text = label
            self.running_button = None
        self.activity.visible = False
        self.liveness.visible = False
        self.status.text = text

    def _phase(self, text: str) -> None:
        self.activity.setRange(0, 0)
        self.activity.textVisible = False
        self.status.text = text

    def _show_liveness(self) -> None:
        if not self.pending:
            return
        text = f"Elapsed {_duration(time.monotonic() - self.started)}."
        if self.endpoint_check is not None:
            reachable, checked_at = self.endpoint_check
            text += (
                f" Endpoint responded at {checked_at}."
                if reachable
                else f" Endpoint check failed at {checked_at}. The inference outcome is unknown."
            )
        self.liveness.text = text
        self.liveness.visible = True

    def _progress(self, update: RequestProgress) -> None:
        if not self.pending:
            return
        if update.endpoint_reachable is not None:
            self.endpoint_check = (update.endpoint_reachable, time.strftime("%H:%M:%S"))
            self._show_liveness()
        elif update.total_bytes:
            self.activity.setRange(0, 100)
            self.activity.textVisible = True
            self.activity.value = round(100 * update.completed_bytes / update.total_bytes)
            text = (
                f"{update.message}: {_size(update.completed_bytes)} of {_size(update.total_bytes)}"
            )
            if update.bytes_per_second:
                text += f" at {_size(update.bytes_per_second)}/s"
            self.status.text = text
        else:
            self._phase(update.message)

    def _discard(self) -> None:
        self.tasks.cancel()
        self.cancel_button.enabled = False
        self._phase(
            "Result will be discarded. Controls unlock when the server responds "
            "or the request times out."
        )

    def failed(self, error) -> None:
        self._end(str(error))

    def refresh(self) -> None:
        if self.pending:
            return
        try:
            connection = self.connection()
            if sys.platform != "emscripten":
                from slicer.packaging import load_requirements, pip_ensure

                requirements = Path(__file__).parent / "Resources/requirements.txt"
                pip_ensure(load_requirements(requirements), requester="Vessel Ray Segmentation")
            self._begin(
                "Discovering endpoint models...", self.refresh_button, "Refreshing models..."
            )
            self.tasks.discover(connection, self._models_ready, self.failed)
        except Exception as error:
            self.failed(error)

    def _models_ready(self, models) -> None:
        self.models = models
        self.arg_models, self.model_o_models = self.logic.select_models(models)
        for combo, available in [
            (self.arg_model, self.arg_models),
            (self.model_o_model, self.model_o_models),
        ]:
            combo.clear()
            for model in available:
                combo.addItem(f"{model.title} [{model.name}]")
        unsupported = len(models) - len(self.arg_models) - len(self.model_o_models)
        self._end(
            f"Endpoint ready. {len(self.arg_models)} image and "
            f"{len(self.model_o_models)} centerline models. {unsupported} unsupported profiles.",
        )
        qt.QSettings().setValue("VesselRaySegmentation/endpoint", str(self.endpoint.text).strip())

    def _run(self, models, combo, button, running: str) -> None:
        if self.pending or self.tasks.disposed:
            return
        self.tasks.cancelled = False
        self._begin("Preparing CT and inference inputs...", button, running)
        qt.QTimer.singleShot(ACKNOWLEDGE_DELAY_MS, lambda: self._submit(models, combo))

    def _submit(self, models, combo) -> None:
        if self.tasks.disposed:
            return
        try:
            if self.tasks.cancelled:
                self._end("Request discarded before submission.")
                return
            if combo.currentIndex < 0 or combo.currentIndex >= len(models):
                raise ValueError("Refresh the endpoint and select an available model")
            request = self.logic.request(
                self.volume.currentNode(),
                models[combo.currentIndex],
                self.connection(),
                self.centerline.currentNode(),
            )
            self.tasks.infer(request, self._inference_ready, self.failed, self._progress)
        except Exception as error:
            self.failed(error)

    def run_arg(self) -> None:
        self._run(self.arg_models, self.arg_model, self.arg_button, "Running ARG inference...")

    def run_model_o(self) -> None:
        self._run(
            self.model_o_models,
            self.model_o_model,
            self.model_o_button,
            "Running ModelO inference...",
        )

    def _inference_ready(self, result) -> None:
        self._phase("Checking the submitted CT and importing the segmentation...")
        qt.QTimer.singleShot(ACKNOWLEDGE_DELAY_MS, lambda: self._import_result(result))

    def _import_result(self, result) -> None:
        if self.tasks.disposed:
            return
        try:
            if self.tasks.cancelled:
                self._end("Result discarded before import.")
                return
            if not scene.source_is_unchanged(result.request.snapshot):
                raise ValueError(
                    "The submitted CT changed or was removed. Result discarded; run inference again."
                )
            segmentation = self.logic.import_result(result)
            if result.request.input_profile == "image":
                self.arg_segment.setCurrentNode(segmentation)
                self.arg_segment.setCurrentSegmentID(
                    segmentation.GetSegmentation().GetNthSegmentID(0)
                )
            elapsed = _duration(time.monotonic() - self.started)
            self._end(
                f"{result.request.model} finished in {elapsed}. Created {segmentation.GetName()}."
            )
        except Exception as error:
            self.failed(error)

    def prepare_cleanup(self) -> None:
        try:
            output, summary = self.logic.prepare_cleanup(
                self.arg_segment.currentNode(),
                self.arg_segment.currentSegmentID(),
                float(self.minimum_mass.value),
                int(self.component_count.value),
            )
            self.cleaned_segment.setCurrentNode(output)
            self.cleaned_segment.setCurrentSegmentID(output.GetSegmentation().GetNthSegmentID(0))
            self.status.text = (
                f"Kept {summary['selected_components']} of {summary['source_components']} "
            )
            self.status.text += f"components, {summary['selected_volume_mm3']:.1f} mm³. Trim this copy in Segment Editor."
        except Exception as error:
            self.failed(error)

    def open_editor(self) -> None:
        try:
            scene.open_segment_editor(
                self.cleaned_segment.currentNode(),
                self.volume.currentNode(),
                self.cleaned_segment.currentSegmentID(),
            )
        except Exception as error:
            self.failed(error)

    def extract(self) -> None:
        try:
            output, summary = self.logic.extract_centerlines(
                self.cleaned_segment.currentNode(), self.cleaned_segment.currentSegmentID()
            )
            self.centerline.setCurrentNode(output)
            self.status.text = f"Created {summary['curve_count']} network curves. "
            self.status.text += f"{summary['failed_components']} failed and {summary['empty_components']} empty components. Inspect all branches."
        except Exception as error:
            self.failed(error)

    def open_extraction(self) -> None:
        if slicer.app.moduleManager().module("ExtractCenterline") is None:
            self.failed(
                RuntimeError(
                    "Install SlicerVMTK for automatic centerline extraction. "
                    "You can also place native Markups curves manually."
                )
            )
            return
        slicer.util.selectModule("ExtractCenterline")

    def create_curve(self) -> None:
        curve = self.logic.create_centerline()
        self.centerline.setCurrentNode(curve)
        self.status.text = "Created an empty curve. Place and edit at least two points in Markups."

    @staticmethod
    def open_markups() -> None:
        import sys

        if sys.platform == "emscripten":
            from .browser import open_panel

            open_panel("Markups")
        else:
            slicer.util.selectModule("Markups")

    def cleanup(self) -> None:
        slicer.mrmlScene.RemoveObserver(self.scene_observer)
        self.clock.stop()
        self.tasks.dispose()
        self.logic.cleanup()
        self.token.text = ""
        if self.tasks.browser:
            observer, callback = self.token_observer
            observer.disconnect()
            callback.destroy()
