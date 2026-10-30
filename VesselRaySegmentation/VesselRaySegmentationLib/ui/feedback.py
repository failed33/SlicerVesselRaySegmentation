"""Present request activity and measured transfer progress."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta

import qt

from ..monai.progress import RequestPhase, RequestProgress

PHASES = {
    RequestPhase.VALIDATING: "Checking endpoint",
    RequestPhase.PREPARING: "Preparing input",
    RequestPhase.CHECKING_SESSION: "Checking uploaded input",
    RequestPhase.SESSION_READY: "Input available on the endpoint",
    RequestPhase.UPLOADING: "Uploading input",
    RequestPhase.WAITING: "Waiting for response",
    RequestPhase.RECEIVING: "Receiving response",
    RequestPhase.READING: "Reading response",
}
MIB = 2**20


class RequestFeedback:
    """Status line, activity bar, elapsed time and the discard action.

    Transports report phases and byte counts only. Render measured transfer
    progress or an indeterminate activity indicator. The owner supplies endpoint
    check results independently of how the request is executed.
    """

    def __init__(self, discard: Callable[[], None]) -> None:
        self.discard = discard
        self.started: float | None = None
        self.last_check: tuple[bool, datetime] | None = None
        self.widget = qt.QWidget()
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status = self._label("")
        layout.addWidget(self.status)
        self.progress = qt.QProgressBar()
        self.progress.visible = False
        self.progress.toolTip = "Shows request activity and measured upload progress."
        layout.addWidget(self.progress)
        self.detail = self._label("")
        self.detail.visible = False
        layout.addWidget(self.detail)
        self.discard_button = qt.QPushButton("Discard pending result")
        self.discard_button.enabled = False
        self.discard_button.connect("clicked()", self._discard)
        layout.addWidget(self.discard_button)
        self.clock = qt.QTimer()
        self.clock.setInterval(1000)
        self.clock.connect("timeout()", self._tick)

    @staticmethod
    def _label(text: str):
        label = qt.QLabel(text)
        label.setTextFormat(qt.Qt.PlainText)
        label.wordWrap = True
        return label

    def begin(self, text: str) -> None:
        self.started = time.monotonic()
        self.last_check = None
        self.discard_button.enabled = True
        self.detail.visible = False
        self.progress.visible = True
        self.clock.start()
        self.step(text)
        self._reveal()

    def step(self, text: str) -> None:
        """Show unmeasured work, such as local preparation or result import."""
        self.progress.setRange(0, 0)
        self.progress.textVisible = False
        self.status.text = text

    def show(self, update: RequestProgress) -> None:
        if self.started is None:
            return
        if not update.total_bytes:
            self.step(PHASES[update.phase])
            return
        self.progress.setRange(0, 100)
        self.progress.textVisible = True
        self.progress.value = round(100 * update.completed_bytes / update.total_bytes)
        text = (
            f"{PHASES[update.phase]}: {update.completed_bytes / MIB:.1f} "
            f"of {update.total_bytes / MIB:.1f} MiB"
        )
        if update.bytes_per_second:
            text += f" at {update.bytes_per_second / MIB:.1f} MiB/s"
        self.status.text = text

    def discarding(self, text: str) -> None:
        """Keep the bar running: the request still holds the controls until it settles."""
        self.discard_button.enabled = False
        self.step(text)

    def finish(self, text: str) -> None:
        self.started = None
        self.clock.stop()
        self.discard_button.enabled = False
        self.progress.visible = False
        self.detail.visible = False
        self.status.text = text

    def notify(self, text: str) -> None:
        self.status.text = text

    def elapsed(self) -> timedelta:
        if self.started is None:
            return timedelta()
        return timedelta(seconds=round(time.monotonic() - self.started))

    def stop(self) -> None:
        self.clock.stop()

    def _discard(self) -> None:
        self.discard()
        self.discarding(
            "Discarding the result. Controls unlock when the endpoint responds "
            "or the request times out."
        )

    def _tick(self) -> None:
        text = f"Elapsed {self.elapsed()}."
        if self.last_check is not None:
            reachable, at = self.last_check
            clock = at.strftime("%H:%M:%S")
            text += (
                f" Endpoint responded at {clock}."
                if reachable
                else f" Endpoint check failed at {clock}. The request outcome is unknown."
            )
        self.detail.text = text
        self.detail.visible = True

    def checked(self, reachable: bool) -> None:
        self.last_check = (reachable, datetime.now())
        self._tick()

    def _reveal(self) -> None:
        """Scroll the module panel so a request started further down shows its feedback."""
        if hasattr(self.widget, "element"):
            # SlicerWeb's QScrollArea has no ensureWidgetVisible; its panel scrolls the page.
            from js import Object
            from pyodide.ffi import to_js

            options = to_js({"block": "nearest"}, dict_converter=Object.fromEntries)
            self.widget.element().scrollIntoView(options)
            return
        area = self.widget.parentWidget()
        while area is not None and not isinstance(area, qt.QScrollArea):
            area = area.parentWidget()
        if area is not None:
            area.ensureWidgetVisible(self.widget)
