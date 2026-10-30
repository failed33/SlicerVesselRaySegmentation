"""Run desktop HTTP off the UI thread and browser fetch on its async loop."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .monai.client import MonaiClient
from .monai.progress import ReportProgress, RequestProgress, ignore_progress
from .monai.submission import run_inference

if TYPE_CHECKING:
    from .browser import BrowserSession


class TaskController:
    def __init__(self) -> None:
        self.browser = sys.platform == "emscripten"
        self.busy = False
        self.cancelled = False
        self.disposed = False
        self.browser_session: BrowserSession | None = None
        self.pending: asyncio.Future[None] | None = None
        self.messages: queue.Queue[tuple[Any, Exception | None]] = queue.Queue()
        self.updates: queue.Queue[RequestProgress] = queue.Queue()
        self.on_progress: ReportProgress = ignore_progress
        self.callbacks: tuple[Callable[[Any], None], Callable[[Exception], None]] | None = None
        self.timer: Any = None  # PythonQt and SlicerWeb supply the timer at runtime.
        if not self.browser:
            import qt

            self.timer = qt.QTimer()
            self.timer.setInterval(50)
            self.timer.connect("timeout()", self._poll)

    def _start(
        self,
        operation: Callable[[], Any],
        on_done: Callable[[Any], None],
        on_failed: Callable[[Exception], None],
    ) -> None:
        if self.busy or self.disposed:
            raise RuntimeError("An endpoint request is already running")
        while not self.updates.empty():
            self.updates.get_nowait()
        self.busy, self.cancelled = True, False
        self.callbacks = (on_done, on_failed)
        if self.browser:

            async def run():
                try:
                    self._deliver(await operation(), None)
                except Exception as error:
                    self._deliver(None, error)

            self.pending = asyncio.ensure_future(run())
            return

        def run():
            try:
                self.messages.put((operation(), None))
            except Exception as error:
                self.messages.put((None, error))

        self.timer.start()
        threading.Thread(target=run, name="vessel-inference", daemon=True).start()

    def _poll(self) -> None:
        # Yield the GIL, as Slicer's SimpleFilters does for Python workers.
        time.sleep(0.001)
        for _ in range(16):
            try:
                self._progress(self.updates.get_nowait())
            except queue.Empty:
                break
        try:
            result, error = self.messages.get_nowait()
        except queue.Empty:
            return
        self.timer.stop()
        while not self.updates.empty():
            self._progress(self.updates.get_nowait())
        self._deliver(result, error)

    def _progress(self, update: RequestProgress) -> None:
        if not self.cancelled and not self.disposed:
            self.on_progress(update)

    def _report(self, update: RequestProgress) -> None:
        if self.browser:
            self._progress(update)
        else:
            self.updates.put(update)

    def _deliver(self, result, error) -> None:
        self.busy = False
        self.on_progress = ignore_progress
        if self.disposed or self.callbacks is None:
            return
        done, failed = self.callbacks
        if self.cancelled:
            failed(RuntimeError("Result discarded. The server may still be finishing its request."))
        elif error:
            failed(error)
        else:
            done(result)

    def _client(self, connection):
        if self.browser:
            from .browser import BrowserClient

            return BrowserClient(connection)
        return MonaiClient(connection)

    def discover(self, connection, on_done, on_failed) -> None:
        self._start(self._client(connection).models, on_done, on_failed)

    def check_endpoint(self, connection, on_done, on_failed) -> None:
        connection = replace(connection, timeout_seconds=min(5, connection.timeout_seconds))
        self._start(self._client(connection).info, on_done, on_failed)

    def infer(self, request, on_done, on_failed, on_progress=ignore_progress) -> None:
        self.on_progress = on_progress
        if self.browser:
            from .browser import BrowserSession

            current = self.browser_session
            if current and (
                current.snapshot is not request.snapshot
                or current.client.connection != request.connection
            ):
                asyncio.ensure_future(current.close())
                current = None
            self.browser_session = current or BrowserSession(request.connection, request.snapshot)
            session = self.browser_session
            operation = lambda: session.infer(request, self._report)
        else:
            operation = lambda: run_inference(request, self._report)
        self._start(operation, on_done, on_failed)

    def cancel(self) -> None:
        # A client disconnect does not prove synchronous server work has stopped.
        self.cancelled = True

    def release_session(self) -> None:
        if self.browser_session:
            asyncio.ensure_future(self.browser_session.close())
            self.browser_session = None

    def dispose(self) -> None:
        self.disposed = True
        self.cancel()
        self.callbacks = None
        if self.timer:
            self.timer.stop()
            self.timer.disconnect("timeout()", self._poll)
        self.release_session()
