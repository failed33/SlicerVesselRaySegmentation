"""Run desktop HTTP off the UI thread and browser fetch on its async loop."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading

from .monai.client import MonaiClient
from .monai.submission import run_inference


class TaskController:
    def __init__(self) -> None:
        self.browser = sys.platform == "emscripten"
        self.busy = False
        self.cancelled = False
        self.disposed = False
        self.browser_session = None
        self.pending = None
        self.messages = queue.Queue()
        self.timer = None
        if not self.browser:
            import qt

            self.timer = qt.QTimer()
            self.timer.setInterval(50)
            self.timer.connect("timeout()", self._poll)

    def _start(self, operation, on_done, on_failed) -> None:
        if self.busy or self.disposed:
            raise RuntimeError("An endpoint request is already running")
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
        try:
            result, error = self.messages.get_nowait()
        except queue.Empty:
            return
        self.timer.stop()
        self._deliver(result, error)

    def _deliver(self, result, error) -> None:
        self.busy = False
        if self.disposed:
            return
        done, failed = self.callbacks
        if self.cancelled:
            failed(RuntimeError("Result discarded. The server may still be finishing its request."))
        elif error:
            failed(error)
        else:
            done(result)

    def discover(self, connection, on_done, on_failed) -> None:
        if self.browser:
            from .browser import BrowserClient

            operation = BrowserClient(connection).models
        else:
            operation = MonaiClient(connection).models
        self._start(operation, on_done, on_failed)

    def infer(self, request, on_done, on_failed) -> None:
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
            operation = lambda: session.infer(request)
        else:
            operation = lambda: run_inference(request)
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
