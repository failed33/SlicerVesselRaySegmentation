"""Browser fetch transport for the same native MONAI Label routes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

from .monai.client import (
    ModelDescriptor,
    MonaiClientError,
    Upload,
    _decode_info,
    _decode_result,
)
from .monai.progress import ReportProgress, RequestPhase, RequestProgress, ignore_progress
from .monai.submission import _read_result


def open_panel(name: str) -> None:
    """Built-in web panels have no Python module-manager entry."""
    from slicerweb import host

    host.emit("select-module", {"name": name})


def mask_token_input(widget):
    """SlicerWeb's Qt shim currently ignores QLineEdit.setEchoMode."""
    from js import MutationObserver, Object
    from pyodide.ffi import create_proxy, to_js

    element = widget.element()

    def mask(*_args):
        field = element.querySelector("input")
        if field is not None:
            field.type = "password"
            field.autocomplete = "off"

    callback = create_proxy(mask)
    observer = MutationObserver.new(callback)
    observer.observe(
        element, to_js({"childList": True, "subtree": True}, dict_converter=Object.fromEntries)
    )
    mask()
    return observer, callback


def _form_data(upload: Upload):
    from js import Blob, FormData, Object
    from pyodide.ffi import to_js

    form = FormData.new()
    if upload.parameters is not None:
        form.append("params", upload.parameter_json())
    if upload.path is not None:
        data = to_js(upload.path.read_bytes())
        options = to_js({"type": "application/octet-stream"}, dict_converter=Object.fromEntries)
        form.append(upload.file_field, Blob.new(to_js([data]), options), upload.filename)
    return form


class BrowserClient:
    """Use fetch, because Pyodide cannot open Python sockets.

    The endpoint must allow this application's origin and Authorization header.
    Tokens are passed only in memory; redirects and credential cookies are disabled.
    """

    def __init__(self, connection) -> None:
        self.connection = connection
        self.report: ReportProgress = ignore_progress

    async def request(self, path: str, method: str = "GET", upload=None) -> tuple[str, bytes]:
        from js import AbortController
        from pyodide.http import pyfetch

        headers: dict[str, str] = {}
        options: dict[str, Any] = {"method": method, "redirect": "error", "credentials": "omit"}
        if self.connection.bearer_token:
            headers["Authorization"] = f"Bearer {self.connection.bearer_token}"
        if upload is not None:
            options["body"] = _form_data(upload)
        controller = AbortController.new()
        options.update(headers=headers, signal=controller.signal)

        async def receive():
            response = await pyfetch(self.connection.endpoint.rstrip("/") + path, **options)
            if upload is not None and response.ok:
                self.report(RequestProgress(RequestPhase.RECEIVING))
            return response, await response.bytes()

        try:
            response, body = await asyncio.wait_for(receive(), self.connection.timeout_seconds)
            if not response.ok:
                raise MonaiClientError(
                    f"MONAI Label request failed (HTTP {response.status})",
                    status_code=response.status,
                )
            return response.headers.get("content-type", ""), body
        except TimeoutError as error:
            controller.abort()
            raise MonaiClientError("MONAI Label request timed out") from error
        except MonaiClientError:
            raise
        except Exception as error:
            raise MonaiClientError(
                "Cannot reach MONAI Label. Check the URL, HTTPS and endpoint CORS policy."
            ) from error

    async def info(self) -> dict[str, Any]:
        _content_type, body = await self.request("/info/")
        return _decode_info(body)

    async def models(self) -> dict[str, ModelDescriptor]:
        info = await self.info()
        return {
            name: ModelDescriptor.from_info(name, value) for name, value in info["models"].items()
        }


class BrowserSession:
    """Retain one immutable upload until all active requests have finished."""

    def __init__(self, connection, snapshot) -> None:
        self.client = BrowserClient(connection)
        self.snapshot = snapshot
        self.session_id = None
        self.uncertain = False
        self.closed = False
        self.lock = asyncio.Lock()

    async def ensure(self) -> str:
        if self.session_id:
            try:
                await self.client.request(
                    f"/session/{quote(self.session_id, safe='')}?update_ts=true"
                )
                self.client.report(RequestProgress(RequestPhase.SESSION_READY))
                return self.session_id
            except MonaiClientError as error:
                if error.status_code != 404:
                    raise
                self.session_id = None
        with TemporaryDirectory(prefix="vessel-upload-") as temporary:
            path = Path(temporary) / "ct.nrrd"
            self.client.report(RequestProgress(RequestPhase.PREPARING))
            await asyncio.sleep(0.05)
            self.snapshot.write_image(path)
            self.client.report(RequestProgress(RequestPhase.UPLOADING))
            _kind, body = await self.client.request(
                "/session/", "PUT", Upload(path=path, file_field="files")
            )
        try:
            session_id = json.loads(body)["session_id"]
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("Invalid session ID")
        except (ValueError, KeyError, TypeError) as error:
            raise MonaiClientError("MONAI Label returned invalid session information") from error
        self.session_id = session_id
        self.uncertain = False
        self.client.report(RequestProgress(RequestPhase.SESSION_READY))
        return session_id

    async def infer(self, request, report: ReportProgress = ignore_progress):
        async with self.lock:
            self.client.report = report
            if self.closed:
                raise MonaiClientError("MONAI image session is closed")
            report(RequestProgress(RequestPhase.VALIDATING))
            models = await self.client.models()
            model = models.get(request.model)
            if model is None or model.input_profile != request.input_profile:
                raise MonaiClientError("Model input requirements changed; refresh the endpoint")
            if self.closed:
                raise MonaiClientError("MONAI image session is closed")
            report(RequestProgress(RequestPhase.CHECKING_SESSION))
            session_id = await self.ensure()
            if self.closed:
                raise MonaiClientError("MONAI image session is closed")
            try:
                report(RequestProgress(RequestPhase.WAITING))
                kind, body = await self.client.request(
                    f"/infer/{quote(request.model, safe='')}?output=all&session_id="
                    + quote(session_id, safe=""),
                    "POST",
                    Upload(parameters=request.parameters),
                )
                report(RequestProgress(RequestPhase.READING))
                response = _decode_result(kind, body)
                with TemporaryDirectory(prefix="vessel-result-") as temporary:
                    return _read_result(request, response, Path(temporary))
            except BaseException:
                self.uncertain = True
                raise

    async def close(self) -> None:
        self.closed = True
        async with self.lock:
            session_id, self.session_id = self.session_id, None
            if session_id and not self.uncertain:
                try:
                    await self.client.request(f"/session/{quote(session_id, safe='')}", "DELETE")
                except MonaiClientError:
                    pass  # Native expiry owns cleanup if the endpoint is unavailable.
