"""Experimental cumulative previews over one authenticated multipart response."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import shutil
import tempfile
import threading
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path
from typing import Annotated, cast

import anyio
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from label_inference.requests import parameter_errors, request_parameters
from label_inference.routes import InferenceFormRoute, session_image, validate_parameters
from monailabel.config import settings
from monailabel.endpoints.user.auth import RBAC, User
from monailabel.interfaces.utils.app import app_instance
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.types import Send

from monai_service.application import InferenceService
from monai_service.config import ServiceSettings
from monai_service.contracts import (
    ArtifactUnavailable,
    ExecutionCancelled,
    ExecutionError,
    ExecutionTimeout,
    InferencePreview,
    InferenceRequest,
    InferenceResult,
    InvalidRequest,
)

logger = logging.getLogger(__name__)
CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class _Failure:
    status: int
    detail: str


StreamItem = InferencePreview | InferenceResult | _Failure


class _OwnedStreamingResponse(StreamingResponse):
    """Close the producer even when ASGI send fails between yielded chunks."""

    def __init__(
        self, body: AsyncGenerator[bytes, None], boundary: str, image_path: Path | None
    ) -> None:
        super().__init__(
            body,
            media_type=f"multipart/mixed; boundary={boundary}",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            background=BackgroundTask(image_path.unlink, missing_ok=True)
            if image_path is not None
            else None,
        )
        self.owned_body = body
        self.image_path = image_path

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.owned_body.aclose()
            if self.image_path is not None:
                self.image_path.unlink(missing_ok=True)


def _failure(error: Exception) -> _Failure:
    if isinstance(error, ExecutionTimeout):
        return _Failure(504, str(error))
    if isinstance(error, InvalidRequest):
        return _Failure(422, str(error))
    if isinstance(error, (ArtifactUnavailable, ExecutionError)):
        return _Failure(502, str(error))
    logger.exception("Unexpected streaming inference failure")
    return _Failure(502, "Inference stream failed")


class _StreamRunner:
    def __init__(self, service: InferenceService, request: InferenceRequest) -> None:
        self.service = service
        self.request = request
        self.cancelled = threading.Event()
        self.items: queue.Queue[StreamItem] = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self.run, name="inference-stream", daemon=True)

    def enqueue(self, item: StreamItem) -> None:
        while not self.cancelled.is_set():
            try:
                self.items.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
        raise ExecutionCancelled("Inference stream was cancelled")

    def run(self) -> None:
        result: InferenceResult | None = None
        try:
            result = self.service.stream(self.request, self.cancelled, self.enqueue)
            self.enqueue(result)
            result = None
        except ExecutionCancelled:
            pass
        except Exception as error:
            try:
                self.enqueue(_failure(error))
            except ExecutionCancelled:
                pass
        finally:
            if result is not None:
                result.path.unlink(missing_ok=True)

    def discard(self) -> None:
        while not self.items.empty():
            item = self.items.get_nowait()
            if isinstance(item, InferencePreview):
                item.release()
            elif isinstance(item, InferenceResult):
                item.path.unlink(missing_ok=True)


def _part_header(boundary: str, content_type: str, size: int, filename: str | None = None) -> bytes:
    headers = EmailMessage(policy=SMTP)
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(size)
    if filename:
        headers.add_header("Content-Disposition", "attachment", filename=filename)
    return f"--{boundary}\r\n".encode("ascii") + headers.as_bytes()


def _json_part(boundary: str, value: dict[str, object]) -> bytes:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _part_header(boundary, "application/json", len(payload)) + payload + b"\r\n"


async def _image_parts(boundary: str, path: Path) -> AsyncGenerator[bytes, None]:
    size = path.stat().st_size
    filename = "segmentation" + "".join(path.suffixes)
    yield _part_header(boundary, "application/octet-stream", size, filename)
    with path.open("rb") as source:
        remaining = size
        while remaining:
            chunk = await run_in_threadpool(source.read, min(CHUNK_BYTES, remaining))
            if not chunk:
                raise ExecutionError("Inference image was truncated during transmission")
            remaining -= len(chunk)
            yield chunk
    yield b"\r\n"


async def _inference_parts(
    boundary: str, item: InferencePreview | InferenceResult
) -> AsyncGenerator[bytes, None]:
    try:
        if isinstance(item, InferencePreview):
            yield _json_part(boundary, {"event": "progress", **item.metadata})
            yield _json_part(boundary, {"event": "preview", **item.metadata})
        else:
            yield _json_part(boundary, {"event": "result", "metadata": dict(item.metadata)})
        async with aclosing(_image_parts(boundary, item.path)) as image:
            async for chunk in image:
                yield chunk
        if isinstance(item, InferenceResult):
            yield _json_part(boundary, {"event": "complete"})
    finally:
        if isinstance(item, InferencePreview):
            item.release()
        else:
            item.path.unlink(missing_ok=True)


async def _next_item(runner: _StreamRunner, request: Request) -> StreamItem:
    while True:
        if await request.is_disconnected():
            raise ExecutionCancelled("Inference stream was cancelled")
        try:
            return runner.items.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)


async def _stream_body(
    runner: _StreamRunner, request: Request, boundary: str, image_path: Path | None
) -> AsyncGenerator[bytes, None]:
    runner.thread.start()
    try:
        while True:
            item = await _next_item(runner, request)
            if isinstance(item, _Failure):
                yield _json_part(
                    boundary, {"event": "error", "status": item.status, "detail": item.detail}
                )
                break
            async with aclosing(_inference_parts(boundary, item)) as parts:
                async for chunk in parts:
                    yield chunk
            if isinstance(item, InferenceResult):
                break
        yield f"--{boundary}--\r\n".encode("ascii")
    except ExecutionCancelled:
        pass
    except (ExecutionError, OSError) as error:
        yield _json_part(boundary, {"event": "error", "status": 502, "detail": str(error)})
        yield f"--{boundary}--\r\n".encode("ascii")
    finally:
        runner.cancelled.set()
        # The request file and workspace remain alive until isolated execution
        # and its process-group/container cleanup have actually finished.
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(runner.thread.join)
        runner.discard()
        if image_path is not None:
            image_path.unlink(missing_ok=True)


def _service(model: str) -> InferenceService:
    instance = app_instance()
    service_settings = cast(ServiceSettings, instance.conf["service_settings"])
    deployment = service_settings.registrations.get(model)
    if deployment is None:
        raise HTTPException(status_code=404, detail=f"Unknown inference model: {model}")
    service = InferenceService(deployment, service_settings.work_dir / "requests")
    if not service.supports_streaming:
        raise HTTPException(status_code=501, detail="Streaming is unavailable for this deployment")
    return service


def _copy_upload(upload: UploadFile) -> Path:
    suffix = "".join(Path(upload.filename or "ct.nii.gz").suffixes)
    with tempfile.NamedTemporaryFile(prefix="stream-input-", suffix=suffix, delete=False) as output:
        path = Path(output.name)
        try:
            shutil.copyfileobj(upload.file, output)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path


@dataclass(frozen=True)
class _StreamInputs:
    params: str
    file: UploadFile | None
    label: UploadFile | None


def _stream_inputs(
    params: Annotated[str, Form()] = "{}",
    file: Annotated[UploadFile | None, File()] = None,
    label: Annotated[UploadFile | None, File()] = None,
) -> _StreamInputs:
    return _StreamInputs(params, file, label)


async def api_stream_inference(
    request: Request,
    model: str,
    _user: Annotated[User, Depends(RBAC(settings.MONAI_LABEL_AUTH_ROLE_USER))],
    inputs: Annotated[_StreamInputs, Depends(_stream_inputs)],
    session_id: str = "",
) -> StreamingResponse:
    """Use a CT upload or native session to receive cumulative worker previews."""
    validate_parameters(inputs.params)
    service = await run_in_threadpool(_service, model)
    if (
        request.query_params.get("image")
        or inputs.label is not None
        or "output" in request.query_params
    ):
        raise HTTPException(status_code=501, detail="Streaming supports CT uploads or sessions")
    if not session_id and inputs.file is None:
        raise HTTPException(status_code=422, detail="Provide a CT file or session_id")
    values = json.loads(inputs.params or "{}")
    if values.get("save_label"):
        raise HTTPException(status_code=501, detail="Streaming does not save datastore labels")
    try:
        parameters = service.deployment.parse_parameters(request_parameters(values))
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=parameter_errors(error)) from error
    except InvalidRequest as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    owned_image = None
    if session_id:
        image_path = await run_in_threadpool(session_image, app_instance(), session_id)
    else:
        assert inputs.file is not None
        image_path = owned_image = await run_in_threadpool(_copy_upload, inputs.file)
    runner = _StreamRunner(service, InferenceRequest(image_path, parameters))
    boundary = "vessel-" + uuid.uuid4().hex
    return _OwnedStreamingResponse(
        _stream_body(runner, request, boundary, owned_image), boundary, owned_image
    )


def register_streaming_route(app: FastAPI) -> None:
    if getattr(app.state, "inference_stream_route", False):
        return
    router = APIRouter(prefix="/infer", tags=["Infer"], route_class=InferenceFormRoute)
    router.add_api_route(
        "/{model}/stream", api_stream_inference, methods=["POST"], response_model=None
    )
    app.include_router(router)
    app.state.inference_stream_route = True
    app.openapi_schema = None
