"""Native MONAI Label routes with HTTPX uploads and standard MIME response parsing."""

from __future__ import annotations

import json
import math
import ssl
from contextlib import ExitStack
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode, urlsplit

from .progress import ReportProgress, RequestPhase, RequestProgress, ignore_progress, upload_chunks

if TYPE_CHECKING:
    import httpx


class MonaiClientError(RuntimeError):
    """An HTTP transport or response-format failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class InferenceResponse:
    image: bytes = field(repr=False)
    filename: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConnectionSettings:
    endpoint: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 900.0
    bearer_token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("MONAI endpoint must be an HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use a plain endpoint URL and the separate bearer-token field")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("MONAI timeout must be positive and finite")
        if "\r" in self.bearer_token or "\n" in self.bearer_token:
            raise ValueError("Invalid bearer token")


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    title: str
    description: str
    labels: tuple[str, ...]
    input_profile: str | None

    @classmethod
    def from_info(cls, name: str, values: Any) -> ModelDescriptor:
        if not isinstance(values, dict) or not isinstance(name, str):
            raise MonaiClientError("MONAI Label returned invalid model information")
        config = values.get("config") or {}
        if not isinstance(config, dict):
            raise MonaiClientError("MONAI Label returned invalid model configuration")
        profile = config.get("input_profile")
        if profile is not None and not isinstance(profile, str):
            raise MonaiClientError("MONAI Label returned invalid model input-profile metadata")
        labels = values.get("labels") or {}
        if not isinstance(labels, (dict, list, tuple)):
            raise MonaiClientError("MONAI Label returned invalid model labels")
        return cls(
            name=name,
            title=str(values.get("title") or name),
            description=str(values.get("description") or ""),
            labels=tuple(str(label) for label in labels),
            input_profile=profile,
        )


@dataclass(frozen=True)
class SessionInput:
    session_id: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("Provide a nonempty session_id")


@dataclass(frozen=True)
class Upload:
    """Label's upload fields, independent of desktop or browser encoding."""

    path: Path | None = None
    parameters: dict[str, Any] | None = None
    file_field: str = "file"

    @property
    def filename(self) -> str:
        return "ct" + "".join(self.path.suffixes) if self.path is not None else ""

    def parameter_json(self) -> str:
        return json.dumps(self.parameters, allow_nan=False)


def _upload_files(upload: Upload, resources: ExitStack) -> dict[str, Any]:
    files: dict[str, Any] = {}
    if upload.parameters is not None:
        files["params"] = (None, upload.parameter_json(), "application/json")
    if upload.path is not None:
        stream = resources.enter_context(upload.path.open("rb"))
        files[upload.file_field] = (upload.filename, stream, "application/octet-stream")
    return files


def _image_query(image: Path | SessionInput) -> dict[str, str]:
    return {"session_id": image.session_id} if isinstance(image, SessionInput) else {}


def _decode_result(content_type: str, body: bytes) -> InferenceResponse:
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if not message.is_multipart() or message.defects:
        raise MonaiClientError("MONAI Label returned no segmentation multipart response")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name in fields:
            raise MonaiClientError("MONAI Label returned duplicate multipart fields")
        fields[name] = part
    if "image" not in fields or "params" not in fields:
        raise MonaiClientError("MONAI Label response must contain image and params fields")
    try:
        metadata = json.loads(fields["params"].get_payload(decode=True))
    except (TypeError, ValueError) as exc:
        raise MonaiClientError("MONAI Label returned invalid result metadata") from exc
    image = fields["image"].get_payload(decode=True)
    filename = fields["image"].get_filename() or ""
    if not isinstance(metadata, dict) or not image or not filename:
        raise MonaiClientError("MONAI Label returned an incomplete segmentation result")
    return InferenceResponse(image, Path(filename).name, metadata)


def _decode_info(body: bytes) -> dict[str, Any]:
    try:
        info = json.loads(body)
    except ValueError as error:
        raise MonaiClientError("MONAI Label returned invalid discovery information") from error
    if not isinstance(info, dict) or not isinstance(info.get("models"), dict):
        raise MonaiClientError("MONAI Label discovery response has no model map")
    return info


class MonaiClient:
    """Use native Label discovery, image sessions and inference routes."""

    def __init__(
        self, connection: ConnectionSettings, report: ReportProgress = ignore_progress
    ) -> None:
        self.connection = connection
        self.report = report
        self._transport: httpx.Client | None = None

    def _create_transport(self) -> httpx.Client:
        import httpx

        headers = {}
        if self.connection.bearer_token:
            headers["Authorization"] = f"Bearer {self.connection.bearer_token}"
        return httpx.Client(
            headers=headers,
            timeout=self.connection.timeout_seconds,
            follow_redirects=False,
            verify=ssl.create_default_context(),
        )

    def __enter__(self) -> MonaiClient:
        self._transport = self._create_transport()
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def _open(
        self, path: str, method: str = "GET", upload: Upload | None = None
    ) -> tuple[str, bytes]:
        # Browser imports the route types but never loads a socket-based client.
        import httpx

        try:
            with ExitStack() as resources:
                transport = self._transport or resources.enter_context(self._create_transport())
                files = _upload_files(upload, resources) if upload is not None else None
                request = transport.build_request(
                    method, self.connection.endpoint.rstrip("/") + path, files=files
                )
                if upload is not None and upload.path is not None:
                    assert isinstance(request.stream, httpx.SyncByteStream)
                    request = transport.build_request(
                        method,
                        request.url,
                        headers=request.headers,
                        content=upload_chunks(
                            request.stream,
                            int(request.headers.get("Content-Length", 0)),
                            self.report,
                        ),
                    )
                response = transport.send(request, stream=True)
                resources.callback(response.close)
                if upload is not None:
                    self.report(RequestProgress(RequestPhase.RECEIVING))
                body = response.read()
            if response.status_code >= 300:
                message = _http_error_message(response.status_code, body[:8192])
                token = self.connection.bearer_token
                raise MonaiClientError(
                    message.replace(token, "[redacted]") if token else message,
                    status_code=response.status_code,
                )
            return response.headers.get("Content-Type", ""), body
        except httpx.TimeoutException as error:
            raise MonaiClientError("MONAI Label request timed out") from error
        except httpx.RequestError as error:
            raise MonaiClientError("Cannot reach MONAI Label: " + str(error)) from error
        except OSError as error:
            raise MonaiClientError("MONAI Label connection failed: " + str(error)) from error

    def info(self) -> dict[str, Any]:
        _content_type, body = self._open("/info/")
        return _decode_info(body)

    def models(self) -> dict[str, ModelDescriptor]:
        """Read native model names and our explicit input-profile convention."""
        return {
            name: ModelDescriptor.from_info(name, values)
            for name, values in self.info()["models"].items()
        }

    def create_session(self, path: Path) -> str:
        _response_type, response = self._open(
            "/session/", "PUT", Upload(path=path, file_field="files")
        )
        try:
            session_id = json.loads(response)["session_id"]
        except (ValueError, KeyError, TypeError) as error:
            raise MonaiClientError("MONAI Label returned invalid session information") from error
        if not isinstance(session_id, str) or not session_id:
            raise MonaiClientError("MONAI Label returned invalid session information")
        self.report(RequestProgress(RequestPhase.SESSION_READY))
        return session_id

    def session_exists(self, session_id: str) -> bool:
        try:
            self._open(f"/session/{quote(session_id, safe='')}?update_ts=true")
        except MonaiClientError as error:
            if error.status_code == 404:
                return False
            raise
        return True

    def delete_session(self, session_id: str) -> None:
        try:
            self._open(f"/session/{quote(session_id, safe='')}", "DELETE")
        except MonaiClientError as error:
            if error.status_code != 404:
                raise

    def infer(
        self, path: Path | SessionInput, model: str, parameters: dict[str, Any]
    ) -> InferenceResponse:
        query = {"output": "all", **_image_query(path)}
        route = f"/infer/{quote(model, safe='')}?{urlencode(query)}"
        upload = Upload(None if isinstance(path, SessionInput) else path, parameters)
        self.report(RequestProgress(RequestPhase.WAITING))
        response_type, response_body = self._open(route, "POST", upload)
        self.report(RequestProgress(RequestPhase.READING))
        return _decode_result(response_type, response_body)


def _http_error_message(status: int, body: bytes) -> str:
    detail = ""
    try:
        payload = json.loads(body)
        detail = str(payload.get("detail", "")) if isinstance(payload, dict) else ""
    except (OSError, ValueError):
        pass
    if status == 404:
        prefix = "MONAI Label model or endpoint is unavailable (HTTP 404)"
    elif status in {401, 403}:
        prefix = f"MONAI Label rejected access (HTTP {status})"
    else:
        prefix = f"MONAI Label inference failed (HTTP {status})"
    return f"{prefix}: {detail}" if detail else prefix
