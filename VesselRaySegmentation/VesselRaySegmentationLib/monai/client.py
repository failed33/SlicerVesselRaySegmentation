"""MONAI Label's standard HTTP and multipart transport using the Python stdlib."""

from __future__ import annotations

import json
import math
import socket
import ssl
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener


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


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        # Never resend a bearer token or the submitted CT to a redirected host.
        return None


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


def _multipart_upload(
    path: Path | None, parameters: dict[str, Any] | None, *, file_field: str = "file"
) -> tuple[str, int, Iterator[bytes]]:
    boundary = uuid.uuid4().hex
    head = b""
    if parameters is not None:
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="params"\r\n'
            "Content-Type: application/json\r\n\r\n"
            f"{json.dumps(parameters, allow_nan=False)}\r\n"
        ).encode("utf-8")
    if path is not None:
        file_headers = EmailMessage(policy=policy.HTTP)
        file_headers.add_header(
            "Content-Disposition",
            "form-data",
            name=file_field,
            filename="ct" + "".join(path.suffixes),
        )
        file_headers["Content-Type"] = "application/octet-stream"
        head += f"--{boundary}\r\n".encode("ascii") + file_headers.as_bytes()
    tail = f"\r\n--{boundary}--\r\n".encode("ascii")

    def chunks() -> Iterator[bytes]:
        yield head
        if path is not None:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk
        yield tail

    return (
        f"multipart/form-data; boundary={boundary}",
        len(head) + (path.stat().st_size if path is not None else 0) + len(tail),
        chunks(),
    )


def _inference_upload(
    image: Path | SessionInput, parameters: dict[str, Any]
) -> tuple[str, int, Iterator[bytes]]:
    return _multipart_upload(None if isinstance(image, SessionInput) else image, parameters)


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


class MonaiClient:
    """Use native Label discovery, image sessions and inference routes."""

    def __init__(self, connection: ConnectionSettings) -> None:
        self.connection = connection
        self._opener = build_opener(
            _NoRedirect(), HTTPSHandler(context=ssl.create_default_context())
        )

    def _open(self, request: Request) -> tuple[str, bytes]:
        if self.connection.bearer_token:
            request.add_header("Authorization", f"Bearer {self.connection.bearer_token}")
        try:
            with self._opener.open(request, timeout=self.connection.timeout_seconds) as response:
                return response.headers.get("Content-Type", ""), response.read()
        except HTTPError as exc:
            message = _http_error_message(exc)
            token = self.connection.bearer_token
            raise MonaiClientError(
                message.replace(token, "[redacted]") if token else message,
                status_code=exc.code,
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise MonaiClientError("MONAI Label request timed out") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise MonaiClientError("MONAI Label request timed out") from exc
            raise MonaiClientError(f"Cannot reach MONAI Label: {exc.reason}") from exc
        except OSError as exc:
            raise MonaiClientError(f"MONAI Label connection failed: {exc}") from exc

    def info(self) -> dict[str, Any]:
        request = Request(self.connection.endpoint.rstrip("/") + "/info/")
        _content_type, body = self._open(request)
        try:
            info = json.loads(body)
        except ValueError as exc:
            raise MonaiClientError("MONAI Label returned invalid discovery information") from exc
        if not isinstance(info, dict) or not isinstance(info.get("models"), dict):
            raise MonaiClientError("MONAI Label discovery response has no model map")
        return info

    def models(self) -> dict[str, ModelDescriptor]:
        """Read native model names and our explicit input-profile convention."""
        return {
            name: ModelDescriptor.from_info(name, values)
            for name, values in self.info()["models"].items()
        }

    def create_session(self, path: Path) -> str:
        content_type, size, body = _multipart_upload(path, None, file_field="files")
        request = Request(
            self.connection.endpoint.rstrip("/") + "/session/", data=body, method="PUT"
        )
        request.add_header("Content-Type", content_type)
        request.add_header("Content-Length", str(size))
        _response_type, response = self._open(request)
        try:
            session_id = json.loads(response)["session_id"]
        except (ValueError, KeyError, TypeError) as error:
            raise MonaiClientError("MONAI Label returned invalid session information") from error
        if not isinstance(session_id, str) or not session_id:
            raise MonaiClientError("MONAI Label returned invalid session information")
        return session_id

    def session_exists(self, session_id: str) -> bool:
        request = Request(
            self.connection.endpoint.rstrip("/")
            + f"/session/{quote(session_id, safe='')}?update_ts=true"
        )
        try:
            self._open(request)
        except MonaiClientError as error:
            if error.status_code == 404:
                return False
            raise
        return True

    def delete_session(self, session_id: str) -> None:
        request = Request(
            self.connection.endpoint.rstrip("/") + f"/session/{quote(session_id, safe='')}",
            method="DELETE",
        )
        try:
            self._open(request)
        except MonaiClientError as error:
            if error.status_code != 404:
                raise

    def infer(
        self, path: Path | SessionInput, model: str, parameters: dict[str, Any]
    ) -> InferenceResponse:
        content_type, size, body = _inference_upload(path, parameters)
        query = {"output": "all", **_image_query(path)}
        url = (
            self.connection.endpoint.rstrip("/")
            + f"/infer/{quote(model, safe='')}?{urlencode(query)}"
        )
        request = Request(url, data=body, method="POST")
        request.add_header("Content-Type", content_type)
        request.add_header("Content-Length", str(size))
        response_type, response_body = self._open(request)
        return _decode_result(response_type, response_body)


def _http_error_message(error: HTTPError) -> str:
    detail = ""
    try:
        payload = json.loads(error.read(8192))
        detail = str(payload.get("detail", "")) if isinstance(payload, dict) else ""
    except (OSError, ValueError):
        pass
    if error.code == 404:
        prefix = "MONAI Label model or endpoint is unavailable (HTTP 404)"
    elif error.code in {401, 403}:
        prefix = f"MONAI Label rejected access (HTTP {error.code})"
    else:
        prefix = f"MONAI Label inference failed (HTTP {error.code})"
    return f"{prefix}: {detail}" if detail else prefix
