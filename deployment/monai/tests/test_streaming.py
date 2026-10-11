"""Real socket checks for progress, final masks and request-lifetime cleanup."""

import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi import Request
from fastapi.testclient import TestClient
from monailabel.config import settings
from monailabel.endpoints.user.auth import User, get_current_user
from monailabel.interfaces.utils.app import clear_cache
from requests_toolbelt.multipart.decoder import MultipartDecoder
from starlette.requests import ClientDisconnect

from monai_service.app import create_app
from monai_service.application import InferenceService
from monai_service.config import ServiceSettings
from monai_service.contracts import ExecutionError, InferenceRequest
from monai_service.schemas.base import NoParameters
from monai_service.streaming import (
    _image_parts,
    _OwnedStreamingResponse,
    _stream_body,
    _StreamRunner,
)
from tests.stream_fixture import fixture_deployment


@pytest.fixture
def volume(tmp_path: Path) -> tuple[bytes, np.ndarray, sitk.Image]:
    values = np.arange(-30, 30, dtype=np.float32).reshape(3, 4, 5)
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.7, 1.3, 2.2))
    image.SetOrigin((12, -15, 60))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    path = tmp_path / "ct.nii.gz"
    sitk.WriteImage(image, str(path))
    return path.read_bytes(), values, image


def _wait_clean(directory: Path) -> None:
    deadline = time.monotonic() + 8
    while directory.exists() and list(directory.iterdir()):
        assert time.monotonic() < deadline, "Request files remain after streaming cleanup"
        time.sleep(0.05)


def _parts(payload: bytes, content_type: str) -> tuple[list[dict], list[bytes]]:
    parts = MultipartDecoder(payload, content_type).parts
    events, images = [], []
    for part in parts:
        assert int(part.headers[b"Content-Length"]) == len(part.content)
        if part.headers[b"Content-Type"] == b"application/json":
            events.append(json.loads(part.content))
        else:
            assert b'filename="segmentation.nii.gz"' in part.headers[b"Content-Disposition"]
            images.append(part.content)
    return events, images


def test_preview_arrives_before_exit_and_final_matches_native_geometry(
    live_service, volume: tuple, tmp_path: Path
) -> None:
    completed = tmp_path / "completed"
    with live_service(
        {"STREAM_TEST_COMPLETED": str(completed)}, app="tests.stream_fixture:app"
    ) as client:
        assert client.get("/info/").json()["models"]["stream-fixture"]["config"]["streaming"]
        with client.stream(
            "POST", "/infer/stream-fixture/stream", files={"file": ("ct.nii.gz", volume[0])}
        ) as response:
            assert response.status_code == 200
            chunks = []
            iterator = response.iter_bytes()
            for chunk in iterator:
                chunks.append(chunk)
                if b'"event":"preview"' in b"".join(chunks):
                    assert not completed.exists(), "Preview only arrived after worker exit"
                    break
            chunks.extend(iterator)
            payload = b"".join(chunks)
        events, images = _parts(payload, response.headers["content-type"])
        assert [event["event"] for event in events] == ["progress", "preview", "result", "complete"]
        assert events[0]["completed"] == 1 and events[0]["total"] == 2
        assert events[2]["metadata"]["runtime"] == "process"
        assert len(images) == 2 and completed.exists()
        for name, payload, expected in (
            ("preview", images[0], volume[1] > 10),
            ("result", images[1], volume[1] > 0),
        ):
            path = tmp_path / f"{name}.nii.gz"
            path.write_bytes(payload)
            image = sitk.ReadImage(str(path))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image), expected)
            assert image.GetSpacing() == pytest.approx(volume[2].GetSpacing())
            assert image.GetOrigin() == pytest.approx(volume[2].GetOrigin())
            assert image.GetDirection() == pytest.approx(volume[2].GetDirection())
        _wait_clean(tmp_path / "service" / "requests")


@pytest.mark.parametrize("mode", ["error", "invalid-preview"])
def test_worker_failures_emit_terminal_error_and_clean_files(
    mode: str, live_service, volume: tuple, tmp_path: Path
) -> None:
    with live_service({"STREAM_TEST_MODE": mode}, app="tests.stream_fixture:app") as client:
        response = client.post(
            "/infer/stream-fixture/stream", files={"file": ("ct.nii.gz", volume[0])}
        )
        events, _ = _parts(response.content, response.headers["content-type"])
        assert events[-1]["event"] == "error"
        assert events[-1]["status"] == 502
        assert not any(event["event"] in {"result", "complete"} for event in events)
        _wait_clean(tmp_path / "service" / "requests")


def test_timeout_emits_terminal_error(live_service, volume: tuple, tmp_path: Path) -> None:
    with live_service(
        {"STREAM_TEST_MODE": "cancel", "STREAM_TEST_TIMEOUT": "1"},
        app="tests.stream_fixture:app",
    ) as client:
        response = client.post(
            "/infer/stream-fixture/stream", files={"file": ("ct.nii.gz", volume[0])}
        )
        events, _ = _parts(response.content, response.headers["content-type"])
        assert events[-1]["event"] == "error" and events[-1]["status"] == 504
        _wait_clean(tmp_path / "service" / "requests")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process cancellation proof")
@pytest.mark.parametrize("source", ["upload", "session"])
def test_client_disconnect_stops_worker_before_deleting_request(
    live_service, volume: tuple, tmp_path: Path, source: str
) -> None:
    started, completed = tmp_path / "started", tmp_path / "completed"
    with live_service(
        {
            "STREAM_TEST_MODE": "cancel",
            "STREAM_TEST_STARTED": str(started),
            "STREAM_TEST_COMPLETED": str(completed),
        },
        app="tests.stream_fixture:app",
    ) as client:
        arguments = {"files": {"file": ("ct.nii.gz", volume[0])}}
        session_id = None
        if source == "session":
            created = client.put("/session/", files={"files": ("ct.nii.gz", volume[0])})
            assert created.status_code == 200
            session_id = created.json()["session_id"]
            arguments = {"params": {"session_id": session_id}}
        with client.stream("POST", "/infer/stream-fixture/stream", **arguments) as response:
            assert response.status_code == 200
            for chunk in response.iter_bytes():
                if b'"event":"preview"' in chunk:
                    break
        _wait_clean(tmp_path / "service" / "requests")
        with pytest.raises(ProcessLookupError):
            os.kill(int(started.read_text()), 0)
        assert not completed.exists()
        if session_id is not None:
            assert client.get(f"/session/{session_id}").status_code == 200
            assert client.delete(f"/session/{session_id}").status_code == 200


def test_stream_checks_auth_identity_and_capability_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "MONAI_LABEL_STUDIES", str(tmp_path / "studies"))
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSION_PATH", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "MONAI_LABEL_AUTH_ENABLE", True)
    deployment = fixture_deployment()
    app = create_app(
        ServiceSettings(
            tmp_path, {"stream": deployment, "direct": replace(deployment, executor=None)}
        )
    )
    try:
        client = TestClient(app, raise_server_exceptions=False)
        try:
            assert client.post("/infer/stream/stream").status_code == 401
            app.dependency_overrides[get_current_user] = lambda: User(username="reader", roles=[])
            assert client.post("/infer/stream/stream").status_code == 403
            app.dependency_overrides[get_current_user] = lambda: User(
                username="user", roles=[settings.MONAI_LABEL_AUTH_ROLE_USER]
            )
            assert client.post("/infer/direct/stream").status_code == 501
            assert client.post("/infer/unknown/stream").status_code == 404
            for parameters, expected in (
                ('{"image":"server-file"}', 422),
                ('{"model":"other"}', 422),
                ("[]", 422),
                ("invalid", 400),
            ):
                assert (
                    client.post("/infer/stream/stream", data={"params": parameters}).status_code
                    == expected
                )
            assert client.post("/infer/stream/stream?session_id=anything").status_code == 404
            assert not (tmp_path / "requests").exists()
        finally:
            client.close()
    finally:
        app.dependency_overrides.clear()
        clear_cache()


def test_sender_detects_image_truncation_before_finishing_part(tmp_path: Path) -> None:
    path = tmp_path / "image.nii.gz"
    path.write_bytes(b"complete image")

    async def truncate_during_transmission() -> None:
        parts = _image_parts("boundary", path)
        header = await anext(parts)
        assert b"Content-Length: 14" in header
        path.write_bytes(b"")
        with pytest.raises(ExecutionError, match="truncated"):
            await anext(parts)
        await parts.aclose()

    asyncio.run(truncate_during_transmission())


@pytest.mark.parametrize("owned", [True, False])
def test_asgi_send_failure_closes_producer_and_stops_worker(
    volume: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned: bool
) -> None:
    started = tmp_path / "started"
    monkeypatch.setenv("STREAM_TEST_MODE", "cancel")
    monkeypatch.setenv("STREAM_TEST_STARTED", str(started))
    image_path = tmp_path / "owned-upload.nii.gz"
    image_path.write_bytes(volume[0])
    service = InferenceService(fixture_deployment(), tmp_path / "requests")
    runner = _StreamRunner(service, InferenceRequest(image_path, NoParameters()))

    async def fail_sending_binary() -> None:
        scope = {"type": "http", "asgi": {"spec_version": "2.4"}}

        async def receive() -> dict:
            await asyncio.sleep(100)
            return {"type": "http.disconnect"}

        binary_header = False

        async def send(message: dict) -> None:
            nonlocal binary_header
            if message["type"] != "http.response.body":
                return
            if binary_header:
                raise OSError("Client disconnected while sending image")
            if b"Content-Type: application/octet-stream" in message.get("body", b""):
                binary_header = True

        response = _OwnedStreamingResponse(
            _stream_body(
                runner, Request(scope, receive), "boundary", image_path if owned else None
            ),
            "boundary",
            image_path if owned else None,
        )
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)

    asyncio.run(fail_sending_binary())
    assert runner.cancelled.is_set() and not runner.thread.is_alive()
    assert image_path.exists() is (not owned)
    assert not list(service.work_dir.iterdir())
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(started.read_text()), 0)
