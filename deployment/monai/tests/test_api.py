"""Checks against MONAI Label's real routes, dependencies and response format."""

import asyncio
import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx
import numpy as np
import pytest
import SimpleITK as sitk
from fastapi import BackgroundTasks, FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from model_artifacts import SelectedModel
from monailabel.app import app as upstream_app
from monailabel.config import settings
from monailabel.endpoints.user.auth import User, get_current_user
from monailabel.interfaces.utils.app import app_instance, clear_cache
from pydantic import ValidationError
from requests_toolbelt.multipart.decoder import MultipartDecoder

from monai_service.adapters import mlflow as model_adapter
from monai_service.adapters.mlflow import (
    MLflowArtifactSource,
    ModelSource,
)
from monai_service.app import create_app
from monai_service.config import ServiceSettings
from monai_service.contracts import (
    ExecutionError,
    ExecutionResult,
    ExecutionTimeout,
    Executor,
    InferenceResult,
    InferenceWorkflow,
    InvalidRequest,
    Invocation,
    ModelDeployment,
    Parameters,
)
from monai_service.models.arg import ArgWorkflow
from monai_service.models.model_o import (
    ModelOParameters,
    ModelOSettings,
    ModelOWorkflow,
)
from monai_service.schemas.base import NoParameters
from tests.fixtures import create_test_app


def _route_inventory(app: FastAPI) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in iter_route_contexts(app.routes)
        for method in (route.methods or ())
    }


UPSTREAM_ROUTES = _route_inventory(upstream_app)
UPSTREAM_INFERENCE_SCHEMA = upstream_app.openapi()["paths"]["/infer/{model}"]
UPSTREAM_DEPENDENCIES = {
    (tuple(sorted(route.methods)), route.path): tuple(
        dependency.call for dependency in route.dependant.dependencies
    )
    for route in iter_route_contexts(upstream_app.routes)
    if isinstance(route.original_route, APIRoute)
}


def _authorized_user() -> User:
    return User(
        username="test-user",
        roles=[
            settings.MONAI_LABEL_AUTH_ROLE_ADMIN,
            settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR,
            settings.MONAI_LABEL_AUTH_ROLE_USER,
        ],
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "MONAI_LABEL_STUDIES", str(tmp_path / "studies"))
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSION_PATH", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "MONAI_LABEL_AUTH_ENABLE", True)
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSIONS", True)
    monkeypatch.setattr(settings, "MONAI_LABEL_DATASTORE_AUTO_RELOAD", False)
    app = create_test_app(ServiceSettings(work_dir=tmp_path))
    app.dependency_overrides[get_current_user] = _authorized_user
    # Upstream's process-wide startup scheduler has no shutdown hook. These
    # HTTP tests use its normal app loader without starting a background timer.
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    test_client.close()
    app.dependency_overrides.clear()
    clear_cache()


@pytest.fixture
def volume(tmp_path: Path) -> tuple[bytes, np.ndarray, sitk.Image]:
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5) - 28
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 1.1, 2.3))
    image.SetOrigin((20.0, -40.0, 60.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "ct.nii.gz"
    sitk.WriteImage(image, str(path))
    return path.read_bytes(), array, image


def _model_o_parameters() -> dict[str, object]:
    return {"polylines_zyx": [[[0.25, 1.0, 1.0], [1.0, 1.0, 1.0]]]}


def _read_result(response: httpx.Response, tmp_path: Path) -> tuple[sitk.Image, dict[str, Any]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("multipart/form-data")
    decoder = MultipartDecoder.from_response(response)
    parts = {
        part.headers[b"Content-Disposition"].split(b'name="')[1].split(b'"')[0]: part
        for part in decoder.parts
    }
    result_path = tmp_path / "result.nii.gz"
    result_path.write_bytes(parts[b"image"].content)
    return sitk.ReadImage(str(result_path)), json.loads(parts[b"params"].content)


def test_routes_and_rbac_dependencies_are_retained(client: TestClient) -> None:
    assert _route_inventory(client.app) == UPSTREAM_ROUTES | {("POST", "/infer/{model}/stream")}
    dependencies = {
        (tuple(sorted(route.methods)), route.path): tuple(
            dependency.call for dependency in route.dependant.dependencies
        )
        for route in iter_route_contexts(client.app.routes)
        if isinstance(route.original_route, APIRoute)
    }
    assert {key: dependencies[key] for key in UPSTREAM_DEPENDENCIES} == UPSTREAM_DEPENDENCIES
    assert ("POST", "/infer/{model}") in UPSTREAM_ROUTES
    assert ("PUT", "/session/") in UPSTREAM_ROUTES
    assert ("POST", "/auth/token") in UPSTREAM_ROUTES
    assert ("POST", "/reload") in UPSTREAM_ROUTES
    client.app.openapi_schema = None
    schema_response = client.get("/openapi.json")
    assert schema_response.status_code == 200
    assert schema_response.json()["paths"]["/infer/{model}"] == UPSTREAM_INFERENCE_SCHEMA


def test_production_app_does_not_advertise_test_fixture(client: TestClient, tmp_path: Path) -> None:
    info = client.get("/info/").json()
    assert set(info["models"]) == {"smoke-test"}
    assert info["models"]["smoke-test"]["config"]["input_profile"] == "image"
    assert not info["trainers"] and not info["strategies"] and not info["scoring"]
    create_app(ServiceSettings(work_dir=tmp_path))
    assert client.get("/info/").json()["models"] == {}


def test_native_lifespan_initializes_app(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    timer_starts = []

    def capture_timer_start(_timer: Any, block: bool = False) -> None:
        timer_starts.append(block)

    # Retain the real native startup hook while isolating its process-wide timer.
    import schedule

    monkeypatch.setattr("monailabel.interfaces.app.Timeloop.start", capture_timer_start)
    monkeypatch.setattr("monailabel.interfaces.app.schedule", schedule.Scheduler())
    with TestClient(client.app) as started_client:
        assert app_instance()._server_mode is True
        assert started_client.get("/info/").status_code == 200
        assert timer_starts == [False]


DEFERRED_REQUESTS = [
    ("GET", "/train/"),
    ("POST", "/train/"),
    ("POST", "/train/unused"),
    ("DELETE", "/train/"),
    ("GET", "/scoring/"),
    ("POST", "/scoring/"),
    ("POST", "/scoring/unused"),
    ("DELETE", "/scoring/"),
    ("POST", "/activelearning/random"),
    ("GET", "/batch/infer"),
    ("POST", "/batch/infer/unused"),
    ("DELETE", "/batch/infer"),
    ("GET", "/model/unused"),
    ("GET", "/model/info/unused"),
    ("PUT", "/model/unused"),
    ("DELETE", "/model/unused"),
    ("POST", "/infer/wsi/unused"),
    ("POST", "/infer/wsi_v2/unused"),
]


@pytest.mark.parametrize(("method", "path"), DEFERRED_REQUESTS)
def test_deferred_capabilities_are_501(client: TestClient, method: str, path: str) -> None:
    kwargs = {"files": {"file": ("model.pt", b"test")}} if method == "PUT" else {}
    response = client.request(method, path, **kwargs)
    assert response.status_code == 501, response.text
    assert response.json() == {"detail": "Not yet implemented"}


@pytest.mark.parametrize(("method", "path"), DEFERRED_REQUESTS)
def test_deferred_capabilities_authenticate_first(
    client: TestClient, method: str, path: str
) -> None:
    client.app.dependency_overrides.pop(get_current_user)
    kwargs = {"files": {"file": ("model.pt", b"test")}} if method == "PUT" else {}
    response = client.request(method, path, **kwargs)
    assert response.status_code == 401, response.text
    assert response.headers["www-authenticate"] == "Bearer"


def test_deferred_capabilities_check_roles(client: TestClient) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: User(username="reader", roles=[])
    assert client.post("/train/unused").status_code == 403
    assert client.put("/model/unused", files={"file": ("model.pt", b"test")}).status_code == 403
    assert client.post("/infer/smoke-test").status_code == 403
    assert client.get("/info/").status_code == 403


def test_direct_upload_uses_native_multipart_response(
    client: TestClient, volume: tuple[bytes, np.ndarray, sitk.Image], tmp_path: Path
) -> None:
    payload, array, reference = volume
    response = client.post(
        "/infer/smoke-test",
        params={"image": "outside-not-registered"},
        data={"params": "{}"},
        files={"file": ("ct.nii.gz", payload)},
    )
    result, metadata = _read_result(response, tmp_path)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), array > 0)
    assert result.GetOrigin() == pytest.approx(reference.GetOrigin(), abs=1e-4)
    assert result.GetSpacing() == pytest.approx(reference.GetSpacing(), abs=1e-6)
    assert result.GetDirection() == pytest.approx(reference.GetDirection(), abs=1e-6)
    assert metadata["test_only"] is True
    assert metadata["worker"] == "smoke"
    assert not list((tmp_path / "requests").iterdir())


@pytest.mark.parametrize("output", ["image", "json", "all"])
def test_supported_native_output_formats_are_retained(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    output: str,
) -> None:
    response = client.post(
        "/infer/smoke-test",
        params={"output": output},
        data={
            "params": json.dumps(
                {"save_label": False, "label_tag": "original", "logging": False, "timeout": 20}
            )
        },
        files={"file": ("ct.nii.gz", volume[0])},
    )
    assert response.status_code == 200, response.text
    if output == "json":
        assert response.json()["test_only"] is True
    else:
        if output == "all":
            result, _ = _read_result(response, tmp_path)
        else:
            result_path = tmp_path / "result.nii.gz"
            result_path.write_bytes(response.content)
            result = sitk.ReadImage(str(result_path))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(result), volume[1] > 0)
    assert not list((tmp_path / "requests").iterdir())


@pytest.mark.parametrize("source", ["upload", "session"])
@pytest.mark.parametrize("output", ["image", "all"])
@pytest.mark.parametrize("outcome", ["success", "start-failure", "body-failure", "cancel"])
def test_native_response_cleans_owned_files_after_every_send_outcome(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    output: str,
    outcome: str,
) -> None:
    query = {"output": output}
    arguments = {"files": {"file": ("ct.nii.gz", volume[0])}}
    session_image = None
    if source == "session":
        created = client.put("/session/", files={"files": ("ct.nii.gz", volume[0])})
        assert created.status_code == 200
        query["session_id"] = created.json()["session_id"]
        session_image = Path(created.json()["session_info"]["image"])
        arguments = {}

    cleanup_calls = []
    native_cleanup = BackgroundTasks.__call__

    async def record_cleanup(tasks: BackgroundTasks) -> None:
        cleanup_calls.append(tuple(Path(task.args[0]) for task in tasks.tasks))
        await native_cleanup(tasks)

    monkeypatch.setattr(BackgroundTasks, "__call__", record_cleanup)
    request = httpx.Request("POST", "http://test/infer/smoke-test", params=query, **arguments)
    body = request.read()
    messages = []

    async def transmit() -> None:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/infer/smoke-test",
            "raw_path": b"/infer/smoke-test",
            "query_string": request.url.query,
            "root_path": "",
            "headers": [(key.lower(), value) for key, value in request.headers.raw],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        }
        delivered = False

        async def receive() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await anyio.sleep_forever()

        with anyio.CancelScope() as cancellation:

            async def send(message: dict) -> None:
                messages.append(message)
                if message["type"] == "http.response.start":
                    assert message["status"] == 200
                    if outcome == "start-failure":
                        raise OSError("Client disconnected before the body")
                elif message["type"] == "http.response.body":
                    if outcome == "body-failure":
                        raise OSError("Client disconnected during the body")
                    if outcome == "cancel":
                        cancellation.cancel()
                        await anyio.lowlevel.checkpoint()

            if outcome.endswith("failure"):
                with pytest.raises(OSError, match="Client disconnected"):
                    await client.app(scope, receive, send)
            else:
                await client.app(scope, receive, send)

    asyncio.run(transmit())
    assert len(cleanup_calls) == 1
    assert len(cleanup_calls[0]) == (2 if source == "upload" else 1)
    assert all(not path.exists() for path in cleanup_calls[0])
    assert not list((tmp_path / "requests").iterdir())
    if session_image is not None:
        assert session_image.read_bytes() == volume[0]
        assert client.get(f"/session/{query['session_id']}").status_code == 200
    if outcome == "success":
        response = httpx.Response(
            200,
            headers=messages[0]["headers"],
            content=b"".join(message.get("body", b"") for message in messages[1:]),
        )
        if output == "all":
            result, metadata = _read_result(response, tmp_path)
            assert metadata["test_only"] is True
        else:
            path = tmp_path / "result.nii.gz"
            path.write_bytes(response.content)
            result = sitk.ReadImage(str(path))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(result), volume[1] > 0)


def test_session_and_datastore_flow(
    client: TestClient, volume: tuple[bytes, np.ndarray, sitk.Image], tmp_path: Path
) -> None:
    payload, array, _ = volume
    response = client.put("/session/", files={"files": ("ct.nii.gz", payload)})
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]
    assert client.get(f"/session/{session_id}").status_code == 200
    result, _ = _read_result(
        client.post(
            "/infer/smoke-test",
            params={"session_id": session_id, "image": "outside-not-registered"},
        ),
        tmp_path,
    )
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), array > 0)
    assert client.delete(f"/session/{session_id}").status_code == 200
    assert client.get(f"/session/{session_id}").status_code == 404

    uploaded = client.put("/datastore/image?image=ct", files={"file": ("ct.nii.gz", payload)})
    assert uploaded.status_code == 200, uploaded.text
    result, _ = _read_result(client.post("/infer/smoke-test?image=ct"), tmp_path)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), array > 0)
    assert client.get("/datastore/").status_code == 200


@pytest.mark.parametrize("image_kind", ["unknown", "absolute", "relative"])
def test_explicit_image_must_be_a_datastore_entry(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_kind: str,
) -> None:
    outside_ct = tmp_path / "ct.nii.gz"
    assert outside_ct.is_file()
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    image_id = {
        "unknown": "absent",
        "absolute": str(outside_ct),
        "relative": "../ct.nii.gz",
    }[image_kind]
    worker_calls = []
    monkeypatch.setattr(
        "monai_service.application.InferenceService.infer",
        lambda *args, **kwargs: worker_calls.append(args),
    )

    response = client.post("/infer/smoke-test", params={"image": image_id})

    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Datastore image not found"}
    assert not worker_calls


def test_registered_image_cannot_shortcut_to_a_cwd_file(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded = client.put("/datastore/image?image=ct", files={"file": ("ct.nii.gz", volume[0])})
    assert uploaded.status_code == 200, uploaded.text
    assert app_instance().datastore().get_image_uri("ct")
    (tmp_path / "ct").write_bytes(volume[0])
    monkeypatch.chdir(tmp_path)
    worker_calls = []
    monkeypatch.setattr(
        "monai_service.application.InferenceService.infer",
        lambda *args, **kwargs: worker_calls.append(args),
    )

    response = client.post("/infer/smoke-test", params={"image": "ct"})

    assert response.status_code == 422, response.text
    assert response.json() == {"detail": "Image identifier collides with a server filesystem path"}
    assert not worker_calls


@pytest.mark.parametrize("image_id", ["collection/ct", "dataset:ct"])
def test_configured_datastore_identifiers_remain_opaque(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_id: str,
) -> None:
    datastore = app_instance().datastore()
    lookups = []

    def lookup(identifier: str) -> str:
        lookups.append(identifier)
        return str(tmp_path / "ct.nii.gz") if identifier == image_id else ""

    monkeypatch.setattr(datastore, "get_image_uri", lookup)
    monkeypatch.setattr(
        datastore, "datalist", lambda *args, **kwargs: pytest.fail("Unexpected datastore scan")
    )

    result, _ = _read_result(client.post("/infer/smoke-test", params={"image": image_id}), tmp_path)

    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), volume[1] > 0)
    assert lookups == [image_id, image_id]


@pytest.mark.parametrize(
    ("path", "params", "expected"),
    [
        ("/infer/unknown", "{}", 404),
        ("/infer/smoke-test", "{broken", 400),
        ("/infer/smoke-test", "[]", 422),
        ("/infer/smoke-test", "null", 422),
        ("/infer/smoke-test", '"secret-value"', 422),
        ("/infer/smoke-test", '{"model": "other"}', 422),
        ("/infer/smoke-test", '{"image": "/other-volume.nii.gz"}', 422),
        ("/infer/smoke-test", '{"session": {}}', 422),
        ("/infer/smoke-test", '{"session_id": "other"}', 422),
        ("/infer/smoke-test", "{}", 422),
        ("/infer/smoke-test?session_id=absent", "{}", 404),
        ("/infer/smoke-test?output=invalid", "{}", 422),
        ("/infer/unknown?output=dicom_seg", "{}", 404),
        ("/infer/smoke-test?output=dicom_seg", "{}", 422),
        ("/infer/smoke-test?output=dicom_seg&session_id=absent", "{}", 404),
    ],
)
def test_inference_validation_is_distinct_from_501(
    client: TestClient, path: str, params: str, expected: int
) -> None:
    response = client.post(path, data={"params": params})
    assert response.status_code == expected, response.text
    assert response.json()["detail"] != "Not yet implemented"


def test_inference_session_requires_a_session_store(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_instance(), "sessions", lambda: None)
    response = client.post("/infer/smoke-test", params={"session_id": "absent"})
    assert response.status_code == 404, response.text
    assert response.json() == {"detail": "Session not found"}


@pytest.mark.parametrize(
    ("error_type", "expected"), [(ExecutionError, 502), (ExecutionTimeout, 504)]
)
def test_worker_errors_and_upload_cleanup(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    expected: int,
) -> None:
    uploaded_paths = []

    def failed_worker(_service: Any, request: Any) -> Any:
        assert isinstance(request.parameters, NoParameters)
        uploaded_paths.append(request.image)
        raise error_type("worker failed for test")

    monkeypatch.setattr("monai_service.application.InferenceService.infer", failed_worker)
    response = client.post("/infer/smoke-test", files={"file": ("ct.nii.gz", volume[0])})
    assert response.status_code == expected, response.text
    assert response.json() == {"detail": "worker failed for test"}
    assert uploaded_paths and all(not path.exists() for path in uploaded_paths)


def test_blocking_inference_does_not_block_event_loop(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def delayed_worker(_service: Any, request: Any) -> InferenceResult:
        started.set()
        if not release.wait(5):
            raise ExecutionTimeout("test did not release worker")
        result = tmp_path / "delayed.nii.gz"
        result.write_bytes(request.image.read_bytes())
        return InferenceResult(result, {"test_only": True})

    monkeypatch.setattr("monai_service.application.InferenceService.infer", delayed_worker)

    async def check_responsiveness() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app), base_url="http://test"
        ) as http:
            infer = asyncio.create_task(
                http.post("/infer/smoke-test", files={"file": ("ct.nii.gz", volume[0])})
            )
            try:
                assert await asyncio.to_thread(started.wait, 2)
                info = await asyncio.wait_for(http.get("/info/"), timeout=1)
                assert info.status_code == 200
                assert not infer.done()
            finally:
                release.set()
            assert (await infer).status_code == 200

    asyncio.run(check_responsiveness())


def _deployment(
    tmp_path: Path,
    workflow: InferenceWorkflow[Parameters],
    parse_parameters: Callable[[object], Parameters],
    executor: Executor | None = None,
) -> ModelDeployment[Parameters]:
    class NoExecution:
        def run(self, invocation: Invocation) -> ExecutionResult:
            raise AssertionError("This request must not execute a workload")

    return ModelDeployment(
        workflow,
        MLflowArtifactSource(
            ModelSource("ConfiguredModel", "accepted", "http://mlflow"), tmp_path / "models"
        ),
        executor or NoExecution(),
        parse_parameters,
    )


def test_registration_descriptors_are_advertised(client: TestClient, tmp_path: Path) -> None:
    create_app(
        ServiceSettings(
            tmp_path,
            {
                "arbitrary-image-id": _deployment(
                    tmp_path, ArgWorkflow(), NoParameters.model_validate
                ),
                "arbitrary-polyline-id": _deployment(
                    tmp_path,
                    ModelOWorkflow(settings=ModelOSettings(assets=tmp_path)),
                    ModelOParameters.model_validate,
                ),
            },
        )
    )
    models = client.get("/info/").json()["models"]
    assert set(models) == {"arbitrary-image-id", "arbitrary-polyline-id"}
    assert models["arbitrary-image-id"]["config"]["input_profile"] == "image"
    assert models["arbitrary-polyline-id"]["config"]["input_profile"] == "polylines"
    assert models["arbitrary-image-id"]["labels"] == {"background": 0, "vessel": 1}
    assert models["arbitrary-image-id"]["title"] == ArgWorkflow().descriptor.title
    assert _route_inventory(client.app) == UPSTREAM_ROUTES | {("POST", "/infer/{model}/stream")}


def test_dicom_seg_is_deferred_before_artifact_or_execution(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_app(
        ServiceSettings(
            tmp_path,
            {"registered-model": _deployment(tmp_path, ArgWorkflow(), NoParameters.model_validate)},
        )
    )
    calls = []

    def unexpected_work(*args: Any) -> Any:
        calls.append(args)
        raise AssertionError("Deferred output must not resolve artifacts or run inference")

    monkeypatch.setattr(MLflowArtifactSource, "resolve", unexpected_work)
    monkeypatch.setattr("monai_service.application.InferenceService.infer", unexpected_work)
    response = client.post(
        "/infer/registered-model",
        params={"output": "dicom_seg"},
        files={"file": ("ct.nii.gz", volume[0])},
    )
    assert response.status_code == 501, response.text
    assert response.json() == {"detail": "Not yet implemented"}
    assert not calls
    assert not (tmp_path / "requests").exists()
    assert not (tmp_path / "models").exists()


def test_deferred_dicom_seg_retains_native_auth(client: TestClient) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: User(username="reader", roles=[])
    assert client.post("/infer/smoke-test?output=dicom_seg").status_code == 403
    client.app.dependency_overrides.pop(get_current_user)
    assert client.post("/infer/smoke-test?output=dicom_seg").status_code == 401


def test_request_captures_version_before_download_and_execution(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []
    selected_versions = iter(["12", "13"])

    def select(source: ModelSource) -> SelectedModel:
        version = next(selected_versions)
        events.append(("select", version))
        return SelectedModel(source.name, version, source.tracking_uri, source.tracking_uri)

    def download(selected: SelectedModel, cache_root: Path) -> Path:
        events.append(("download", selected.version))
        path = cache_root / selected.version
        path.mkdir(parents=True)
        return path

    class Executor:
        def run(self, invocation: Any) -> ExecutionResult:
            paths = {binding.name: binding.path for binding in invocation.bindings}
            events.append(("execute", paths["model"].name))
            assert (
                invocation.arguments[invocation.arguments.index("--model-file") + 1] == "model.onnx"
            )
            image = sitk.ReadImage(str(paths["image"]))
            sitk.WriteImage(
                sitk.Cast(image > 0, sitk.sitkUInt8), str(paths["output"] / "mask.nii.gz")
            )
            (paths["output"] / "metadata.json").write_text(
                json.dumps(
                    {
                        "model": "untrusted-output",
                        "model_version": "untrusted-output",
                        "model_artifact": "untrusted-output",
                        "runtime": "untrusted-output",
                        "worker": "arg",
                        "preprocessing": "native-plans",
                        "device_preference": "auto",
                        "model_file": "model.onnx",
                    }
                )
            )
            return ExecutionResult("configured-runtime", "configured-reference")

    monkeypatch.setattr(model_adapter, "select_model", select)
    monkeypatch.setattr(model_adapter, "download_model", download)
    create_app(
        ServiceSettings(
            tmp_path,
            {
                "registered-model": _deployment(
                    tmp_path, ArgWorkflow(), NoParameters.model_validate, Executor()
                )
            },
        )
    )
    for version in ("12", "13"):
        response = client.post("/infer/registered-model", files={"file": ("ct.nii.gz", volume[0])})
        result, metadata = _read_result(response, tmp_path)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(result), volume[1] > 0)
        assert metadata["model"] == "ConfiguredModel"
        assert metadata["model_version"] == version
        assert metadata["model_artifact"] == f"models:/ConfiguredModel/{version}"
        assert metadata["preprocessing"] == "native-plans"
        assert metadata["runtime"] == "configured-runtime"
        assert metadata["runtime_reference"] == "configured-reference"
    assert events == [
        (action, version)
        for version in ("12", "13")
        for action in ("select", "download", "execute")
    ]
    assert not list((tmp_path / "requests").iterdir())


@pytest.mark.parametrize("stage", ["select", "download"])
def test_artifact_failures_are_clear_without_credential_leaks(
    stage: str,
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_app(
        ServiceSettings(
            tmp_path,
            {"registered-model": _deployment(tmp_path, ArgWorkflow(), NoParameters.model_validate)},
        )
    )
    monkeypatch.setattr(
        model_adapter,
        "select_model",
        lambda source: SelectedModel("Model", "12", "http://mlflow", "http://mlflow"),
    )

    def unavailable(*args: Any) -> Any:
        raise RuntimeError("http://user:password@private-store is unavailable")

    monkeypatch.setattr(model_adapter, f"{stage}_model", unavailable)
    response = client.post("/infer/registered-model", files={"file": ("ct.nii.gz", volume[0])})
    assert response.status_code == 502
    assert response.json() == {"detail": "Configured model artifact is unavailable"}
    assert "password" not in response.text


@pytest.mark.parametrize(
    "key",
    [
        "model_path",
        "worker_image",
        "tracking_uri",
        "registry_uri",
        "device",
        "model_file",
        "preprocessing",
    ],
)
def test_parameter_schema_rejects_client_selected_runtime_and_artifacts(
    client: TestClient, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "monai_service.application.InferenceService.infer",
        lambda *args: pytest.fail("Invalid parameters must not enter the inference service"),
    )
    response = client.post(
        "/infer/smoke-test",
        data={"params": json.dumps({key: "client-choice"})},
        files={"file": ("ct.nii.gz", b"not read")},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == [
        {
            "type": "extra_forbidden",
            "loc": ["body", "params", key],
            "msg": "Extra inputs are not permitted",
        }
    ]


def test_parameter_parser_domain_errors_are_422_before_inference(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_parameters(value: object) -> NoParameters:
        raise InvalidRequest("Parameters rejected by this deployment")

    create_app(
        ServiceSettings(
            tmp_path,
            {"registered-model": _deployment(tmp_path, ArgWorkflow(), reject_parameters)},
        )
    )
    monkeypatch.setattr(
        "monai_service.application.InferenceService.infer",
        lambda *args: pytest.fail("Rejected parameters must not enter the inference service"),
    )
    response = client.post("/infer/registered-model", files={"file": ("ct.nii.gz", volume[0])})
    assert response.status_code == 422, response.text
    assert response.json() == {"detail": "Parameters rejected by this deployment"}
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "requests").exists()


@pytest.mark.parametrize("point_count", [2, 60000])
def test_model_o_prepares_validated_polylines_and_shared_provenance(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point_count: int,
) -> None:
    parameters = {"polylines_zyx": [[[0.25 + i % 2, 1.0, 1.0] for i in range(point_count)]]}
    serialized_parameters = json.dumps(parameters)
    if point_count > 2:
        assert len(serialized_parameters.encode()) > 1024 * 1024
    model_path = tmp_path / "model-version-7"
    model_path.mkdir()
    selected = SelectedModel("ModelO", "7", "http://mlflow", "http://mlflow")
    selections = []

    def select(source: ModelSource) -> SelectedModel:
        selections.append(source)
        return selected

    class Executor:
        def run(self, invocation: Any) -> ExecutionResult:
            paths = {binding.name: binding.path for binding in invocation.bindings}
            assert json.loads(paths["parameters"].read_text()) == {**parameters, "overlap": 0.5}
            assert paths["assets"] == tmp_path
            assert paths["model"] == model_path
            image = sitk.ReadImage(str(paths["image"]))
            sitk.WriteImage(
                sitk.Cast(image > 0, sitk.sitkUInt8), str(paths["output"] / "mask.nii.gz")
            )
            (paths["output"] / "metadata.json").write_text(json.dumps({"fixture_marker": 3}))
            return ExecutionResult("test", "test-executor")

    monkeypatch.setattr(model_adapter, "select_model", select)
    monkeypatch.setattr(model_adapter, "download_model", lambda _model, _cache: model_path)
    create_app(
        ServiceSettings(
            tmp_path,
            {
                "polyline-model": _deployment(
                    tmp_path,
                    ModelOWorkflow(settings=ModelOSettings(assets=tmp_path)),
                    ModelOParameters.model_validate,
                    Executor(),
                )
            },
        )
    )
    response = client.post(
        "/infer/polyline-model",
        files={"file": ("ct.nii.gz", volume[0])},
        data={"params": serialized_parameters},
    )
    result, metadata = _read_result(response, tmp_path)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), volume[1] > 0)
    assert metadata["model_version"] == "7" and metadata["model_artifact"] == "models:/ModelO/7"
    assert metadata["fixture_marker"] == 3 and len(selections) == 1
    assert not list((tmp_path / "requests").iterdir())


@pytest.mark.parametrize(
    "limit, part_size, expected",
    [
        (1024, 1024, 200),
        (1024, 1025, 400),
        (None, 16 * 1024 * 1024, 200),
        (None, 16 * 1024 * 1024 + 1, 400),
    ],
)
def test_inference_form_uses_configured_part_limit_and_closes_uploads(
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: int | None,
    part_size: int,
    expected: int,
) -> None:
    from starlette import formparsers

    spools = []
    native_spool = formparsers.SpooledTemporaryFile

    def capture_spool(*args: Any, **kwargs: Any) -> Any:
        spool = native_spool(*args, **kwargs)
        spools.append(spool)
        return spool

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", capture_spool)
    if limit is not None:
        monkeypatch.setenv("MONAI_LABEL_INFER_MAX_PART_SIZE", str(limit))
    create_test_app(ServiceSettings(work_dir=tmp_path))
    response = client.post(
        "/infer/smoke-test",
        files=[
            ("file", ("ct.nii.gz", volume[0])),
            ("params", (None, "{}" + " " * (part_size - 2))),
        ],
    )
    assert response.status_code == expected, response.text
    assert spools and all(spool.closed for spool in spools)
    if expected == 400:
        maximum = limit or 16 * 1024 * 1024
        assert response.json() == {"detail": f"Part exceeded maximum size of {maximum // 1024}KB."}
        assert not (tmp_path / "requests").exists()


@pytest.mark.parametrize("roles, expected", [(None, 401), ([], 403)])
def test_large_inference_form_retains_auth_and_spooled_upload_cleanup(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    roles: list[str] | None,
    expected: int,
) -> None:
    from starlette import formparsers

    spools = []
    native_spool = formparsers.SpooledTemporaryFile

    def capture_spool(*args: Any, **kwargs: Any) -> Any:
        spool = native_spool(*args, **kwargs)
        spools.append(spool)
        return spool

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", capture_spool)
    if roles is None:
        client.app.dependency_overrides.pop(get_current_user)
    else:
        client.app.dependency_overrides[get_current_user] = lambda: User(
            username="reader", roles=roles
        )
    response = client.post(
        "/infer/smoke-test",
        files=[
            ("file", ("ct.nii.gz", b"x" * (1024 * 1024 + 1))),
            ("params", (None, "{}" + " " * (1024 * 1024))),
        ],
    )
    assert response.status_code == expected, response.text
    assert spools and all(spool.closed for spool in spools)
    assert all(spool._rolled for spool in spools)


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_invalid_inference_form_limit_fails_configuration(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    monkeypatch.setenv("MONAI_LABEL_INFER_MAX_PART_SIZE", limit)
    with pytest.raises(ValidationError, match="infer_max_part_size"):
        create_test_app(ServiceSettings(work_dir=tmp_path))


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"polylines_zyx": None},
        {"polylines_zyx": []},
        {"polylines_zyx": "filename.json"},
        {"polylines_zyx": [[]]},
        {"polylines_zyx": [[[0, 1, 1]]]},
        {"polylines_zyx": [[[0, 1], [1, 1, 1]]]},
        {"polylines_zyx": [[["0", 1, 1], [1, 1, 1]]]},
        {"polylines_zyx": [[[True, 1, 1], [1, 1, 1]]]},
        {"polylines_zyx": [[[float("nan"), 1, 1], [1, 1, 1]]]},
        {"polylines_zyx": [[[float("inf"), 1, 1], [1, 1, 1]]]},
        {"polylines_zyx": [[[float("-inf"), 1, 1], [1, 1, 1]]]},
        {"tree": {"nodes": [], "edges": []}},
        {"hu_ceiling": 950},
        {"apply_cleanup": False},
        {"tracking_uri": "https://user:secret-password@registry.example"},
    ],
)
def test_model_o_invalid_options_are_422_before_artifact_resolution(
    params: dict[str, object],
    client: TestClient,
    volume: tuple[bytes, np.ndarray, sitk.Image],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_app(
        ServiceSettings(
            tmp_path,
            {
                "polyline-model": _deployment(
                    tmp_path,
                    ModelOWorkflow(settings=ModelOSettings(assets=tmp_path)),
                    ModelOParameters.model_validate,
                )
            },
        )
    )

    def unexpected_resolution(source: Any) -> Any:
        pytest.fail("Malformed requests must not resolve or download artifacts")

    monkeypatch.setattr(MLflowArtifactSource, "resolve", unexpected_resolution)
    monkeypatch.setattr("monai_service.application.InferenceService.infer", unexpected_resolution)
    response = client.post(
        "/infer/polyline-model",
        files={"file": ("ct.nii.gz", volume[0])},
        data={"params": json.dumps({**_model_o_parameters(), **params} if params else {})},
    )
    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    assert all(error["loc"][:2] == ["body", "params"] for error in errors)
    assert any(error["loc"][2] == next(iter(params), "polylines_zyx") for error in errors)
    assert all("input" not in error and "ctx" not in error for error in errors)
    assert "secret-password" not in response.text
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "requests").exists()
