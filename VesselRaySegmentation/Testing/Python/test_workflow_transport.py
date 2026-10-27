"""Real local HTTP multipart transport and native image-grid validation."""

import json
import ssl
import threading
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from VesselRaySegmentationLib.monai import client, submission
from VesselRaySegmentationLib.monai.images import VolumeSnapshot


def multipart(image: bytes, *, filename="mask.nrrd", metadata=None) -> bytes:
    boundary = "monai-test-boundary"
    return (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="params"\r\n'
            f"Content-Type: application/json\r\n\r\n{json.dumps(metadata or {})}\r\n"
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )


@pytest.fixture
def label_server():
    state = SimpleNamespace(
        requests=[],
        models={"smoke-test": {"config": {"input_profile": "image"}}},
        status=200,
        delay=0.0,
        body=multipart(b"result\x00\xff", metadata={"test_only": True}),
        content_type="multipart/form-data; boundary=monai-test-boundary",
        sessions={},
        deleted_sessions=[],
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.requests.append((self.path, self.headers, b""))
            session_id = self.path.split("/")[-1].split("?")[0]
            is_session = self.path.startswith("/session/")
            self.send_response(404 if is_session and session_id not in state.sessions else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": state.models}).encode())

        def do_PUT(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state.requests.append((self.path, self.headers, body))
            session_id = f"session-{len(state.sessions) + 1}"
            state.sessions[session_id] = body
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"session_id": session_id}).encode())

        def do_DELETE(self):
            session_id = self.path.split("/")[-1]
            state.deleted_sessions.append(session_id)
            state.sessions.pop(session_id, None)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state.requests.append((self.path, self.headers, body))
            time.sleep(state.delay)
            self.send_response(state.status)
            self.send_header("Content-Type", state.content_type)
            if state.status == 302:
                self.send_header("Location", "/redirected")
            self.end_headers()
            try:
                self.wfile.write(state.body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "filename, uploaded_filename",
    [
        ("patient-secret.nrrd", "ct.nrrd"),
        ("patient-secret.nii.gz", "ct.nii.gz"),
        ('patient-secret.n"rrd', "ct.n%22rrd"),
    ],
)
def test_actual_multipart_upload_and_response(label_server, tmp_path, filename, uploaded_filename):
    image = tmp_path / filename
    image.write_bytes(b"submitted\x00\r\n\xff")
    connection = client.ConnectionSettings(label_server.endpoint, 2, "private-test-token")
    transport = client.MonaiClient(connection)
    assert transport.info()["models"]["smoke-test"]["config"]["input_profile"] == "image"
    parameters = {"polylines_zyx": [[[0.25, 1.5, 2.75], [1.75, 2.25, 3.5]]]}
    result = transport.infer(image, "smoke-test", parameters)
    path, headers, body = label_server.requests[-1]
    assert path == "/infer/smoke-test?output=all"
    assert headers["Authorization"] == "Bearer private-test-token"
    assert int(headers["Content-Length"]) == len(body)
    assert b"patient-secret" not in body
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body
    )
    parts = {
        part.get_param("name", header="content-disposition"): part for part in message.iter_parts()
    }
    payload = parts["params"].get_payload(decode=True)
    assert isinstance(payload, bytes)
    assert json.loads(payload) == parameters
    assert parts["file"].get_filename() == uploaded_filename
    assert parts["file"].get_payload(decode=True) == image.read_bytes()
    assert result.image == b"result\x00\xff" and result.filename == "mask.nrrd"
    assert result.metadata == {"test_only": True}
    assert "private-test-token" not in repr(connection)


@pytest.mark.parametrize(
    "status, text", [(404, "unavailable"), (401, "rejected access"), (501, "Not yet implemented")]
)
def test_http_failures_are_visible_and_redact_tokens(label_server, tmp_path, status, text):
    path = tmp_path / "ct.nrrd"
    path.write_bytes(b"ct")
    label_server.status = status
    label_server.content_type = "application/json"
    label_server.body = json.dumps({"detail": "Not yet implemented private-token"}).encode()
    transport = client.MonaiClient(
        client.ConnectionSettings(label_server.endpoint, 2, "private-token")
    )
    with pytest.raises(client.MonaiClientError, match=text) as error:
        transport.infer(path, "model_o", {})
    assert "private-token" not in str(error.value)
    assert error.value.status_code == status


def test_native_session_upload_and_parameters_only_inference(label_server, tmp_path):
    path = tmp_path / "patient-secret.nii.gz"
    path.write_bytes(b"submitted-ct")
    transport = client.MonaiClient(
        client.ConnectionSettings(label_server.endpoint, 2, "private-token")
    )
    session_id = transport.create_session(path)
    _url, headers, body = label_server.requests[-1]
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body
    )
    parts = list(message.iter_parts())
    assert len(parts) == 1
    assert parts[0].get_param("name", header="content-disposition") == "files"
    assert parts[0].get_filename() == "ct.nii.gz"
    assert parts[0].get_payload(decode=True) == path.read_bytes()
    assert headers["Authorization"] == "Bearer private-token"
    assert transport.session_exists(session_id)
    transport.infer(client.SessionInput(session_id), "model_o", {"overlap": 0.25})
    url, headers, body = label_server.requests[-1]
    assert url == f"/infer/model_o?output=all&session_id={session_id}"
    assert b'name="file"' not in body and b"submitted-ct" not in body
    assert b'"overlap": 0.25' in body
    transport.delete_session(session_id)
    assert not transport.session_exists(session_id)


def test_session_input_requires_an_identifier():
    with pytest.raises(ValueError, match="nonempty"):
        client.SessionInput("")


def test_network_timeout_and_redirect_are_explicit(label_server, tmp_path):
    path = tmp_path / "ct.nrrd"
    path.write_bytes(b"ct")
    label_server.delay = 0.15
    transport = client.MonaiClient(client.ConnectionSettings(label_server.endpoint, 0.02))
    with pytest.raises(client.MonaiClientError, match="timed out"):
        transport.infer(path, "smoke-test", {})
    label_server.delay = 0
    label_server.status = 302
    transport = client.MonaiClient(
        client.ConnectionSettings(label_server.endpoint, 2, "private-token")
    )
    with pytest.raises(client.MonaiClientError, match="HTTP 302"):
        transport.infer(path, "smoke-test", {})
    assert not any(path == "/redirected" for path, _headers, _body in label_server.requests)


def test_https_uses_certificate_and_hostname_verification(monkeypatch):
    contexts = []

    def request(_method, _url, **options):
        contexts.append(options["verify"])
        return httpx.Response(200, json={"models": {}})

    monkeypatch.setattr(httpx, "request", request)
    client.MonaiClient(client.ConnectionSettings("https://example.invalid")).info()
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert contexts[0].check_hostname is True


def test_missing_segmentation_multipart_is_rejected(label_server, tmp_path):
    path = tmp_path / "ct.nrrd"
    path.write_bytes(b"ct")
    label_server.content_type = "application/json"
    label_server.body = b'{"test_only":true}'
    with pytest.raises(client.MonaiClientError, match="no segmentation"):
        client.MonaiClient(client.ConnectionSettings(label_server.endpoint)).infer(
            path, "smoke-test", {}
        )


def test_unadvertised_model_fails_before_upload(label_server):
    snapshot = VolumeSnapshot.capture(
        np.zeros((3, 4, 5), dtype=np.float32), np.eye(4), ("id", "CT")
    )
    request = submission.InferenceRequest(
        snapshot, client.ConnectionSettings(label_server.endpoint), "model_o", {}, "polylines"
    )
    with pytest.raises(client.MonaiClientError, match="input requirements changed"):
        submission.run_inference(request)
    assert len(label_server.requests) == 1


def test_changed_input_profile_is_rejected_before_upload(label_server):
    label_server.models = {"selected": {"config": {"input_profile": "polylines"}}}
    snapshot = VolumeSnapshot.capture(
        np.zeros((3, 4, 5), dtype=np.float32), np.eye(4), ("id", "CT")
    )
    request = submission.InferenceRequest(
        snapshot, client.ConnectionSettings(label_server.endpoint), "selected", {}, "image"
    )
    with pytest.raises(client.MonaiClientError, match="input requirements changed"):
        submission.run_inference(request)
    assert len(label_server.requests) == 1


def endpoint_interpreter():
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    return Path(
        os.environ.get("VRS_TEST_ENDPOINT_PYTHON", str(root / "build/backend/installed/bin/python"))
    )


@pytest.fixture
def canonical_endpoint(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    interpreter = endpoint_interpreter()
    if not interpreter.is_file():
        pytest.skip("Canonical Label HTTP acceptance requires the backend service environment")
    path = root / "tests/integration/slicer/endpoint.py"
    spec = importlib.util.spec_from_file_location("workflow_test_endpoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with module.start_endpoint(tmp_path / "label", python=interpreter) as endpoint:
        yield endpoint


def fixture_image(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    values = np.full((8, 9, 10), -20, np.float32)
    values[1:7, 2:7, 2:8] = 250
    image = sitk.GetImageFromArray(values)
    image.SetOrigin((17.3, -8.6, 5.2))
    image.SetSpacing((0.8, 1.1, 1.4))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    path = tmp_path / "private-source.nrrd"
    sitk.WriteImage(image, str(path))
    return path, image, values


def read_fixture_response(response, tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    path = tmp_path / response.filename
    path.write_bytes(response.image)
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def test_canonical_label_auth_profiles_and_cross_origin_preflight(canonical_endpoint):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    endpoint = canonical_endpoint
    connection = client.ConnectionSettings(endpoint.endpoint, 10, endpoint.token)
    models = client.MonaiClient(connection).models()
    assert models["arg-fixture"].input_profile == "image"
    assert models["model-o-fixture"].input_profile == "polylines"
    for token in ("", "wrong-token"):
        with pytest.raises(client.MonaiClientError) as error:
            client.MonaiClient(client.ConnectionSettings(endpoint.endpoint, 10, token)).models()
        assert error.value.status_code == 401
    headers = {
        "Origin": "http://127.0.0.1:4173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }
    url = endpoint.endpoint + "/infer/model-o-fixture"
    with urlopen(Request(url, headers=headers, method="OPTIONS"), timeout=10) as response:
        assert response.status == 200
        assert response.headers["access-control-allow-origin"] == headers["Origin"]
        assert "authorization" in response.headers["access-control-allow-headers"].lower()
    headers["Origin"] = "http://untrusted.invalid"
    with pytest.raises(HTTPError) as error:
        urlopen(Request(url, headers=headers, method="OPTIONS"), timeout=10)
    assert error.value.code == 400
    assert "access-control-allow-origin" not in error.value.headers


def test_scene_close_fixture_holds_inference_until_explicit_release(canonical_endpoint, tmp_path):
    import sys
    from concurrent.futures import ThreadPoolExecutor

    module = sys.modules["workflow_test_endpoint"]
    interpreter = endpoint_interpreter()
    image, _, _ = fixture_image(tmp_path)
    with module.start_endpoint(
        tmp_path / "barrier", python=interpreter, hold_request=1
    ) as endpoint:
        transport = client.MonaiClient(
            client.ConnectionSettings(endpoint.endpoint, 20, endpoint.token)
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(transport.infer, image, "arg-fixture", {})
            try:
                deadline = time.monotonic() + 10
                while not any(event["kind"] == "inference-start" for event in endpoint.events()):
                    assert time.monotonic() < deadline, "Inference did not start"
                    time.sleep(0.01)
                assert not response.done()
                assert not any(event["kind"] == "inference" for event in endpoint.events())
                released_at = time.time()
            finally:
                endpoint.release_inference()
            assert response.result(timeout=10).image
        finishes = [event for event in endpoint.events() if event["kind"] == "inference"]
        assert finishes[0]["timestamp"] > released_at


def test_canonical_session_arg_and_model_o_use_uploaded_ct_and_edited_polylines(
    canonical_endpoint,
    tmp_path,
):
    endpoint = canonical_endpoint
    path, image, values = fixture_image(tmp_path)
    transport = client.MonaiClient(client.ConnectionSettings(endpoint.endpoint, 20, endpoint.token))
    session_id = transport.create_session(path)
    uploaded = client.SessionInput(session_id)
    response = transport.infer(uploaded, "arg-fixture", {})
    returned, mask = read_fixture_response(response, tmp_path)
    np.testing.assert_array_equal(mask, values > 0)
    assert returned.GetOrigin() == image.GetOrigin()
    assert returned.GetSpacing() == image.GetSpacing()
    assert returned.GetDirection() == image.GetDirection()
    assert response.metadata["test_only"] is True
    first = {"polylines_zyx": [[[1.25, 3.0, 3.0], [5.75, 3.0, 3.0]]]}
    second = {"polylines_zyx": [[[1.25, 4.0, 4.0], [5.75, 4.0, 4.0]]]}
    masks = []
    for parameters in (first, second):
        response = transport.infer(uploaded, "model-o-fixture", parameters)
        _grid, mask = read_fixture_response(response, tmp_path)
        masks.append(mask.copy())
        assert mask.any() and np.all(mask <= (values > 0))
        assert transport.session_exists(session_id)
    assert not np.array_equal(*masks), "Edited centerlines did not affect actual test computation"
    inference = [event for event in endpoint.events() if event["kind"] == "inference"]
    assert [event["profile"] for event in inference] == ["image", "polylines", "polylines"]
    assert [event["parameters"]["polylines_zyx"] for event in inference[1:]] == [
        first["polylines_zyx"],
        second["polylines_zyx"],
    ]
    requests = [event for event in endpoint.events() if event["kind"] == "http"]
    assert sum(event["method"] == "PUT" for event in requests) == 1
    transport.delete_session(session_id)
    assert not transport.session_exists(session_id)


def test_canonical_model_profile_guard_precedes_upload(canonical_endpoint):
    endpoint = canonical_endpoint
    snapshot = VolumeSnapshot.capture(np.ones((3, 4, 5), np.float32), np.eye(4), ("id", "CT"))
    request = submission.InferenceRequest(
        snapshot,
        client.ConnectionSettings(endpoint.endpoint, 10, endpoint.token),
        "model-o-fixture",
        {},
        "image",
    )
    with pytest.raises(client.MonaiClientError, match="input requirements changed"):
        submission.run_inference(request)
    assert not any(event.get("method") == "PUT" for event in endpoint.events())
    assert not any(event["kind"] == "inference" for event in endpoint.events())


def test_canonical_invalid_polyline_does_not_execute(canonical_endpoint, tmp_path):
    endpoint = canonical_endpoint
    path, _image, _values = fixture_image(tmp_path)
    transport = client.MonaiClient(client.ConnectionSettings(endpoint.endpoint, 10, endpoint.token))
    with pytest.raises(client.MonaiClientError) as error:
        transport.infer(path, "model-o-fixture", {"polylines_zyx": []})
    assert error.value.status_code in (400, 422)
    assert not any(event["kind"] == "inference" for event in endpoint.events())


def test_native_session_owner_reuses_real_upload_and_deletes_after_success(
    canonical_endpoint,
    tmp_path,
    monkeypatch,
):
    import shutil

    from VesselRaySegmentationLib.monai.sessions import VolumeSession

    path, _image, values = fixture_image(tmp_path)
    endpoint = canonical_endpoint
    connection = client.ConnectionSettings(endpoint.endpoint, 10, endpoint.token)
    snapshot = VolumeSnapshot.capture(values, np.eye(4), ("id", "CT"))
    monkeypatch.setattr(
        VolumeSnapshot, "write_image", lambda self, target: shutil.copyfile(path, target)
    )
    owner = VolumeSession(connection, snapshot)
    identifiers = []
    for _ in range(2):
        with owner.use() as (transport, image):
            identifiers.append(image.session_id)
            response = transport.infer(image, "arg-fixture", {})
            assert response.metadata["test_only"] is True
    assert identifiers[0] == identifiers[1]
    assert client.MonaiClient(connection).session_exists(identifiers[0])
    owner.close()
    assert not client.MonaiClient(connection).session_exists(identifiers[0])
    events = canonical_endpoint.events()
    assert sum(event.get("method") == "PUT" for event in events) == 1
    assert sum(event.get("method") == "DELETE" for event in events) == 1


@pytest.mark.parametrize("stage", ["models", "ensure"])
def test_browser_disposal_during_discovery_or_upload_prevents_inference(stage):
    import asyncio

    from VesselRaySegmentationLib.browser import BrowserSession

    async def scenario():
        owner = BrowserSession(client.ConnectionSettings(), SimpleNamespace())
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def models():
            if stage == "models":
                started.set()
                await release.wait()
            return {
                "selected": client.ModelDescriptor.from_info(
                    "selected", {"config": {"input_profile": "image"}}
                )
            }

        async def ensure():
            started.set()
            await release.wait()
            owner.session_id = "uploaded-session"
            return owner.session_id

        async def request(path, method="GET", upload=None):
            calls.append((method, path))
            return "application/json", b"{}"

        owner.client.models = models
        owner.client.request = request
        owner.ensure = ensure
        operation = asyncio.create_task(
            owner.infer(SimpleNamespace(model="selected", input_profile="image", parameters={}))
        )
        await started.wait()
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(client.MonaiClientError, match="session is closed"):
            await operation
        await closing
        assert not any(method == "POST" for method, _path in calls)
        assert calls == ([] if stage == "models" else [("DELETE", "/session/uploaded-session")])

    asyncio.run(scenario())


def test_browser_timeout_includes_response_body_and_aborts_fetch(monkeypatch):
    import asyncio
    import sys
    from types import ModuleType

    from VesselRaySegmentationLib.browser import BrowserClient

    state = SimpleNamespace(aborted=False)
    controller = SimpleNamespace(signal=object(), abort=lambda: setattr(state, "aborted", True))
    js = ModuleType("js")
    monkeypatch.setattr(
        js, "AbortController", SimpleNamespace(new=lambda: controller), raising=False
    )
    http = ModuleType("pyodide.http")

    async def read_body():
        await asyncio.sleep(2)
        return b"unreachable"

    async def pyfetch(*_args, **_kwargs):
        return SimpleNamespace(ok=True, bytes=read_body)

    monkeypatch.setattr(http, "pyfetch", pyfetch, raising=False)
    monkeypatch.setitem(sys.modules, "js", js)
    monkeypatch.setitem(sys.modules, "pyodide", ModuleType("pyodide"))
    monkeypatch.setitem(sys.modules, "pyodide.http", http)

    async def scenario():
        transport = BrowserClient(client.ConnectionSettings("http://127.0.0.1:9999", 0.02))
        with pytest.raises(client.MonaiClientError, match="timed out"):
            await asyncio.wait_for(transport.request("/info/"), timeout=0.3)
        assert state.aborted

    asyncio.run(scenario())


def test_endpoint_service_interpreter_ignores_embedded_slicer_python_environment(
    canonical_endpoint,
    tmp_path,
    monkeypatch,
):
    import sys

    module = sys.modules["workflow_test_endpoint"]
    interpreter = endpoint_interpreter()
    for name, value in {
        "PYTHONHOME": "/Applications/Slicer.app/Contents",
        "PYTHONEXECUTABLE": "/Applications/Slicer.app/Contents/bin/PythonSlicer",
        "__PYVENV_LAUNCHER__": "/Applications/Slicer.app/Contents/bin/PythonSlicer",
        "_PYTHON_SYSCONFIGDATA_NAME": "_missing_slicer_sysconfigdata",
        "PYTHONPLATLIBDIR": "missing-slicer-stdlib",
        "DYLD_LIBRARY_PATH": "/Applications/Slicer.app/Contents/lib",
        "LD_LIBRARY_PATH": "/Applications/Slicer.app/Contents/lib",
        "PYTHONPATH": "/invalid-inherited-slicer-pythonpath",
    }.items():
        monkeypatch.setenv(name, value)
    with module.start_endpoint(tmp_path / "isolated", python=interpreter) as endpoint:
        models = client.MonaiClient(
            client.ConnectionSettings(endpoint.endpoint, 10, endpoint.token)
        ).models()
        assert models["arg-fixture"].input_profile == "image"
        assert models["model-o-fixture"].input_profile == "polylines"


def test_large_upload_streams_and_closes_the_source(label_server, tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "private-source.nrrd"
    path.write_bytes(b"ct" * (2 * 1024 * 1024))
    opened = []
    original = Path.open

    class Source:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024, "Upload attempted to buffer the complete CT"
            return self.stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

    def open_source(source, *args, **kwargs):
        stream = original(source, *args, **kwargs)
        if source == path:
            opened.append(stream)
            return Source(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_source)
    transport = client.MonaiClient(client.ConnectionSettings(label_server.endpoint, 2))
    transport.create_session(path)
    assert opened and all(stream.closed for stream in opened)
    assert len(label_server.requests[0][2]) > path.stat().st_size


def test_failed_upload_closes_the_source(label_server, tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "ct.nrrd"
    path.write_bytes(b"submitted-ct")
    opened = []
    original = Path.open

    def open_source(source, *args, **kwargs):
        stream = original(source, *args, **kwargs)
        if source == path:
            opened.append(stream)
        return stream

    def fail(*_args, **_kwargs):
        raise httpx.ReadTimeout("offline")

    monkeypatch.setattr(Path, "open", open_source)
    monkeypatch.setattr(httpx, "request", fail)
    transport = client.MonaiClient(client.ConnectionSettings(label_server.endpoint, 2))
    with pytest.raises(client.MonaiClientError, match="timed out"):
        transport.create_session(path)
    assert opened and all(stream.closed for stream in opened)
