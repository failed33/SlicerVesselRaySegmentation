"""Accept installed backend wheels through real ONNX, MLflow and Label HTTP."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import runpy
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
import SimpleITK as sitk
from mlflow import MlflowClient
from requests_toolbelt.multipart.decoder import MultipartDecoder

if TYPE_CHECKING:
    from tests.fixture_factory import ArgCase


def check_installed_packages() -> dict[str, str]:
    locations = {}
    for name in (
        "dawn_ort",
        "arg_inference",
        "model_o_inference",
        "monai_service",
        "asgi",
        "vessel_deployment",
        "label_app.main",
    ):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None, name
        path = Path(spec.origin).resolve()
        assert path.is_relative_to(Path(sys.prefix).resolve()), (name, path, sys.prefix)
        locations[name] = str(path.relative_to(sys.prefix))
    from model_o_inference.shaders import load_shader

    for name in ("patch_inputs",):
        assert "@compute" in load_shader(name).code
    return locations


def assert_mask(path: Path, case: ArgCase) -> None:
    mask, source = sitk.ReadImage(str(path)), sitk.ReadImage(str(case.image))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), case.expected)
    for getter in ("GetSize", "GetOrigin", "GetSpacing", "GetDirection"):
        np.testing.assert_allclose(
            getattr(mask, getter)(), getattr(source, getter)(), rtol=0, atol=1e-6
        )


def service_environment(root: Path, mode: str, container: bool) -> dict[str, str]:
    tracking = f"sqlite:///{root / 'registry.db'}"
    executable = "/opt/monai/.venv/bin/python" if container else sys.executable
    worker = (
        {"mode": "direct"}
        if mode == "direct"
        else {
            "mode": "process",
            "command": [executable, "-m", "monai_service.models.arg"],
            "timeout_seconds": 45,
        }
    )
    return {
        "MONAI_SERVICE_ENABLE_ARG": "true",
        "MONAI_SERVICE_ENABLE_MODEL_O": "false",
        "MONAI_SERVICE_MLFLOW_TRACKING_URI": tracking,
        "MONAI_SERVICE_MLFLOW_REGISTRY_URI": tracking,
        "MONAI_SERVICE_ARG_MODEL_NAME": "packaging-fixture",
        "MONAI_SERVICE_ARG_MODEL_ALIAS": "champion",
        "MONAI_SERVICE_ARG_DEVICE": "cpu",
        "MONAI_SERVICE_ARG_WORKER": json.dumps(worker),
        "MONAI_SERVICE_WORK_DIR": str(root / f"service-{mode}"),
        "MONAI_LABEL_SESSION_PATH": str(root / f"sessions-{mode}"),
        "MONAI_LABEL_AUTH_ENABLE": "false",
        "MONAI_LABEL_DATASTORE_AUTO_RELOAD": "false",
        "ORT_DISABLE_TELEMETRY": "1",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "MPLCONFIGDIR": str(root / "matplotlib"),
        "TMPDIR": str(root),
    }


@contextmanager
def running_service(root: Path, mode: str, server_image: str | None):
    host = os.environ.get("VRS_CONTAINER_HOST", "127.0.0.1") if server_image else "127.0.0.1"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = service_environment(root, mode, server_image is not None)
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    name = f"vessel-ci-{uuid.uuid4().hex}"
    command = [
        sys.executable,
        "-I",
        "-m",
        "uvicorn",
        "asgi:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if server_image:
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--volume",
            f"{root}:{root}",
            "--publish",
            f"{port}:8000",
        ]
        for key, value in settings.items():
            command.extend(["--env", f"{key}={value}"])
        command.append(server_image)
    log_path = root / f"server-{mode}.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            command, cwd=root, env={**environment, **settings}, stdout=log, stderr=log
        )
        try:
            with httpx.Client(base_url=f"http://{host}:{port}", timeout=60) as client:
                deadline = time.monotonic() + 90
                while True:
                    assert process.poll() is None, "Server exited during startup"
                    try:
                        if client.get("/info/", timeout=1).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert time.monotonic() < deadline, "Server startup timed out"
                    time.sleep(0.1)
                yield client
        except BaseException:
            log.seek(0)
            print(log.read()[-12000:])
            raise
        finally:
            if server_image:
                subprocess.run(
                    ["docker", "stop", "--time", "10", name], capture_output=True, timeout=30
                )
            else:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def check_http(root: Path, case: ArgCase, mode: str, server_image: str | None) -> dict:
    with running_service(root, mode, server_image) as client:
        info = client.get("/info/")
        info.raise_for_status()
        assert "arg_vessel" in info.json()["models"], info.text
        response = client.post(
            "/infer/arg_vessel",
            files={"file": ("ct.nii.gz", case.image.read_bytes())},
            data={"params": "{}"},
        )
        response.raise_for_status()
        parts = {
            part.headers[b"Content-Disposition"].split(b'name="')[1].split(b'"')[0]: part.content
            for part in MultipartDecoder(response.content, response.headers["content-type"]).parts
        }
        mask_path = root / f"mask-{mode}.nii.gz"
        mask_path.write_bytes(parts[b"image"])
        assert_mask(mask_path, case)
        metadata = json.loads(parts[b"params"])
        assert metadata["runtime"] == mode, metadata
        assert metadata["inference_runtime"] == "onnxruntime", metadata
        assert metadata["providers"] == ["CPUExecutionProvider"], metadata
        requests = root / f"service-{mode}" / "requests"
        deadline = time.monotonic() + 5
        while list(requests.iterdir()):
            assert time.monotonic() < deadline, "Request cleanup did not finish"
            time.sleep(0.01)
        return {
            "mode": mode,
            "status": "passed",
            "foreground_voxels": int(case.expected.sum()),
            "runtime": metadata["inference_runtime"],
            "providers": metadata["providers"],
        }


def check_worker_container(root: Path, case: ArgCase, image: str) -> dict:
    output, metadata = root / "worker.nii.gz", root / "worker.json"
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--volume",
            f"{root}:{root}",
            image,
            "--input",
            str(case.image),
            "--model",
            str(case.model_dir),
            "--output",
            str(output),
            "--metadata",
            str(metadata),
            "--device",
            "cpu",
        ],
        check=True,
        timeout=90,
    )
    assert_mask(output, case)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--entrypoint",
            "/opt/monai/.venv/bin/python",
            image,
            "-c",
            "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('nnunetv2') is None",
        ],
        check=True,
        timeout=30,
    )
    return {
        "status": "passed",
        "image": image,
        "foreground_voxels": int(case.expected.sum()),
        "torch": False,
        "nnunet": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-image")
    parser.add_argument("--worker-image")
    args = parser.parse_args()
    locations = check_installed_packages()
    fixture = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "deployment/monai/tests/fixture_factory.py")
    )
    with tempfile.TemporaryDirectory(prefix="vessel-backend-") as temporary:
        root = Path(temporary)
        case = fixture["make_arg_case"](root / "fixture")
        uri = f"sqlite:///{root / 'registry.db'}"
        registry = MlflowClient(tracking_uri=uri, registry_uri=uri)
        registry.create_registered_model("packaging-fixture")
        version = registry.create_model_version("packaging-fixture", case.model_dir.as_uri())
        registry.set_registered_model_alias("packaging-fixture", "champion", version.version)
        root.chmod(0o777)
        (root / "registry.db").chmod(0o666)
        checks = [check_http(root, case, mode, args.server_image) for mode in ("direct", "process")]
        worker = (
            check_worker_container(root, case, args.worker_image) if args.worker_image else None
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for log in root.glob("server-*.log"):
            (args.output.parent / log.name).write_bytes(log.read_bytes())
        report = {
            "status": "passed",
            "synthetic": True,
            "source_imports": False,
            "packages": locations,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("vessel-inference", "monai-label-service", "onnxruntime", "monailabel")
            },
            "http": checks,
            "worker_container": worker,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
