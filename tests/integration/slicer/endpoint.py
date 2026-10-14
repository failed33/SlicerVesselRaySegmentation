"""Real MONAI Label test endpoint for packaged desktop/browser workflow acceptance.

The two advertised tasks are explicitly synthetic. They execute on uploaded CT
voxels and submitted polylines, preserve image geometry and use native Label
routes, sessions, request validation and multipart responses. No trained weights
or production-only test routes are involved.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

DEFAULT_TOKEN = "workflow-test-token"
MODELS = {"arg": "arg-fixture", "model_o": "model-o-fixture"}


@dataclass(frozen=True)
class Endpoint:
    endpoint: str
    token: str
    events_file: Path

    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.events_file.read_text().splitlines()]


@contextmanager
def start_endpoint(
    work_dir: Path, *, python: Path | None = None, inference_delay: float = 0.0
) -> Iterator[Endpoint]:
    """Start an isolated official Label process with the service dependency set."""
    root = Path(__file__).resolve().parents[3]
    interpreter = python or Path(os.environ.get("VRS_TEST_ENDPOINT_PYTHON", sys.executable))
    work_dir.mkdir(parents=True, exist_ok=True)
    ready, events, log = (work_dir / name for name in ("ready.json", "events.jsonl", "server.log"))
    ready.unlink(missing_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(root / "deployment/monai"), str(root / "deployment/inference/src")]
        ),
        "ORT_DISABLE_TELEMETRY": "1",
        "MONAI_LABEL_AUTH_ENABLE": "true",
        "MONAI_LABEL_DATASTORE_AUTO_RELOAD": "false",
        "MLFLOW_DISABLE_TELEMETRY": "true",
    }
    # Slicer launches with an embedded Python home and bundled library paths.
    # The separately selected service interpreter must use its own installation.
    for name in (
        "PYTHONHOME",
        "PYTHONEXECUTABLE",
        "__PYVENV_LAUNCHER__",
        "_PYTHON_SYSCONFIGDATA_NAME",
        "PYTHONPLATLIBDIR",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "LD_LIBRARY_PATH",
    ):
        environment.pop(name, None)
    with log.open("w+") as output:
        process = subprocess.Popen(
            [
                str(interpreter),
                str(Path(__file__).resolve()),
                "--work-dir",
                str(work_dir),
                "--ready-file",
                str(ready),
                "--events-file",
                str(events),
                "--inference-delay",
                str(inference_delay),
            ],
            env=environment,
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError("MONAI fixture exited: " + log.read_text()[-6000:])
                if ready.is_file():
                    info = json.loads(ready.read_text())
                    request = Request(
                        info["endpoint"] + "/info/",
                        headers={"Authorization": "Bearer " + info["token"]},
                    )
                    try:
                        with urlopen(request, timeout=1) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("MONAI fixture startup: " + log.read_text()[-6000:])
                time.sleep(0.1)
            yield Endpoint(info["endpoint"], info["token"], events)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def create_application(
    work_dir: Path, events_file: Path, token: str, origins: list[str], inference_delay: float = 0.0
):
    """Compose deterministic tasks through the production service contracts."""
    import shutil

    import numpy as np
    import SimpleITK as sitk
    from fastapi import HTTPException
    from fastapi import Request as HttpRequest
    from fastapi.middleware.cors import CORSMiddleware
    from model_o_inference.schemas import ModelOParameters
    from monai_service.app import create_app
    from monai_service.config import ServiceSettings
    from monai_service.contracts import (
        ArtifactIdentity,
        InferenceResult,
        ModelDeployment,
        ModelDescriptor,
        ResolvedArtifact,
    )
    from monai_service.schemas.base import NoParameters
    from monailabel.config import settings
    from tests.fixtures import FixedArtifactSource, SmokeWorkflow

    settings.MONAI_LABEL_AUTH_ENABLE = True
    settings.MONAI_LABEL_CORS_ORIGINS = origins
    settings.MONAI_LABEL_SESSIONS = True
    settings.MONAI_LABEL_SESSION_PATH = str(work_dir / "sessions")
    settings.MONAI_LABEL_STUDIES = str(work_dir / "studies")
    lock = threading.Lock()
    events_file.parent.mkdir(parents=True, exist_ok=True)
    events_file.write_text("")

    def record(event: dict) -> None:
        with lock, events_file.open("a") as stream:
            stream.write(json.dumps({"timestamp": time.time(), **event}) + "\n")

    class SyntheticWorkflow(SmokeWorkflow):
        def __init__(self, profile: str) -> None:
            object.__setattr__(self, "profile", profile)

        @property
        def descriptor(self):
            return ModelDescriptor(
                title="Synthetic ARG" if self.profile == "image" else "Synthetic ModelO",
                description="Test-only uploaded-image/geometry computation; no trained weights.",
                labels={"background": 0, "vessel": 1},
                dimension=3,
                input_profile=self.profile,
            )

        def infer(self, request, artifact, workspace):
            record({"kind": "inference-start", "profile": self.profile})
            time.sleep(inference_delay)
            image = sitk.ReadImage(str(request.image))
            values = sitk.GetArrayFromImage(image)
            mask = values > 0
            parameters = request.parameters.model_dump(mode="json")
            if self.profile == "polylines":
                covered = np.zeros(values.shape, bool)
                for line in request.parameters.polylines_zyx:
                    for first, last in zip(line[:-1], line[1:], strict=True):
                        points = np.linspace(
                            first, last, max(2, int(np.linalg.norm(np.array(last) - first) * 4) + 1)
                        )
                        indices = np.rint(points).astype(np.int64)
                        indices = indices[np.all((indices >= 0) & (indices < values.shape), axis=1)]
                        if len(indices):
                            covered[tuple(indices.T)] = True
                mask &= covered
            result = sitk.GetImageFromArray(mask.astype(np.uint8))
            result.CopyInformation(image)
            path = workspace / "mask.nrrd"
            sitk.WriteImage(result, str(path), True)
            record(
                {
                    "kind": "inference",
                    "profile": self.profile,
                    "parameters": parameters,
                    "shape_zyx": list(values.shape),
                    "foreground_voxels": int(mask.sum()),
                    "origin_lps": list(image.GetOrigin()),
                    "spacing_xyz": list(image.GetSpacing()),
                }
            )
            return InferenceResult(path, {"test_only": True, "synthetic_input_dependent": True})

    artifact = FixedArtifactSource(
        ResolvedArtifact(
            ArtifactIdentity("workflow-fixture", "1", "test-fixture:workflow"),
            work_dir,
        )
    )
    app = create_app(
        ServiceSettings(
            work_dir,
            {
                MODELS["arg"]: ModelDeployment(
                    SyntheticWorkflow("image"), artifact, None, NoParameters.model_validate
                ),
                MODELS["model_o"]: ModelDeployment(
                    SyntheticWorkflow("polylines"), artifact, None, ModelOParameters.model_validate
                ),
            },
        )
    )
    # Label writes beneath app_dir. Keep the installed/source app immutable.
    app_dir = work_dir / "label-app"
    app_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(settings.MONAI_LABEL_APP_DIR) / "main.py", app_dir / "main.py")
    settings.MONAI_LABEL_APP_DIR = str(app_dir)
    from monailabel.endpoints.user.auth import User, get_current_user

    def authorized_user(request: HttpRequest):
        if request.headers.get("authorization") != "Bearer " + token:
            raise HTTPException(status_code=401, detail="Invalid fixture bearer token")
        return User(
            username="workflow-test",
            roles=[
                settings.MONAI_LABEL_AUTH_ROLE_ADMIN,
                settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR,
                settings.MONAI_LABEL_AUTH_ROLE_USER,
            ],
        )

    app.dependency_overrides[get_current_user] = authorized_user
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "PUT", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def trace(request, call_next):
        response = await call_next(request)
        record(
            {
                "kind": "http",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
            }
        )
        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--events-file", type=Path, required=True)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--inference-delay", type=float, default=0.0)
    parser.add_argument(
        "--origin", action="append", default=["http://127.0.0.1:4173", "http://localhost:4173"]
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    app = create_application(
        args.work_dir, args.events_file, args.token, args.origin, args.inference_delay
    )
    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        info = {
            "endpoint": f"http://127.0.0.1:{listener.getsockname()[1]}",
            "token": args.token,
            "models": MODELS,
        }
        args.ready_file.write_text(json.dumps(info))
        uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off")).run(
            sockets=[listener]
        )


if __name__ == "__main__":
    main()
