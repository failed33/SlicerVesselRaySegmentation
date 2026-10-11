"""Process fixtures exercise the actual server lifecycle and HTTP transport."""

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from monailabel.config import settings

# Upstream binds authentication while importing its routes. Establish it before
# collecting any test module; individual fixtures override authenticated users.
settings.MONAI_LABEL_AUTH_ENABLE = True

# ORT 1.30 imports otherwise write a telemetry session file into the working directory.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
PROJECT = Path(__file__).resolve().parents[1]
WORKER_PATHS = (PROJECT, PROJECT.parent / "inference/src")


@pytest.fixture
def live_service(tmp_path):
    @contextmanager
    def start(overrides, *, app="asgi:app"):
        environment = {
            **os.environ,
            "MONAI_SERVICE_WORK_DIR": str(tmp_path / "service"),
            "MONAI_LABEL_AUTH_ENABLE": "false",
            "MONAI_LABEL_DATASTORE_AUTO_RELOAD": "false",
            "MLFLOW_DISABLE_TELEMETRY": "true",
            **overrides,
        }
        # pytest's pythonpath setting does not reach the real server/worker processes.
        environment["PYTHONPATH"] = os.pathsep.join(
            [*map(str, WORKER_PATHS), environment.get("PYTHONPATH", "")]
        )
        with socket.socket() as listener, (tmp_path / "server.log").open("w+") as log:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    app,
                    "--fd",
                    str(listener.fileno()),
                ],
                pass_fds=(listener.fileno(),),
                env=environment,
                stdout=log,
                stderr=log,
            )
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
                    deadline = time.monotonic() + 30
                    while True:
                        assert process.poll() is None, "Server exited during startup"
                        try:
                            if client.get("/info/", timeout=0.2).status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        assert time.monotonic() < deadline, "Server startup deadline exceeded"
                        time.sleep(0.1)
                    yield client
            except BaseException:
                log.seek(0)
                print(log.read()[-12000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    return start
