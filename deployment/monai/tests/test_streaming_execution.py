"""Cancellation uses the existing isolated process and container cleanup paths."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from monai_service.adapters import execution
from monai_service.adapters.execution import DockerExecutor, LocalProcessExecutor
from monai_service.contracts import ExecutionCancelled, ExecutionControl, Invocation
from monai_service.schemas.execution import ContainerConfig, ProcessConfig


def test_already_cancelled_stream_never_starts_process(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled = threading.Event()
    cancelled.set()
    monkeypatch.setattr(
        execution.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Process started")
    )
    executor = LocalProcessExecutor(ProcessConfig(mode="process", command=("unused",)))
    with pytest.raises(ExecutionCancelled):
        executor.run_observed(Invocation((), ()), ExecutionControl(cancelled, lambda: None))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cancellation proof")
def test_cancelled_docker_adapter_stops_runtime_group_and_removes_container(tmp_path: Path) -> None:
    started, removed, child_done = tmp_path / "started", tmp_path / "removed", tmp_path / "child"
    child_code = (
        f"import time; from pathlib import Path; time.sleep(0.7); Path({str(child_done)!r}).touch()"
    )
    runtime = tmp_path / "fake-docker"
    runtime.write_text(
        f"#!{sys.executable}\n"
        "import json,sys,subprocess,time\nfrom pathlib import Path\n"
        "if sys.argv[1]=='run':\n"
        f"    subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        f"    Path({str(started)!r}).write_text(json.dumps(sys.argv))\n"
        "    time.sleep(30)\n"
        "else:\n"
        f"    Path({str(removed)!r}).write_text(json.dumps(sys.argv))\n"
    )
    runtime.chmod(0o700)
    cancelled = threading.Event()

    def cancel_started_runtime() -> None:
        if started.exists():
            cancelled.set()

    executor = DockerExecutor(
        ContainerConfig(mode="container", image="fixture", runtime=str(runtime), timeout_seconds=5)
    )
    with pytest.raises(ExecutionCancelled):
        executor.run_observed(
            Invocation((), ()), ExecutionControl(cancelled, cancel_started_runtime)
        )
    command = json.loads(started.read_text())
    name = command[command.index("--name") + 1]
    assert json.loads(removed.read_text()) == [str(runtime), "rm", "--force", name]
    time.sleep(0.8)
    assert not child_done.exists()
