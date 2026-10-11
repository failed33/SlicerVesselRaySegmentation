"""Behavioral checks for runtime adapters and native model workflows."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from pydantic import ValidationError

from monai_service.adapters import execution
from monai_service.adapters.execution import DockerExecutor, LocalProcessExecutor
from monai_service.application import InferenceService
from monai_service.contracts import (
    ArtifactIdentity,
    ExecutionError,
    ExecutionTimeout,
    FileArgument,
    FileBinding,
    InferenceRequest,
    Invocation,
    ModelDeployment,
    ResolvedArtifact,
)
from monai_service.models.arg import ArgPolicy, ArgWorkflow
from monai_service.models.model_o import ModelOParameters, ModelOSettings, ModelOWorkflow
from monai_service.schemas.base import NoParameters
from monai_service.schemas.execution import ContainerConfig, DeviceMapping, ProcessConfig
from monai_service.worker_io import prepare_worker_files, read_worker_result
from tests.fixtures import FixedArtifactSource, smoke_deployment


@pytest.fixture
def input_path(tmp_path: Path) -> Path:
    image = sitk.GetImageFromArray(np.arange(-30, 30, dtype=np.int16).reshape(3, 4, 5))
    angle = math.pi / 6
    image.SetDirection(
        (math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0, 0, 0, 1)
    )
    image.SetSpacing((0.7, 1.3, 2.2))
    image.SetOrigin((1000.123456, -85.4321, 3.14))
    path = tmp_path / "ct.nrrd"
    sitk.WriteImage(image, str(path))
    return path


@pytest.fixture
def artifact(tmp_path: Path) -> ResolvedArtifact:
    path = tmp_path / "captured-version"
    path.mkdir()
    return ResolvedArtifact(
        ArtifactIdentity("selected-model", "7", "models:/selected-model/7"), path
    )


def _script_executor(
    tmp_path: Path, body: str, timeout: float = 30, environment: dict[str, str] | None = None
) -> LocalProcessExecutor:
    worker = tmp_path / "worker.py"
    worker.write_text(body)
    return LocalProcessExecutor(
        ProcessConfig(
            mode="process",
            command=(sys.executable, str(worker)),
            timeout_seconds=timeout,
            environment=environment or {},
        )
    )


def _smoke_service(tmp_path: Path, executor=None) -> InferenceService:
    return InferenceService(smoke_deployment(executor), tmp_path / "requests")


def _model_o_parameters() -> ModelOParameters:
    return ModelOParameters.model_validate({"polylines_zyx": [[[1.25, 1.0, 1.0], [2.0, 1.0, 1.0]]]})


def test_output_survives_request_cleanup(input_path: Path, tmp_path: Path) -> None:
    service = _smoke_service(tmp_path)
    result = service.infer(InferenceRequest(input_path, NoParameters()))
    image = sitk.ReadImage(str(result.path))
    assert np.array_equal(
        sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(sitk.ReadImage(str(input_path))) > 0
    )
    assert image.GetPixelID() == sitk.sitkUInt8
    assert image.GetOrigin() == pytest.approx((1000.123456, -85.4321, 3.14), abs=1e-4)
    assert image.GetSpacing() == pytest.approx((0.7, 1.3, 2.2), abs=1e-6)
    assert image.GetDirection() == pytest.approx(
        sitk.ReadImage(str(input_path)).GetDirection(), abs=1e-6
    )
    assert result.metadata["test_only"] is True
    assert result.metadata["runtime"] == "process"
    assert list(service.work_dir.iterdir()) == [result.path]
    result.path.unlink()
    assert not list(service.work_dir.iterdir())


@pytest.mark.parametrize(
    "contents, message",
    [
        (
            "import sys\nfrom pathlib import Path\nPath(sys.argv[4]).write_bytes(b'partial')\nsys.exit(7)\n",
            "status 7",
        ),
        (
            "import sys\nfrom pathlib import Path\nPath(sys.argv[6]).write_text('{}')\n",
            "result file",
        ),
    ],
)
def test_failed_or_missing_output_is_cleaned(
    contents: str, message: str, input_path: Path, tmp_path: Path
) -> None:
    service = _smoke_service(tmp_path, _script_executor(tmp_path, contents))
    with pytest.raises(ExecutionError, match=message):
        service.infer(InferenceRequest(input_path, NoParameters()))
    assert not list(service.work_dir.iterdir())


def test_missing_model_does_not_start_process(
    input_path: Path, artifact: ResolvedArtifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution.subprocess, "Popen", lambda *a, **_kwargs: pytest.fail("Started process")
    )
    deployment = ModelDeployment(
        ArgWorkflow(),
        FixedArtifactSource(replace(artifact, path=tmp_path / "missing-model")),
        LocalProcessExecutor(ProcessConfig(mode="process", command=("never-start",))),
        NoParameters.model_validate,
    )
    service = InferenceService(deployment, tmp_path / "requests")
    with pytest.raises(ExecutionError, match="bindings"):
        service.infer(InferenceRequest(input_path, NoParameters()))
    assert not list(service.work_dir.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
@pytest.mark.parametrize("parent_exit", ["time.sleep(10)", "sys.exit(7)"])
def test_failed_or_timed_out_parent_stops_descendants(parent_exit: str, tmp_path: Path) -> None:
    child_output = tmp_path / "child-completed"
    child_code = f"import time; from pathlib import Path; time.sleep(0.8); Path({str(child_output)!r}).touch()"
    executor = _script_executor(
        tmp_path,
        f"import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c',{child_code!r}])\n{parent_exit}\n",
        timeout=0.2,
    )
    started = time.monotonic()
    expected = ExecutionTimeout if parent_exit == "time.sleep(10)" else ExecutionError
    with pytest.raises(expected):
        executor.run(Invocation((), ()))
    assert time.monotonic() - started < 5
    time.sleep(1)
    assert not child_output.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin zombie-only killpg EPERM")
def test_mac_zombie_only_group_is_already_stopped() -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    try:
        # Leave the exited child unreaped, so its group still contains a zombie.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = subprocess.run(
                ["ps", "-p", str(process.pid), "-o", "stat="],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            ).stdout.strip()
            if state.startswith("Z"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("Test child did not reach its unreaped zombie state")
        with pytest.raises(PermissionError):
            os.killpg(process.pid, 0)
        execution._stop_process(process)
        assert process.returncode == 0
    finally:
        process.wait(timeout=5)


@pytest.mark.parametrize("snapshot", ["12345 S\n", "invalid state snapshot"])
def test_cleanup_permission_denial_is_only_accepted_for_confirmed_dead_members(
    snapshot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(pid=12345, wait=lambda **_kwargs: 0)
    monkeypatch.setattr(execution.os, "name", "posix")
    monkeypatch.setattr(execution.sys, "platform", "darwin")

    def deny_signal(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(execution.os, "killpg", deny_signal)
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda *a, **_kwargs: subprocess.CompletedProcess(a, 0, snapshot, ""),
    )
    with pytest.raises(ExecutionError, match="Could not .* inference process"):
        execution._stop_process(process)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_runtime_cleanup_runs_even_when_signaling_fails(
    exit_code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanups = []

    def fail_stop(process):
        raise ExecutionError("Signal failed")

    monkeypatch.setattr(execution, "_stop_process", fail_stop)
    with pytest.raises(ExecutionError, match="Signal failed"):
        execution._run_process(
            [sys.executable, "-c", f"import sys; sys.exit({exit_code})"],
            10,
            None,
            cleanup=lambda: cleanups.append("container removed"),
        )
    assert cleanups == ["container removed"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize(
    "config_class, values",
    [
        (ProcessConfig, {"mode": "process", "command": ("python",)}),
        (ContainerConfig, {"mode": "container", "image": "image"}),
    ],
)
def test_timeout_must_be_bounded(config_class, values, timeout: float) -> None:
    with pytest.raises(ValidationError, match="timeout_seconds"):
        config_class(**values, timeout_seconds=timeout)


@pytest.mark.parametrize(
    "executor",
    [
        LocalProcessExecutor(ProcessConfig(mode="process", command=("never-start",))),
        DockerExecutor(ContainerConfig(mode="container", image="image")),
    ],
)
def test_missing_binding_is_rejected_before_launch(
    executor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        execution.subprocess, "Popen", lambda *a, **_kwargs: pytest.fail("Started process")
    )
    invocation = Invocation((FileArgument("input"),), (FileBinding("input", tmp_path / "missing"),))
    with pytest.raises(ExecutionError, match="existing files or directories"):
        executor.run(invocation)


@pytest.mark.skipif(
    not os.environ.get("MONAI_SMOKE_IMAGE"), reason="Built local test image required"
)
def test_local_container_handoff(input_path: Path, tmp_path: Path) -> None:
    image = os.environ["MONAI_SMOKE_IMAGE"]
    service = _smoke_service(
        tmp_path, DockerExecutor(ContainerConfig(mode="container", image=image, timeout_seconds=30))
    )
    result = service.infer(InferenceRequest(input_path, NoParameters()))
    assert result.path.is_file()
    assert result.metadata["runtime"] == "docker" and result.metadata["runtime_reference"] == image
    assert list(service.work_dir.iterdir()) == [result.path]
    result.path.unlink()


def test_docker_translates_bindings_and_cleans_failed_container(
    input_path: Path, artifact: ResolvedArtifact, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = ModelOWorkflow(settings=ModelOSettings(assets=tmp_path))
    request = InferenceRequest(input_path, _model_o_parameters())
    invocation = workflow.prepare(request, artifact, tmp_path)
    captured = {}

    def fail_process(command, timeout, environment, cleanup):
        captured["command"] = command
        assert environment is None
        cleanup()
        raise ExecutionTimeout("deadline")

    def remove(command, **_kwargs):
        captured["remove"] = command
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(execution, "_run_process", fail_process)
    monkeypatch.setattr(execution.subprocess, "run", remove)
    with pytest.raises(ExecutionTimeout):
        DockerExecutor(
            ContainerConfig(
                mode="container",
                image="model-o:configured",
                environment={"ORT_DISABLE_TELEMETRY": "0"},
            )
        ).run(invocation)
    command = captured["command"]
    for binding in invocation.bindings:
        suffix = f"/{binding.path.name}" if binding.path.is_file() else ""
        access = "rw" if binding.writable else "ro"
        assert f"{binding.path}:/bindings/{binding.name}{suffix}:{access}" in command
    assert command[command.index("--parameters") + 1] == "/bindings/parameters/parameters.json"
    assert command[command.index("--assets") + 1] == "/bindings/assets"
    assert command[command.index("--metadata") + 1] == "/bindings/output/metadata.json"
    assert "ORT_DISABLE_TELEMETRY=0" in command
    name = command[command.index("--name") + 1]
    assert name.startswith("inference-")
    assert captured["remove"] == ["docker", "rm", "--force", name]


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, []),
        ({"gpus": "all"}, ["--gpus", "all"]),
        ({"gpus": ("0", "2")}, ["--gpus", '"device=0,2"']),
        (
            {"cdi_devices": ("nvidia.com/gpu=all", "vendor.com/accelerator=0")},
            ["--device", "nvidia.com/gpu=all", "--device", "vendor.com/accelerator=0"],
        ),
        (
            {
                "devices": (
                    DeviceMapping(
                        host=PurePosixPath("/dev/dri/renderD128"),
                        container=PurePosixPath("/dev/dri/renderD129"),
                    ),
                    DeviceMapping(
                        host=PurePosixPath("/dev/kfd"),
                        container=PurePosixPath("/dev/kfd"),
                    ),
                ),
            },
            [
                "--device",
                "/dev/dri/renderD128:/dev/dri/renderD129:rw",
                "--device",
                "/dev/kfd:/dev/kfd:rw",
            ],
        ),
    ],
)
def test_docker_launches_with_only_configured_accelerator_flags(settings, expected, monkeypatch):
    captured = []

    def run_process(command, timeout, environment, cleanup):
        captured.extend(command)

    monkeypatch.setattr(execution, "_run_process", run_process)
    result = DockerExecutor(
        ContainerConfig(mode="container", image="worker-image", **settings)
    ).run(Invocation(("--input", "image.nrrd"), ()))
    assert captured == [
        "docker",
        "run",
        "--rm",
        "--name",
        captured[4],
        "--network",
        "none",
        *expected,
        "worker-image",
        "--input",
        "image.nrrd",
    ]
    assert result.runtime == "docker" and result.reference == "worker-image"


def test_arg_handoff_uses_captured_artifact_and_fixed_policy(
    input_path: Path, artifact: ResolvedArtifact, tmp_path: Path
) -> None:
    executor = _script_executor(
        tmp_path,
        "import argparse,json\nimport SimpleITK as sitk\nfrom pathlib import Path\np=argparse.ArgumentParser()\n"
        "for key in ('input','output','model','device','model-file','metadata'): p.add_argument('--'+key)\n"
        "a=p.parse_args()\nassert a.device=='auto'\nassert a.model_file=='model.onnx'\n"
        f"assert Path(a.model)==Path({str(artifact.path)!r})\n"
        "image=sitk.ReadImage(a.input)\nsitk.WriteImage(sitk.Cast(image>0,sitk.sitkUInt8),a.output)\n"
        "Path(a.metadata).write_text(json.dumps({'runtime_reference':'untrusted','model_version':'untrusted',"
        "'worker':'arg','preprocessing':'native-plans','device_preference':a.device,'model_file':a.model_file,"
        "'providers':['FixtureExecutionProvider']}))\n",
    )
    service = InferenceService(
        ModelDeployment(
            ArgWorkflow(), FixedArtifactSource(artifact), executor, NoParameters.model_validate
        ),
        tmp_path / "requests",
    )
    result = service.infer(InferenceRequest(input_path, NoParameters()))
    assert result.metadata["test_only"] is False and result.metadata["worker"] == "arg"
    assert result.metadata["preprocessing"] == "native-plans"
    assert result.metadata["device_preference"] == "auto"
    assert result.metadata["model_file"] == "model.onnx"
    assert result.metadata["providers"] == ["FixtureExecutionProvider"]
    assert result.metadata["model_version"] == "7"
    assert result.metadata["runtime_reference"] == " ".join(executor.config.command)
    assert list(service.work_dir.iterdir()) == [result.path]
    result.path.unlink()


def test_model_o_stages_polylines_and_executor_supplies_runtime_environment(
    input_path: Path, artifact: ResolvedArtifact, tmp_path: Path
) -> None:
    python_path = tmp_path / "python-modules"
    python_path.mkdir()
    (python_path / "fixture_marker.py").write_text("VALUE = 42\n")
    executor = _script_executor(
        tmp_path,
        "import argparse,json,os\nfrom pathlib import Path\nimport SimpleITK as sitk\nimport fixture_marker\n"
        "p=argparse.ArgumentParser()\nfor key in ('input','output','model','metadata','parameters','assets','queue-depth'): p.add_argument('--'+key)\n"
        "a=p.parse_args()\nassert fixture_marker.VALUE==42\n"
        f"assert json.loads(Path(a.parameters).read_text())=={_model_o_parameters().model_dump(mode='json')!r}\n"
        f"assert Path(a.assets)==Path({str(tmp_path)!r})\n"
        "image=sitk.ReadImage(a.input)\nsitk.WriteImage(sitk.Cast(image>0,sitk.sitkUInt8),a.output)\n"
        "Path(a.metadata).write_text(json.dumps({'worker':'model_o','fixture_marker':7,'telemetry_disabled':os.environ['ORT_DISABLE_TELEMETRY']}))\n",
        environment={"PYTHONPATH": str(python_path), "ORT_DISABLE_TELEMETRY": "0"},
    )
    service = InferenceService(
        ModelDeployment(
            ModelOWorkflow(settings=ModelOSettings(assets=tmp_path)),
            FixedArtifactSource(artifact),
            executor,
            ModelOParameters.model_validate,
        ),
        tmp_path / "requests",
    )
    result = service.infer(InferenceRequest(input_path, _model_o_parameters()))
    assert result.metadata["fixture_marker"] == 7 and result.metadata["worker"] == "model_o"
    assert result.metadata["test_only"] is False and result.metadata["telemetry_disabled"] == "0"
    assert list(service.work_dir.iterdir()) == [result.path]
    result.path.unlink()


@pytest.mark.parametrize("model", ["arg", "model_o"])
def test_workflow_preserves_worker_metadata(model: str, tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "mask.nii.gz").write_bytes(b"fixture mask")
    metadata = {"provider": "recorded-provider", "test_only": True}
    if model == "arg":
        metadata.update(
            worker="arg",
            preprocessing="native-plans",
            device_preference="cpu",
            model_file="cpu.onnx",
        )
        workflow = ArgWorkflow()
        parameters = NoParameters()
    else:
        metadata.update(model_file="modelo.onnx")
        workflow = ModelOWorkflow(settings=ModelOSettings(assets=tmp_path))
        parameters = _model_o_parameters()
    (output / "metadata.json").write_text(json.dumps(metadata))

    result = workflow.read_result(InferenceRequest(tmp_path / "ct.nii.gz", parameters), tmp_path)

    assert result.path == output / "mask.nii.gz"
    assert result.metadata == {**metadata, "test_only": False}


def test_failure_logs_only_bounded_stderr_tail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    executor = _script_executor(
        tmp_path, "import sys\nsys.stderr.write('FIRST'+('x'*20000)+'LAST')\nsys.exit(7)\n"
    )
    with pytest.raises(ExecutionError, match="status 7"):
        executor.run(Invocation((), ()))
    assert "LAST" in caplog.text and "FIRST" not in caplog.text
    assert len(caplog.text) < execution.STDERR_TAIL_BYTES + 500


def test_worker_receives_original_input_without_conversion(
    input_path: Path, tmp_path: Path
) -> None:
    workspace = tmp_path / "request"
    workspace.mkdir()
    invocation = prepare_worker_files(input_path, workspace)
    image_binding, output_binding = invocation.bindings
    assert image_binding.path == input_path
    assert not image_binding.writable
    assert output_binding.path == workspace / "output"
    assert output_binding.writable
    assert list(workspace.iterdir()) == [output_binding.path]


@pytest.mark.parametrize("contents", ["[]", "not json"])
def test_invalid_result_metadata_is_rejected(contents: str, tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "metadata.json").write_text(contents)
    with pytest.raises(ExecutionError, match="metadata"):
        read_worker_result(tmp_path)


def test_model_o_invalid_polylines_rejected_before_artifact_resolution(
    input_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = replace(
        smoke_deployment(),
        workflow=ModelOWorkflow(settings=ModelOSettings(assets=tmp_path)),
        parse_parameters=ModelOParameters.model_validate,
    )
    monkeypatch.setattr(
        type(deployment.artifacts), "resolve", lambda _: pytest.fail("Resolved artifact")
    )
    service = InferenceService(deployment, tmp_path / "requests")
    data = _model_o_parameters().model_dump(mode="json")
    data["polylines_zyx"][0][0][0] = float("nan")
    with pytest.raises(ValidationError):
        service.infer(InferenceRequest(input_path, deployment.parse_parameters(data)))
    assert not service.work_dir.exists()


@pytest.mark.parametrize("schema", [NoParameters, ModelOParameters])
@pytest.mark.parametrize(
    "key",
    [
        "device",
        "gpus",
        "cdi_devices",
        "devices",
        "model_file",
        "model_path",
        "worker_image",
        "tracking_uri",
        "preprocessing",
    ],
)
def test_client_cannot_select_model_or_runtime_policy(schema, key: str) -> None:
    parameters = _model_o_parameters().model_dump(mode="json") if schema is ModelOParameters else {}
    with pytest.raises(ValidationError) as failure:
        schema.model_validate({**parameters, key: "caller-selected"})
    assert any(
        error["type"] == "extra_forbidden" and error["loc"] == (key,)
        for error in failure.value.errors()
    )


@pytest.mark.parametrize("model_file", ["", ".", "..", "../model.onnx", "/model.onnx"])
def test_arg_model_file_cannot_escape_artifact(model_file: str) -> None:
    with pytest.raises(ValidationError) as failure:
        ArgPolicy(model_file=model_file)
    assert failure.value.errors()[0]["loc"] == ("model_file",)
