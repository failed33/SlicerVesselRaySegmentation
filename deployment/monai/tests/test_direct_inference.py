"""Direct Label inference and the isolated CLI share one ARG predictor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from monai_service.application import InferenceService
from monai_service.contracts import (
    ArtifactIdentity,
    ExecutionError,
    InferenceRequest,
    ModelDeployment,
    ResolvedArtifact,
)
from monai_service.models.arg import ArgPolicy, ArgWorkflow
from monai_service.schemas.base import NoParameters

from .fixture_factory import make_arg_case
from .fixtures import FixedArtifactSource


def test_direct_arg_and_portable_cli_use_the_same_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_arg_case(tmp_path)
    artifact = ResolvedArtifact(
        ArtifactIdentity("synthetic-arg", "1", "fixture:arg"), case.model_dir
    )
    workflow = ArgWorkflow(policy=ArgPolicy(device="cpu"))

    def unexpected_worker_call(*args: object) -> None:
        pytest.fail("Direct inference called the isolated worker path")

    monkeypatch.setattr(ArgWorkflow, "prepare", unexpected_worker_call)
    monkeypatch.setattr(ArgWorkflow, "read_result", unexpected_worker_call)
    service = InferenceService(
        ModelDeployment(workflow, FixedArtifactSource(artifact), None, NoParameters.model_validate),
        tmp_path / "requests",
    )
    result = service.infer(InferenceRequest(case.image, NoParameters()))
    direct = sitk.ReadImage(str(result.path))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(direct), case.expected)
    assert result.metadata["runtime"] == "direct"
    assert result.metadata["runtime_reference"] == "python"
    assert list(service.work_dir.iterdir()) == [result.path]

    cli_output, cli_metadata = tmp_path / "cli-mask.nii.gz", tmp_path / "cli-metadata.json"
    project = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "ORT_DISABLE_TELEMETRY": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(project), str(project.parent / "inference/src"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "monai_service.models.arg",
            "--input",
            str(case.image),
            "--model",
            str(case.model_dir),
            "--output",
            str(cli_output),
            "--metadata",
            str(cli_metadata),
            "--device",
            "cpu",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    isolated = sitk.ReadImage(str(cli_output))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(isolated), case.expected)
    for getter in ("GetSize", "GetOrigin", "GetSpacing", "GetDirection"):
        assert getattr(direct, getter)() == getattr(isolated, getter)()
    for key, value in json.loads(cli_metadata.read_text()).items():
        assert result.metadata[key] == value
    result.path.unlink()
    assert not list(service.work_dir.iterdir())


@pytest.mark.parametrize("failure_type", [ValueError, RuntimeError, OSError])
def test_direct_failure_removes_request_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_type: type[Exception]
) -> None:
    artifact = ResolvedArtifact(ArtifactIdentity("synthetic-arg", "1", "fixture:arg"), tmp_path)

    def fail_after_partial_output(
        self: ArgWorkflow, request: object, artifact: object, workspace: Path
    ) -> None:
        (workspace / "partial.nii.gz").write_bytes(b"partial")
        raise failure_type("Inference failed after creating output")

    monkeypatch.setattr(ArgWorkflow, "infer", fail_after_partial_output)
    service = InferenceService(
        ModelDeployment(
            ArgWorkflow(), FixedArtifactSource(artifact), None, NoParameters.model_validate
        ),
        tmp_path / "requests",
    )
    with pytest.raises(ExecutionError, match="Inference failed after creating output") as error:
        service.infer(InferenceRequest(tmp_path / "input.nii.gz", NoParameters()))
    assert isinstance(error.value.__cause__, failure_type)
    assert not list(service.work_dir.iterdir())
