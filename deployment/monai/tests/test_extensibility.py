"""A third deployment uses native HTTP without changing shared production code."""

import json
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient
from monailabel.config import settings
from monailabel.endpoints.user.auth import User, get_current_user
from monailabel.interfaces.utils.app import clear_cache
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_core import PydanticCustomError

from monai_service.adapters.execution import LocalProcessExecutor
from monai_service.app import create_app
from monai_service.config import ServiceSettings
from monai_service.contracts import (
    ArtifactIdentity,
    FileArgument,
    FileBinding,
    InferenceRequest,
    InferenceResult,
    Invocation,
    ModelDeployment,
    ModelDescriptor,
    ResolvedArtifact,
)
from monai_service.schemas.execution import ProcessConfig
from monai_service.worker_io import prepare_worker_files, read_worker_result


class FixedArtifactSource:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.calls = 0

    def resolve(self) -> ResolvedArtifact:
        self.calls += 1
        return ResolvedArtifact(
            ArtifactIdentity("three-label-example", "7", "example:7"), self.directory
        )


class ThreeLabelParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_map: Literal["labels.nii.gz"] = "labels.nii.gz"

    @field_validator("label_map", mode="before")
    @classmethod
    def known_label_map(cls, value: object) -> object:
        if value != "labels.nii.gz":
            # Deliberately includes the value: the hosting adapter must not echo
            # raw inputs from custom validator messages or exception contexts.
            if isinstance(value, str) and value.startswith("custom:"):
                raise PydanticCustomError(
                    "label_map_unavailable", "Unsupported label map: {value}", {"value": value}
                )
            raise ValueError(f"Unsupported label map: {value}")
        return value


class ThreeLabelWorkflow:
    descriptor = ModelDescriptor(
        "Three-label example",
        "Extensibility acceptance fixture",
        {"background": 0, "a": 1, "b": 2},
        3,
    )

    def prepare(
        self,
        request: InferenceRequest[ThreeLabelParameters],
        artifact: ResolvedArtifact,
        workspace: Path,
    ) -> Invocation:
        prepare_worker_files(request.image, workspace)
        (workspace / "output" / "metadata.json").write_text('{"test_only": true}')
        return Invocation(
            (
                FileArgument("model", PurePosixPath(request.parameters.label_map)),
                FileArgument("result", PurePosixPath("mask.nii.gz")),
            ),
            (
                FileBinding("model", artifact.path),
                FileBinding("result", workspace / "output", writable=True),
            ),
        )

    def read_result(
        self, request: InferenceRequest[ThreeLabelParameters], workspace: Path
    ) -> InferenceResult:
        return read_worker_result(workspace)


@pytest.fixture
def third_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, FixedArtifactSource, Path]]:
    monkeypatch.setattr(settings, "MONAI_LABEL_STUDIES", str(tmp_path / "studies"))
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSION_PATH", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "MONAI_LABEL_DATASTORE_AUTO_RELOAD", False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    labels = np.arange(60, dtype=np.uint8).reshape(3, 4, 5) % 3
    image = sitk.GetImageFromArray(labels)
    image.SetSpacing((0.7, 1.1, 2.3))
    image.SetOrigin((20, -40, 60))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    artifact_path = model_dir / "labels.nii.gz"
    sitk.WriteImage(image, str(artifact_path))
    source = FixedArtifactSource(model_dir)
    deployment = ModelDeployment(
        ThreeLabelWorkflow(),
        source,
        LocalProcessExecutor(ProcessConfig(mode="process", command=("/bin/cp",))),
        ThreeLabelParameters.model_validate,
    )
    app = create_app(ServiceSettings(tmp_path / "service", {"third-model": deployment}))
    app.dependency_overrides[get_current_user] = lambda: User(
        username="test",
        roles=[
            settings.MONAI_LABEL_AUTH_ROLE_ADMIN,
            settings.MONAI_LABEL_AUTH_ROLE_USER,
            settings.MONAI_LABEL_AUTH_ROLE_ANNOTATOR,
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, source, artifact_path
    finally:
        client.close()
        app.dependency_overrides.clear()
        clear_cache()


def test_third_multiclass_workflow_runs_through_native_api_with_non_python_command(
    third_client: tuple[TestClient, FixedArtifactSource, Path], tmp_path: Path
) -> None:
    client, source, artifact_path = third_client
    reference = sitk.ReadImage(str(artifact_path))
    info = client.get("/info/")
    assert info.status_code == 200, info.text
    descriptor = info.json()["models"]["third-model"]
    assert descriptor["labels"] == {"background": 0, "a": 1, "b": 2}
    assert descriptor["config"]["input_profile"] == "image"
    response = client.post(
        "/infer/third-model",
        params={"output": "image"},
        data={"params": json.dumps({"label_map": "labels.nii.gz"})},
        files={"file": ("ct.nii.gz", artifact_path.read_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    assert source.calls == 1
    result_path = tmp_path / "received.nii.gz"
    result_path.write_bytes(response.content)
    result = sitk.ReadImage(str(result_path))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(reference))
    np.testing.assert_allclose(result.GetOrigin(), reference.GetOrigin())
    np.testing.assert_allclose(result.GetSpacing(), reference.GetSpacing())
    np.testing.assert_allclose(result.GetDirection(), reference.GetDirection())
    assert not list((tmp_path / "service" / "requests").rglob("inference-*"))


@pytest.mark.parametrize(
    ("label_map", "error_type"),
    [
        ("https://user:secret-password@private", "value_error"),
        ("custom:https://user:secret-password@private", "label_map_unavailable"),
    ],
)
def test_custom_parameter_validation_does_not_echo_inputs_or_resolve_artifacts(
    third_client: tuple[TestClient, FixedArtifactSource, Path],
    label_map: str,
    error_type: str,
) -> None:
    client, source, artifact_path = third_client
    response = client.post(
        "/infer/third-model",
        data={"params": json.dumps({"label_map": label_map})},
        files={"file": ("ct.nii.gz", artifact_path.read_bytes())},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == [
        {
            "type": error_type,
            "loc": ["body", "params", "label_map"],
            "msg": "Invalid parameter value",
        }
    ]
    assert "secret-password" not in response.text
    assert source.calls == 0
