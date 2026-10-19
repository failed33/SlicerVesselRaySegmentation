"""Offline releases preserve MLflow identity and request-local version paths."""

import shutil
from pathlib import Path

import pytest
from model_artifacts import SelectedModel
from model_artifacts.snapshot import read_snapshot, write_snapshot

from monai_service.adapters.directory import DirectoryArtifactSource
from monai_service.contracts import ArtifactUnavailable
from monai_service.models.arg import TASK
from monai_service.registry import from_env


def snapshot(path: Path, version: str) -> Path:
    path.mkdir()
    (path / "model.onnx").write_bytes(b"selected ONNX graph")
    write_snapshot(
        path,
        SelectedModel("arg-onnx", version, "https://tracking.example", "https://registry.example"),
    )
    return path


def test_sync_update_keeps_the_inflight_artifact_and_reports_exact_versions(tmp_path):
    first = snapshot(tmp_path / "one", "7")
    second = snapshot(tmp_path / "two", "8")
    current = tmp_path / "current"
    current.symlink_to(first, target_is_directory=True)
    source = DirectoryArtifactSource(current)
    inflight = source.resolve()
    replacement = tmp_path / "replacement"
    replacement.symlink_to(second, target_is_directory=True)
    replacement.replace(current)
    updated = source.resolve()
    assert inflight.path == first
    assert updated.path == second
    assert (inflight.identity.version, updated.identity.version) == ("7", "8")
    assert updated.identity.uri == "models:/arg-onnx/8"
    assert read_snapshot(second)["model"]["registry_uri"] == "https://registry.example"


@pytest.mark.parametrize("change", ["corruption", "extra", "missing", "symlink"])
def test_incomplete_or_modified_transfers_are_rejected(tmp_path, change):
    path = snapshot(tmp_path / "model", "7")
    if change == "corruption":
        (path / "model.onnx").write_bytes(b"incomplete update")
    elif change == "extra":
        (path / "untracked.onnx").write_bytes(b"unexpected")
    elif change == "missing":
        (path / "model.onnx").unlink()
    else:
        shutil.move(path / "model.onnx", tmp_path / "outside.onnx")
        (path / "model.onnx").symlink_to(tmp_path / "outside.onnx")
    with pytest.raises(ArtifactUnavailable):
        DirectoryArtifactSource(path).resolve()


def test_local_model_configuration_needs_no_registry_settings_or_request(tmp_path, monkeypatch):
    path = snapshot(tmp_path / "model", "7")
    monkeypatch.setenv("MONAI_SERVICE_ENABLE_ARG", "true")
    monkeypatch.setenv("MONAI_SERVICE_ARG_MODEL_DIR", str(path))
    for name in (
        "MONAI_SERVICE_ARG_MODEL_NAME",
        "MONAI_SERVICE_ARG_MODEL_ALIAS",
        "MONAI_SERVICE_MLFLOW_TRACKING_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = from_env((TASK,), default_work_dir=tmp_path)
    artifact = settings.registrations["arg_vessel"].artifacts.resolve()
    assert artifact.identity.version == "7"
