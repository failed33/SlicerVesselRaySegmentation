"""Local model paths retain MLflow version numbers without registry access."""

from monai_service.adapters.directory import DirectoryArtifactSource
from monai_service.models.arg import TASK
from monai_service.registry import from_env


def model(root, version):
    path = root / "arg-onnx" / version
    path.mkdir(parents=True)
    return path


def test_sync_update_keeps_the_inflight_artifact_and_reports_exact_versions(tmp_path):
    first = model(tmp_path, "7")
    second = model(tmp_path, "8")
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


def test_local_model_configuration_needs_no_registry_settings_or_request(tmp_path, monkeypatch):
    path = model(tmp_path, "7")
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
