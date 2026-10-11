"""Concrete deployment configuration stays outside generic serving settings."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from monai_service.adapters.execution import DockerExecutor, LocalProcessExecutor
from monai_service.adapters.mlflow import ModelSource
from vessel_deployment import from_env


def enable_model(monkeypatch: pytest.MonkeyPatch, kind: str, mode: str = "process") -> None:
    for key, value in {
        f"MONAI_SERVICE_ENABLE_{kind}": "true",
        "MONAI_SERVICE_MLFLOW_TRACKING_URI": "http://tracking",
        "MONAI_SERVICE_MLFLOW_REGISTRY_URI": "http://registry",
        f"MONAI_SERVICE_{kind}_MODEL_NAME": "ConfiguredModel",
        f"MONAI_SERVICE_{kind}_MODEL_ALIAS": "accepted",
        f"MONAI_SERVICE_{kind}_WORKER__MODE": mode,
    }.items():
        monkeypatch.setenv(key, value)
    if mode == "process":
        monkeypatch.setenv(
            f"MONAI_SERVICE_{kind}_WORKER__COMMAND",
            json.dumps([sys.executable, "-m", f"monai_service.models.{kind.lower()}"]),
        )


def test_disabled_models_do_not_require_source_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONAI_SERVICE_MLFLOW_TRACKING_URI", "http://unused")
    assert not from_env().registrations


@pytest.mark.parametrize("kind", ["ARG", "MODEL_O"])
def test_incomplete_configuration_fails_startup(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"MONAI_SERVICE_ENABLE_{kind}", "true")
    with pytest.raises(ValidationError) as caught:
        from_env()
    assert {error["loc"] for error in caught.value.errors()} == {
        ("model_name",),
        ("model_alias",),
        ("MONAI_SERVICE_MLFLOW_TRACKING_URI",),
    }


def test_arg_registration_captures_explicit_source_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_model(monkeypatch, "ARG", "container")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__IMAGE", "worker@sha256:configured-digest")
    registration = from_env().registrations["arg_vessel"]
    assert registration.artifacts.config == ModelSource(
        "ConfiguredModel", "accepted", "http://tracking", "http://registry"
    )
    assert isinstance(registration.executor, DockerExecutor)
    assert registration.executor.config.image == "worker@sha256:configured-digest"
    assert registration.workflow.policy.device == "auto"
    assert registration.workflow.policy.model_file == "model.onnx"
    assert registration.workflow.descriptor.labels == {"background": 0, "vessel": 1}


@pytest.mark.parametrize("timeout", ["0", "-1", "inf", "nan"])
def test_deadline_is_finite_at_startup(timeout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__TIMEOUT_SECONDS", timeout)
    with pytest.raises(ValidationError, match="timeout_seconds"):
        from_env()


def test_container_requires_explicit_image(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_model(monkeypatch, "ARG", "container")
    with pytest.raises(ValidationError, match=r"container\.image"):
        from_env()


def test_process_requires_explicit_command(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.delenv("MONAI_SERVICE_ARG_WORKER__COMMAND")
    with pytest.raises(ValidationError, match=r"process\.command"):
        from_env()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DEVICE", "tpu"),
        ("DEVICE", ""),
        ("MODEL_FILE", "../model.onnx"),
        ("MODEL_FILE", ".."),
        ("MODEL_FILE", ""),
    ],
)
def test_invalid_model_policy_fails_startup(
    key: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv(f"MONAI_SERVICE_ARG_{key}", value)
    with pytest.raises(ValueError):
        from_env()


@pytest.mark.parametrize("telemetry", [None, "0"])
def test_model_o_runtime_environment_is_server_owned(
    telemetry: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_model(monkeypatch, "MODEL_O")
    monkeypatch.setenv("MONAI_SERVICE_MODEL_O_ASSETS", str(tmp_path))
    python_path = tmp_path / "installed-module"
    python_path.mkdir()
    worker_environment = {"PYTHONPATH": str(python_path), "ORT_DISABLE_TELEMETRY": "1"}
    monkeypatch.setenv("MONAI_SERVICE_MODEL_O_WORKER__ENVIRONMENT", json.dumps(worker_environment))
    if telemetry is None:
        monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    else:
        monkeypatch.setenv("ORT_DISABLE_TELEMETRY", telemetry)
    registration = from_env().registrations["model_o"]
    assert isinstance(registration.executor, LocalProcessExecutor)
    assert registration.executor.config.environment == worker_environment
    assert registration.workflow.descriptor.input_profile == "polylines"
    assert registration.workflow.settings.assets == tmp_path


def test_process_command_can_select_an_external_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__COMMAND", '["custom-inference", "--configured"]')
    registration = from_env().registrations["arg_vessel"]
    assert registration.executor.config.command == ("custom-inference", "--configured")


@pytest.mark.parametrize("value", ["[]", '[" "]', "null"])
def test_invalid_command_configuration_fails_startup(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__COMMAND", value)
    with pytest.raises(ValidationError, match="command"):
        from_env()


@pytest.mark.parametrize(
    "value",
    ['{"": "x"}', '{"A=B": "x"}'],
)
def test_invalid_environment_configuration_fails_startup(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__ENVIRONMENT", value)
    with pytest.raises(ValidationError, match="env"):
        from_env()


def test_container_does_not_accept_process_command(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_model(monkeypatch, "ARG", "container")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__IMAGE", "configured-image")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__COMMAND", '["unused-command"]')
    with pytest.raises(ValidationError, match=r"container\.command"):
        from_env()


def test_model_o_does_not_discover_a_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_model(monkeypatch, "MODEL_O")
    monkeypatch.setenv("MONAI_SERVICE_MODEL_O_ASSETS", str(tmp_path))
    registration = from_env().registrations["model_o"]
    assert "PYTHONPATH" not in registration.executor.config.environment


@pytest.mark.parametrize("mode", ["python", "unknown", "", "null"])
def test_unknown_worker_mode_fails_startup(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    enable_model(monkeypatch, "ARG", mode)
    with pytest.raises(ValidationError, match="mode"):
        from_env()


@pytest.mark.parametrize(
    "value, expected", [("1", True), ("yes", True), ("0", False), ("no", False)]
)
def test_enable_flags_use_native_boolean_values(
    value: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv("MONAI_SERVICE_ENABLE_ARG", value)
    assert bool(from_env().registrations) is expected


@pytest.mark.parametrize("value", ["", "null"])
def test_invalid_enable_flags_fail_startup(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONAI_SERVICE_ENABLE_ARG", value)
    with pytest.raises(ValidationError, match="valid boolean"):
        from_env()


@pytest.mark.parametrize("image", ["configured-image", "", " "])
def test_process_does_not_accept_container_image(
    image: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv("MONAI_SERVICE_ARG_WORKER__IMAGE", image)
    with pytest.raises(ValidationError, match="image"):
        from_env()


def test_disabled_models_do_not_parse_their_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONAI_SERVICE_ARG_DEVICE", "invalid")
    monkeypatch.setenv("MONAI_SERVICE_MODEL_O_WORKER__COMMAND", "not-json")
    assert not from_env().registrations


def test_empty_registry_uri_falls_back_to_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_model(monkeypatch, "ARG")
    monkeypatch.setenv("MONAI_SERVICE_MLFLOW_REGISTRY_URI", "")
    assert from_env().registrations["arg_vessel"].artifacts.config.registry_uri is None
