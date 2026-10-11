"""Execution settings normalize native sources and reject contradictory variants."""

import json
from pathlib import Path, PurePosixPath

import pytest
from pydantic import TypeAdapter, ValidationError

from monai_service.adapters.executor_config import executor_config_from_env
from monai_service.schemas.execution import (
    ContainerConfig,
    DeviceMapping,
    DirectConfig,
    ExecutorConfig,
    KubernetesConfig,
    ProcessConfig,
)


def test_missing_worker_defaults_to_direct_but_explicit_null_is_invalid(monkeypatch):
    monkeypatch.delenv("MONAI_TEST_WORKER", raising=False)
    assert executor_config_from_env("MONAI_TEST") == DirectConfig()
    monkeypatch.setenv("MONAI_TEST_WORKER", "null")
    with pytest.raises(ValidationError, match="worker"):
        executor_config_from_env("MONAI_TEST")


def _values(mode):
    if mode == "process":
        return {"mode": mode, "command": ["worker"]}
    if mode == "container":
        return {"mode": mode, "image": "worker:cpu"}
    return {
        "mode": mode,
        "job_template": "/etc/monai/arg-job.yaml",
        "namespace": "monai",
        "pvc_claim": "workspace",
        "shared_root": "/workspace",
    }


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "process", "command": ["worker"], "image": "image"},
        {"mode": "process", "command": ["worker"], "gpus": "all"},
        {"mode": "process", "command": ["worker"], "devices": []},
        {"mode": "container", "image": "image", "command": ["worker"]},
        {"mode": "process"},
        {"mode": "container"},
        {"mode": "kubernetes"},
        {"mode": "unknown"},
    ],
)
def test_variants_reject_incompatible_or_missing_fields(values):
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutorConfig).validate_python(values)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("mode", ["process", "container", "kubernetes"])
def test_timeout_is_finite_and_positive(timeout, mode):
    with pytest.raises(ValidationError, match="timeout_seconds"):
        TypeAdapter(ExecutorConfig).validate_python({**_values(mode), "timeout_seconds": timeout})


@pytest.mark.parametrize("command", [[], [1], [True], [" "], ["worker\x00suffix"], "worker"])
def test_argv_is_a_nonempty_sequence_of_valid_strings(command):
    with pytest.raises(ValidationError, match="command"):
        ProcessConfig(mode="process", command=command)


def test_json_style_configuration_normalizes_sequences_paths_and_numbers():
    process = ProcessConfig(mode="process", command=["worker", "--option"], timeout_seconds="12.5")
    assert process.command == ("worker", "--option")
    assert process.timeout_seconds == 12.5
    container = ContainerConfig(
        mode="container", image="worker", devices=[{"host": "/dev/dri", "container": "/dev/dri"}]
    )
    assert container.devices == (
        DeviceMapping(host=PurePosixPath("/dev/dri"), container=PurePosixPath("/dev/dri")),
    )


@pytest.mark.parametrize(
    "environment", [{"VALUE": 1}, {1: "value"}, {"": "value"}, {"A=B": "value"}, {"A": "a\x00b"}]
)
def test_environment_contains_only_os_compatible_strings(environment):
    with pytest.raises(ValidationError, match="environment"):
        ContainerConfig(mode="container", image="image", environment=environment)


@pytest.mark.parametrize(
    "gpus",
    [
        "all",
        ["0", "2"],
        ["GPU-3a23c669-1f69-c64e-cf85-44e9b07e7a2a"],
        ["MIG-GPU-3a23c669-1f69-c64e-cf85-44e9b07e7a2a/1/0"],
    ],
)
def test_gpu_requests_are_all_or_chosen_string_identifiers(gpus):
    config = ContainerConfig(mode="container", image="image", gpus=gpus)
    assert config.gpus == (gpus if isinstance(gpus, str) else tuple(gpus))


@pytest.mark.parametrize(
    "gpus",
    [2, True, "2", "device=0", [], [0], [True], [""], ["0,1"], ["--privileged"], ["GPU-x\x00"]],
)
def test_gpu_requests_reject_counts_and_docker_option_syntax(gpus):
    with pytest.raises(ValidationError, match="gpus"):
        ContainerConfig(mode="container", image="image", gpus=gpus)


@pytest.mark.parametrize(
    "device", ["nvidia.com/gpu=all", "nvidia.com/gpu=0", "vendor/class=device:1"]
)
def test_cdi_devices_are_qualified_names(device):
    config = ContainerConfig(mode="container", image="image", cdi_devices=[device])
    assert config.cdi_devices == (device,)


@pytest.mark.parametrize(
    "device",
    [
        "all",
        "/dev/nvidia0",
        "nvidia.com/gpu",
        "nvidia.com/gpu=",
        "--privileged",
        "nvidia.com/gpu=all\x00",
    ],
)
def test_cdi_devices_reject_paths_and_unqualified_names(device):
    with pytest.raises(ValidationError, match="cdi_devices"):
        ContainerConfig(mode="container", image="image", cdi_devices=[device])


def test_cdi_and_legacy_gpu_requests_cannot_be_combined():
    with pytest.raises(ValidationError, match="either gpus or cdi_devices"):
        ContainerConfig(
            mode="container", image="image", gpus="all", cdi_devices=["nvidia.com/gpu=all"]
        )


def test_nested_environment_selects_cdi_without_legacy_gpu_flags(monkeypatch):
    monkeypatch.setenv("MONAI_TEST_WORKER__MODE", "container")
    monkeypatch.setenv("MONAI_TEST_WORKER__IMAGE", "worker:cuda")
    monkeypatch.setenv("MONAI_TEST_WORKER__GPUS", "null")
    monkeypatch.setenv("MONAI_TEST_WORKER__CDI_DEVICES", '["nvidia.com/gpu=all"]')
    config = executor_config_from_env("MONAI_TEST")
    assert config.gpus is None
    assert config.cdi_devices == ("nvidia.com/gpu=all",)


@pytest.mark.parametrize("field", ["host", "container"])
@pytest.mark.parametrize(
    "path", ["dev/dri", "/tmp/dri", "/dev", "/dev/../etc/passwd", "/dev/dri:rw", "/dev/dri\x00"]
)
def test_device_mapping_paths_are_native_absolute_device_paths(field, path):
    values = {"host": "/dev/dri/renderD128", "container": "/dev/dri/renderD128", field: path}
    with pytest.raises(ValidationError, match=field):
        DeviceMapping.model_validate_json(json.dumps(values))


@pytest.mark.parametrize("mode", ["process", "container", "kubernetes"])
def test_one_worker_json_selects_the_native_discriminated_variant(mode, monkeypatch):
    monkeypatch.setenv("MONAI_TEST_WORKER", json.dumps(_values(mode)))
    config = executor_config_from_env("MONAI_TEST")
    assert config.mode == mode
    if isinstance(config, KubernetesConfig):
        assert config.job_template == Path("/etc/monai/arg-job.yaml")
        assert config.shared_root == Path("/workspace")


def test_nested_environment_parses_and_overrides_worker_json(monkeypatch):
    monkeypatch.setenv("MONAI_TEST_WORKER", json.dumps(_values("container")))
    monkeypatch.setenv("MONAI_TEST_WORKER__IMAGE", "worker:cuda")
    monkeypatch.setenv("MONAI_TEST_WORKER__TIMEOUT_SECONDS", "42.5")
    monkeypatch.setenv("MONAI_TEST_WORKER__GPUS", '["0","2"]')
    monkeypatch.setenv("MONAI_TEST_WORKER__ENVIRONMENT", '{"EMPTY":"","BACKEND":"cuda"}')
    config = executor_config_from_env("MONAI_TEST")
    assert isinstance(config, ContainerConfig)
    assert config.image == "worker:cuda"
    assert config.timeout_seconds == 42.5
    assert config.gpus == ("0", "2")
    assert config.environment == {"EMPTY": "", "BACKEND": "cuda"}


def test_process_nested_environment_requires_explicit_command(monkeypatch):
    monkeypatch.setenv("MONAI_TEST_WORKER__MODE", "process")
    with pytest.raises(ValidationError, match="command"):
        executor_config_from_env("MONAI_TEST")
    monkeypatch.setenv("MONAI_TEST_WORKER__COMMAND", '["python","-m","native.worker"]')
    assert executor_config_from_env("MONAI_TEST").command == ("python", "-m", "native.worker")


@pytest.mark.parametrize(
    "field,value", [("GPUS", '"all"'), ("DEVICES", "[]"), ("JOB_TEMPLATE", "/etc/job.yaml")]
)
def test_nested_process_environment_rejects_incompatible_fields(field, value, monkeypatch):
    monkeypatch.setenv("MONAI_TEST_WORKER__MODE", "process")
    monkeypatch.setenv("MONAI_TEST_WORKER__COMMAND", '["worker"]')
    monkeypatch.setenv(f"MONAI_TEST_WORKER__{field}", value)
    with pytest.raises(ValidationError, match=field.lower()):
        executor_config_from_env("MONAI_TEST")


@pytest.mark.parametrize("field,value", [("environment", None), ("gpus", []), ("devices", {})])
def test_worker_json_rejects_malformed_fields(field, value, monkeypatch):
    monkeypatch.setenv("MONAI_TEST_WORKER", json.dumps({**_values("container"), field: value}))
    with pytest.raises(ValidationError, match=field):
        executor_config_from_env("MONAI_TEST")


@pytest.mark.parametrize(
    "field,value",
    [
        ("shared_root", "relative"),
        ("shared_root", "/workspace/../outside"),
        ("mount_path", "relative"),
        ("image", "worker"),
        ("resources", {}),
        ("tolerations", []),
        ("run_as_user", 1000),
        ("environment", {}),
    ],
)
def test_kubernetes_configuration_contains_only_executor_owned_fields(field, value):
    with pytest.raises(ValidationError, match=field):
        TypeAdapter(ExecutorConfig).validate_python({**_values("kubernetes"), field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_template", "/etc/monai/job.yaml"),
        ("namespace", "monai"),
        ("pvc_claim", "workspace"),
        ("shared_root", "/workspace"),
    ],
)
def test_kubernetes_nested_paths_and_bindings_are_native_settings(field, value, monkeypatch):
    values = _values("kubernetes")
    for name, configured in values.items():
        monkeypatch.setenv(f"MONAI_TEST_WORKER__{name.upper()}", str(configured))
    monkeypatch.setenv(f"MONAI_TEST_WORKER__{field.upper()}", value)
    config = executor_config_from_env("MONAI_TEST")
    assert isinstance(config, KubernetesConfig)
    assert str(getattr(config, field)) == value
