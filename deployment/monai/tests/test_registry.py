"""Explicit task definitions reuse model-independent deployment composition."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from monai_service.adapters.execution import DockerExecutor, LocalProcessExecutor
from monai_service.adapters.mlflow import ModelSource
from monai_service.contracts import ModelDescriptor, TaskDefinition
from monai_service.registry import from_env
from monai_service.schemas.base import NoParameters


class OrdinaryImageWorkflow:
    descriptor = ModelDescriptor("Another image model", "Registry fixture", {"target": 1}, 3)

    def prepare(self, request, artifact, workspace):
        raise AssertionError("Configuration must not execute inference")

    def read_result(self, request, workspace):
        raise AssertionError("Configuration must not read inference output")


def definition(identifier="another-image", key="THIRD", factory=OrdinaryImageWorkflow):
    return TaskDefinition(
        identifier=identifier,
        key=key,
        workflow_factory=factory,
        parse_parameters=NoParameters.model_validate,
    )


def enable(monkeypatch, key, mode="process"):
    for name, value in {
        f"MONAI_SERVICE_ENABLE_{key}": "true",
        f"MONAI_SERVICE_{key}_MODEL_NAME": f"Model-{key}",
        f"MONAI_SERVICE_{key}_MODEL_ALIAS": f"alias-{key}",
        f"MONAI_SERVICE_{key}_WORKER__MODE": mode,
        "MONAI_SERVICE_MLFLOW_TRACKING_URI": "http://tracking-fixture",
        "MONAI_SERVICE_MLFLOW_REGISTRY_URI": "http://registry-fixture",
    }.items():
        monkeypatch.setenv(name, value)
    if mode == "process":
        monkeypatch.setenv(f"MONAI_SERVICE_{key}_WORKER__COMMAND", '["/bin/cp"]')


def test_third_definition_registers_without_shared_model_switches(tmp_path, monkeypatch):
    calls = []
    workflow = OrdinaryImageWorkflow()

    def create_workflow():
        calls.append("third")
        return workflow

    third = definition(factory=create_workflow)
    enable(monkeypatch, third.key)
    settings = from_env(
        (definition("existing-a", "FIRST"), definition("existing-b", "SECOND"), third),
        default_work_dir=tmp_path,
    )
    assert tuple(settings.registrations) == ("another-image",)
    registration = settings.registrations[third.identifier]
    assert calls == ["third"]
    assert registration.workflow is workflow
    assert registration.parse_parameters is third.parse_parameters
    assert isinstance(registration.parse_parameters({}), NoParameters)
    assert registration.workflow.descriptor.input_profile == "image"
    assert isinstance(registration.executor, LocalProcessExecutor)
    assert registration.executor.config.command == ("/bin/cp",)


@pytest.mark.parametrize("collision", ["identifier", "key", "key-case"])
def test_duplicate_definitions_are_rejected_before_factory_use(collision, tmp_path):
    def unexpected_factory():
        pytest.fail("Ambiguous catalogue invoked a workflow factory")

    first = definition(factory=unexpected_factory)
    second = replace(first, identifier="other", key="OTHER")
    if collision == "identifier":
        second = replace(second, identifier=first.identifier)
    else:
        second = replace(second, key=first.key.lower() if collision == "key-case" else first.key)
    with pytest.raises(ValueError, match="unique"):
        from_env((first, second), default_work_dir=tmp_path)


def test_disabled_definition_does_not_require_source_runtime_or_policy(tmp_path, monkeypatch):
    def unexpected_factory():
        pytest.fail("Disabled workflow factory was invoked")

    task = definition(factory=unexpected_factory)
    monkeypatch.setenv("MONAI_SERVICE_THIRD_WORKER__COMMAND", "not-json")
    settings = from_env((task,), default_work_dir=tmp_path)
    assert settings.work_dir == tmp_path
    assert not settings.registrations
    assert not list(tmp_path.iterdir())


def test_each_definition_has_its_own_source_and_runtime(tmp_path, monkeypatch):
    first = definition("first", "FIRST")
    second = definition("second", "SECOND")
    enable(monkeypatch, first.key)
    enable(monkeypatch, second.key, "container")
    monkeypatch.setenv("MONAI_SERVICE_WORK_DIR", str(tmp_path / "configured-workspace"))
    monkeypatch.setenv("MONAI_SERVICE_SECOND_WORKER__RUNTIME", "/configured/container-runtime")
    monkeypatch.setenv("MONAI_SERVICE_FIRST_WORKER__TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("MONAI_SERVICE_FIRST_WORKER__COMMAND", '["configured-executable", "--flag"]')
    monkeypatch.setenv(
        "MONAI_SERVICE_FIRST_WORKER__ENVIRONMENT", json.dumps({"FIRST_VALUE": "selected"})
    )
    monkeypatch.setenv("MONAI_SERVICE_SECOND_WORKER__IMAGE", "configured-image@sha256:fixed")
    monkeypatch.setenv("MONAI_SERVICE_SECOND_WORKER__TIMEOUT_SECONDS", "45")
    monkeypatch.setenv(
        "MONAI_SERVICE_SECOND_WORKER__ENVIRONMENT", json.dumps({"SECOND_VALUE": "selected"})
    )
    settings = from_env((first, second), default_work_dir=tmp_path)
    assert settings.work_dir == tmp_path / "configured-workspace"
    for task in (first, second):
        registration = settings.registrations[task.identifier]
        assert registration.artifacts.config == ModelSource(
            f"Model-{task.key}",
            f"alias-{task.key}",
            "http://tracking-fixture",
            "http://registry-fixture",
        )
        assert registration.artifacts.cache_root == settings.work_dir / "models"
    process = settings.registrations[first.identifier].executor
    container = settings.registrations[second.identifier].executor
    assert isinstance(process, LocalProcessExecutor)
    assert process.config.command == ("configured-executable", "--flag")
    assert process.config.timeout_seconds == 12.5
    assert process.config.environment["FIRST_VALUE"] == "selected"
    assert isinstance(container, DockerExecutor)
    assert container.config.image == "configured-image@sha256:fixed"
    assert container.config.runtime == "/configured/container-runtime"
    assert container.config.timeout_seconds == 45
    assert container.config.environment["SECOND_VALUE"] == "selected"


def test_missing_source_is_rejected_before_workflow_factory(tmp_path, monkeypatch):
    def unexpected_factory():
        pytest.fail("Workflow was constructed before source validation")

    task = definition(factory=unexpected_factory)
    monkeypatch.setenv("MONAI_SERVICE_ENABLE_THIRD", "true")
    with pytest.raises(ValidationError, match="model_name"):
        from_env((task,), default_work_dir=tmp_path)


@pytest.mark.parametrize(
    "value, expected",
    [("true", True), ("yes", True), ("1", True), ("false", False), ("no", False), ("0", False)],
)
def test_enable_flag_uses_native_boolean_values(value, expected, tmp_path, monkeypatch):
    task = definition()
    enable(monkeypatch, task.key)
    monkeypatch.setenv("MONAI_SERVICE_ENABLE_THIRD", value)
    settings = from_env((task,), default_work_dir=tmp_path)
    assert bool(settings.registrations) is expected


@pytest.mark.parametrize("value", ["sometimes", ""])
def test_invalid_enable_flag_is_rejected_before_workflow_factory(value, tmp_path, monkeypatch):
    def unexpected_factory():
        pytest.fail("Invalid enable flag invoked a workflow factory")

    monkeypatch.setenv("MONAI_SERVICE_ENABLE_THIRD", value)
    with pytest.raises(ValidationError, match="bool_parsing"):
        from_env((definition(factory=unexpected_factory),), default_work_dir=tmp_path)


@pytest.mark.parametrize(
    "module",
    [
        "monai_service.registry",
        "monai_service.models.arg",
        "monai_service.models.model_o",
        "vessel_deployment",
    ],
)
def test_imports_leave_native_inference_libraries_unloaded(module):
    code = """
import importlib
import importlib.abc
import sys

class BlockInferenceLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'nnunetv2', 'VesselPathTracerLib', 'onnxruntime'}:
            raise AssertionError('Imported native inference library: ' + fullname)

sys.meta_path.insert(0, BlockInferenceLibraries())
importlib.import_module(sys.argv[1])
assert not {'nnunetv2', 'VesselPathTracerLib', 'onnxruntime'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, module],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
