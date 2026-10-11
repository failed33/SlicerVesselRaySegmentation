"""Compose registered tasks with shared artifact and execution infrastructure."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field, StringConstraints, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict

from monai_service.adapters.executor_config import executor_from_env
from monai_service.adapters.mlflow import MLflowArtifactSource, ModelSource
from monai_service.config import ServiceSettings
from monai_service.contracts import ModelDeployment, Parameters, TaskDefinition

_enabled = TypeAdapter(bool)
NonBlankString = Annotated[str, StringConstraints(strict=True, pattern=r"\S")]


class RegistrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONAI_SERVICE_", frozen=True, extra="forbid")

    work_dir: Path | None = None


class ModelSourceSettings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, extra="forbid")

    model_name: NonBlankString
    model_alias: NonBlankString
    tracking_uri: NonBlankString = Field(validation_alias="MONAI_SERVICE_MLFLOW_TRACKING_URI")
    registry_uri: Annotated[
        NonBlankString | None, BeforeValidator(lambda value: None if value == "" else value)
    ] = Field(default=None, validation_alias="MONAI_SERVICE_MLFLOW_REGISTRY_URI")

    def artifacts(self, work_dir: Path) -> MLflowArtifactSource:
        return MLflowArtifactSource(
            ModelSource(self.model_name, self.model_alias, self.tracking_uri, self.registry_uri),
            work_dir / "models",
        )


def _validate_definitions(definitions: Sequence[TaskDefinition]) -> None:
    identifiers = {task.identifier for task in definitions}
    environment_keys = {task.key.casefold() for task in definitions}
    if len(identifiers) != len(definitions) or len(environment_keys) != len(definitions):
        raise ValueError("Task identifiers and environment keys must be unique")


def _enabled_definitions(definitions: Sequence[TaskDefinition]) -> Iterator[TaskDefinition]:
    for task in definitions:
        value = os.environ.get(f"MONAI_SERVICE_ENABLE_{task.key}", "false")
        if _enabled.validate_python(value):
            yield task


def _deployment_from_env(
    task: TaskDefinition[Parameters], work_dir: Path
) -> ModelDeployment[Parameters]:
    prefix = f"MONAI_SERVICE_{task.key}"
    source = ModelSourceSettings(_env_prefix=f"{prefix}_")
    return ModelDeployment(
        workflow=task.workflow_factory(),
        artifacts=source.artifacts(work_dir),
        executor=executor_from_env(prefix),
        parse_parameters=task.parse_parameters,
    )


def from_env(definitions: Sequence[TaskDefinition], *, default_work_dir: Path) -> ServiceSettings:
    """Validate the catalogue, select enabled tasks, and construct their deployments."""
    _validate_definitions(definitions)
    settings = RegistrySettings()
    work_dir = settings.work_dir if settings.work_dir is not None else default_work_dir
    registrations = {
        task.identifier: _deployment_from_env(task, work_dir)
        for task in _enabled_definitions(definitions)
    }
    return ServiceSettings(work_dir, registrations)
