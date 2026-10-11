"""Load one execution variant through native Pydantic settings sources."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from monai_service.adapters.execution import DockerExecutor, LocalProcessExecutor
from monai_service.adapters.kubernetes import KubernetesExecutor
from monai_service.contracts import Executor
from monai_service.schemas.execution import (
    ContainerConfig,
    DirectConfig,
    ExecutorConfig,
    ProcessConfig,
)


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True, extra="forbid", env_nested_delimiter="__", env_parse_none_str="null"
    )

    worker: ExecutorConfig = DirectConfig()


def executor_config_from_env(prefix: str) -> ExecutorConfig:
    """Accept a complete WORKER JSON object or WORKER__FIELD overrides."""
    return WorkerSettings(_env_prefix=f"{prefix}_").worker


def executor_from_env(prefix: str) -> Executor | None:
    config = executor_config_from_env(prefix)
    if isinstance(config, DirectConfig):
        return None
    if isinstance(config, ProcessConfig):
        return LocalProcessExecutor(config)
    if isinstance(config, ContainerConfig):
        return DockerExecutor(config)
    return KubernetesExecutor(config)
