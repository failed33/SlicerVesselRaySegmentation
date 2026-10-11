"""Validated configurations carried into execution adapters."""

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

Argument = Annotated[str, StringConstraints(strict=True, pattern=r"^[^\x00]*[^\x00\s][^\x00]*$")]
Command = Annotated[tuple[Argument, ...], Field(min_length=1)]
EnvironmentName = Annotated[str, StringConstraints(strict=True, pattern=r"^[^\x00=]+$")]
EnvironmentValue = Annotated[str, StringConstraints(strict=True, pattern=r"^[^\x00]*$")]
Environment = dict[EnvironmentName, EnvironmentValue]
Timeout = Annotated[float, Field(gt=0, allow_inf_nan=False)]
GpuIdentifier = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(?:[0-9]+|(?:GPU|MIG)-[A-Za-z0-9/-]+)$"),
]
GpuRequest = Literal["all"] | Annotated[tuple[GpuIdentifier, ...], Field(min_length=1)]
CdiDevice = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^[A-Za-z](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/"
            r"[A-Za-z](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?="
            r"[A-Za-z0-9](?:[A-Za-z0-9_.:-]*[A-Za-z0-9])?$"
        ),
    ),
]


def _device_path(path: PurePosixPath) -> PurePosixPath:
    if not path.is_relative_to("/dev") or path == PurePosixPath("/dev") or ".." in path.parts:
        raise ValueError("Device paths must be absolute paths within /dev")
    if "\x00" in str(path) or ":" in str(path):
        raise ValueError("Device paths cannot contain NUL or Docker's mapping separator")
    return path


DevicePath = Annotated[PurePosixPath, AfterValidator(_device_path)]


class DeviceMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: DevicePath
    container: DevicePath


class DirectConfig(BaseModel):
    """Use the Label environment's installed inference runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["direct"] = "direct"


class ProcessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["process"]
    command: Command
    environment: Environment = Field(default_factory=dict)
    timeout_seconds: Timeout = 600.0


class ContainerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["container"]
    image: Argument
    runtime: Argument = "docker"
    gpus: GpuRequest | None = None
    cdi_devices: tuple[CdiDevice, ...] = ()
    devices: tuple[DeviceMapping, ...] = ()
    environment: Environment = Field(default_factory=dict)
    timeout_seconds: Timeout = 600.0

    @model_validator(mode="after")
    def accelerator_request_is_unambiguous(self) -> Self:
        if self.gpus is not None and self.cdi_devices:
            raise ValueError("Select either gpus or cdi_devices, not both")
        return self


def _absolute_path(path: Path | PurePosixPath) -> Path | PurePosixPath:
    if not path.is_absolute() or ".." in path.parts or "\x00" in str(path):
        raise ValueError(
            "Workspace paths must be absolute and cannot contain parent traversal or NUL"
        )
    return path


class KubernetesConfig(BaseModel):
    """File-exchange and lifecycle settings owned by the Job executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["kubernetes"]
    job_template: Path
    namespace: Argument
    pvc_claim: Argument
    shared_root: Annotated[Path, AfterValidator(_absolute_path)]
    mount_path: Annotated[PurePosixPath, AfterValidator(_absolute_path)] = PurePosixPath(
        "/bindings"
    )
    timeout_seconds: Timeout = 600.0


ExecutorConfig = Annotated[
    DirectConfig | ProcessConfig | ContainerConfig | KubernetesConfig, Field(discriminator="mode")
]
