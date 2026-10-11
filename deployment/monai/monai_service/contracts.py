"""Application contracts, independent of model libraries and hosting frameworks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import Event
from typing import Generic, Protocol, TypeVar, runtime_checkable

Parameters = TypeVar("Parameters")


class InferenceError(RuntimeError):
    """An inference operation could not complete."""


class InvalidRequest(InferenceError):
    """The submitted inputs do not satisfy the selected workflow."""


class ArtifactUnavailable(InferenceError):
    """The configured model artifact could not be resolved or downloaded."""


class ExecutionError(InferenceError):
    """Execution failed or produced an unusable result."""


class ExecutionTimeout(ExecutionError):
    """Execution exceeded the configured deadline."""


class ExecutionCancelled(ExecutionError):
    """The requesting client stopped consuming inference."""


@dataclass(frozen=True)
class ModelDescriptor:
    title: str
    description: str
    labels: Mapping[str, int]
    dimension: int
    input_profile: str = "image"
    task_type: str = "segmentation"


@dataclass(frozen=True)
class InferenceRequest(Generic[Parameters]):
    """Inputs already parsed by the selected deployment's parameter schema."""

    image: Path
    parameters: Parameters


@dataclass(frozen=True)
class ArtifactIdentity:
    name: str
    version: str
    uri: str


@dataclass(frozen=True)
class ResolvedArtifact:
    identity: ArtifactIdentity
    path: Path


@dataclass(frozen=True)
class FileBinding:
    """A request-local file or directory exchanged with a workload."""

    name: str
    path: Path
    writable: bool = False

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("File binding names must be identifiers")


@dataclass(frozen=True)
class FileArgument:
    """A CLI argument referring to a binding, optionally to a file within it."""

    binding: str
    relative_path: PurePosixPath = PurePosixPath(".")

    def __post_init__(self) -> None:
        if self.relative_path.is_absolute() or ".." in self.relative_path.parts:
            raise ValueError("File arguments must remain within their binding")


@dataclass(frozen=True)
class Invocation:
    """Runtime-neutral arguments and file access required by one operation."""

    arguments: tuple[str | FileArgument, ...]
    bindings: tuple[FileBinding, ...]

    def __post_init__(self) -> None:
        names = [binding.name for binding in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("File binding names must be unique")
        for argument in self.arguments:
            if isinstance(argument, FileArgument) and argument.binding not in names:
                raise ValueError(f"Unknown file binding: {argument.binding}")


@dataclass(frozen=True)
class ExecutionResult:
    runtime: str
    reference: str


@dataclass(frozen=True)
class InferenceResult:
    path: Path
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class InferencePreview:
    """An immutable snapshot whose owner releases it after transmission."""

    path: Path
    metadata: Mapping[str, object]
    release: Callable[[], None]


@dataclass(frozen=True)
class ExecutionControl:
    cancelled: Event
    poll: Callable[[], None]


class ArtifactSource(Protocol):
    def resolve(self) -> ResolvedArtifact:
        """Capture one immutable version and return its complete local directory."""
        ...


class Executor(Protocol):
    def run(self, invocation: Invocation) -> ExecutionResult:
        """Execute, retrieve writable outputs, and finish execution cleanup."""
        ...


@runtime_checkable
class ObservableExecutor(Protocol):
    def run_observed(self, invocation: Invocation, control: ExecutionControl) -> ExecutionResult:
        """Poll request progress and stop isolated execution on cancellation."""
        ...


@runtime_checkable
class StreamingWorkflow(Protocol[Parameters]):
    def prepare_stream(
        self,
        request: InferenceRequest[Parameters],
        artifact: ResolvedArtifact,
        workspace: Path,
    ) -> Invocation: ...

    def read_preview(self, workspace: Path) -> InferencePreview | None: ...


class InferenceWorkflow(Protocol[Parameters]):
    @property
    def descriptor(self) -> ModelDescriptor: ...

    def infer(
        self, request: InferenceRequest[Parameters], artifact: ResolvedArtifact, workspace: Path
    ) -> InferenceResult:
        """Run inference directly in the Label environment."""
        ...

    def prepare(
        self,
        request: InferenceRequest[Parameters],
        artifact: ResolvedArtifact,
        workspace: Path,
    ) -> Invocation:
        """Stage inputs and describe the complete operation."""
        ...

    def read_result(
        self, request: InferenceRequest[Parameters], workspace: Path
    ) -> InferenceResult:
        """Validate and interpret workflow outputs within the workspace."""
        ...


@dataclass(frozen=True)
class ModelDeployment(Generic[Parameters]):
    """Pair parsing with its workflow; no executor means direct Label inference."""

    workflow: InferenceWorkflow[Parameters]
    artifacts: ArtifactSource
    executor: Executor | None
    parse_parameters: Callable[[object], Parameters]


@dataclass(frozen=True)
class TaskDefinition(Generic[Parameters]):
    """A task's registration data and factory, owned by its implementation module."""

    identifier: str
    key: str
    workflow_factory: Callable[[], InferenceWorkflow[Parameters]]
    parse_parameters: Callable[[object], Parameters]
