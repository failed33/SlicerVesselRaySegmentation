"""Map request file bindings to isolated subpaths on a shared worker PVC."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from monai_service.adapters.bindings import binding_destination, resolve_binding_paths
from monai_service.contracts import ExecutionError, Invocation
from monai_service.schemas.execution import KubernetesConfig


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copyfile(source, destination)


@dataclass(frozen=True)
class StagedOutput:
    original: Path
    staged: Path

    def retrieve(self) -> None:
        paths = [self.staged, *self.staged.rglob("*")] if self.staged.is_dir() else [self.staged]
        if any(path.is_symlink() for path in paths):
            raise ExecutionError("Kubernetes worker outputs cannot contain symbolic links")
        if self.original.is_dir():
            shutil.rmtree(self.original)
        elif self.original.exists():
            self.original.unlink()
        _copy(self.staged, self.original)


@dataclass(frozen=True)
class StagedInvocation:
    root: Path
    paths: dict[str, PurePosixPath]
    mounts: list[dict[str, object]]
    outputs: tuple[StagedOutput, ...]

    def retrieve(self) -> None:
        for output in self.outputs:
            output.retrieve()

    def cleanup(self) -> None:
        shutil.rmtree(self.root)


def stage_invocation(invocation: Invocation, config: KubernetesConfig) -> StagedInvocation:
    shared = config.shared_root.resolve()
    if not shared.is_dir():
        raise ExecutionError("Kubernetes shared_root must be an existing mounted directory")
    sources = resolve_binding_paths(invocation.bindings)
    root = Path(tempfile.mkdtemp(prefix="job-", dir=shared))
    paths: dict[str, PurePosixPath] = {}
    mounts: list[dict[str, object]] = []
    outputs = []
    try:
        for binding in invocation.bindings:
            source = sources[binding.name]
            destination = binding_destination(config.mount_path, binding, source)
            staged = source
            if binding.writable or not source.is_relative_to(shared):
                staged = binding_destination(root, binding, source)
                _copy(source, staged)
            if binding.writable:
                outputs.append(StagedOutput(source, staged))
            paths[binding.name] = destination
            mounts.append(
                {
                    "name": "workspace",
                    "mountPath": str(destination),
                    "subPath": staged.relative_to(shared).as_posix(),
                    "readOnly": not binding.writable,
                }
            )
        return StagedInvocation(root, paths, mounts, tuple(outputs))
    except OSError as error:
        shutil.rmtree(root)
        raise ExecutionError("Could not stage Kubernetes worker files") from error
    except BaseException:
        shutil.rmtree(root)
        raise
