"""Resolve invocation bindings and arguments independently of worker runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TypeVar

from monai_service.contracts import ExecutionError, FileArgument, FileBinding, Invocation

BindingPath = TypeVar("BindingPath", Path, PurePosixPath)


def expand_arguments(
    invocation: Invocation, paths: Mapping[str, Path | PurePosixPath]
) -> list[str]:
    arguments = []
    for argument in invocation.arguments:
        if isinstance(argument, FileArgument):
            arguments.append(str(paths[argument.binding] / argument.relative_path))
        else:
            arguments.append(argument)
    return arguments


def resolve_binding_paths(bindings: tuple[FileBinding, ...]) -> dict[str, Path]:
    paths = {binding.name: binding.path.resolve() for binding in bindings}
    if any(not (path.is_file() or path.is_dir()) for path in paths.values()):
        raise ExecutionError("Invocation bindings must refer to existing files or directories")
    return paths


def binding_destination(root: BindingPath, binding: FileBinding, source: Path) -> BindingPath:
    destination = root / binding.name
    # Native file readers need the original filename extension.
    return destination / source.name if source.is_file() else destination
