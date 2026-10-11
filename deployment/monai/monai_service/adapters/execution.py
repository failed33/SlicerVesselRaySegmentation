"""Local subprocess and Docker adapters for runtime-neutral invocations."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO

from monai_service.adapters.bindings import (
    binding_destination,
    expand_arguments,
    resolve_binding_paths,
)
from monai_service.contracts import (
    ExecutionCancelled,
    ExecutionControl,
    ExecutionError,
    ExecutionResult,
    ExecutionTimeout,
    FileBinding,
    Invocation,
)
from monai_service.schemas.execution import ContainerConfig, ProcessConfig

STDERR_TAIL_BYTES = 8192
logger = logging.getLogger(__name__)
type ProcessCleanup = Callable[[], None]


def _group_has_live_members(group_id: int) -> bool:
    try:
        snapshot = subprocess.run(
            ["ps", "-A", "-o", "pgid=", "-o", "stat="],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExecutionError("Could not inspect inference process-group cleanup") from error
    for row in snapshot.stdout.splitlines():
        fields = row.split()
        if len(fields) != 2:
            raise ExecutionError("Could not inspect inference process-group cleanup")
        if fields[0] != str(group_id):
            continue
        if not fields[1].startswith("Z"):
            return True
    return False


def _darwin_group_is_already_stopped(group_id: int) -> bool:
    # Darwin excludes zombies from killpg, returning EPERM even when every
    # group member has exited. A live member or failed inspection is not success.
    return sys.platform == "darwin" and not _group_has_live_members(group_id)


def _stop_process_group(group_id: int) -> None:
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        if not _darwin_group_is_already_stopped(group_id):
            raise ExecutionError("Could not stop inference process group") from error


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            _stop_process_group(process.pid)
        elif process.poll() is None:
            process.kill()
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise ExecutionError("Could not stop inference process") from error


def _log_failure(stderr_log: BinaryIO) -> None:
    stderr_log.seek(0, os.SEEK_END)
    stderr_log.seek(max(0, stderr_log.tell() - STDERR_TAIL_BYTES))
    tail = stderr_log.read(STDERR_TAIL_BYTES).decode("utf-8", errors="replace")
    if tail:
        logger.error("Inference process stderr tail: %s", tail)


def _start_process(
    command: list[str], environment: Mapping[str, str] | None, stderr_log: BinaryIO
) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
            start_new_session=os.name == "posix",
            env={**os.environ, **(environment or {})},
        )
    except OSError as error:
        raise ExecutionError("Could not start inference process") from error


def _wait_for_exit(process: subprocess.Popen[bytes], timeout: float) -> int:
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise ExecutionTimeout("Inference process exceeded its execution deadline") from error


def _finish_process(
    process: subprocess.Popen[bytes],
    exit_code: int | None,
    stderr_log: BinaryIO,
    cleanup: ProcessCleanup | None,
) -> None:
    cleanup_required = True
    try:
        _stop_process(process)
        cleanup_required = exit_code != 0
    finally:
        if cleanup_required:
            _log_failure(stderr_log)
            if cleanup is not None:
                cleanup()


def _run_process(
    command: list[str],
    timeout: float,
    environment: Mapping[str, str] | None,
    cleanup: ProcessCleanup | None = None,
    control: ExecutionControl | None = None,
) -> None:
    # Spool stderr to disk; only the bounded tail is read on failure.
    if control is not None and control.cancelled.is_set():
        raise ExecutionCancelled("Inference stream was cancelled")
    with tempfile.TemporaryFile() as stderr_log:
        process = _start_process(command, environment, stderr_log)
        exit_code: int | None = None
        try:
            exit_code = (
                _wait_for_exit(process, timeout)
                if control is None
                else _wait_observed(process, timeout, control)
            )
        finally:
            _finish_process(process, exit_code, stderr_log, cleanup)
        if exit_code != 0:
            raise ExecutionError(f"Inference process exited unsuccessfully with status {exit_code}")


def _wait_observed(
    process: subprocess.Popen[bytes], timeout: float, control: ExecutionControl
) -> int:
    deadline = time.monotonic() + timeout
    while True:
        if control.cancelled.is_set():
            raise ExecutionCancelled("Inference stream was cancelled")
        control.poll()
        if (exit_code := process.poll()) is not None:
            return exit_code
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionTimeout("Inference process exceeded its execution deadline")
        control.cancelled.wait(min(0.1, remaining))


def _container_bindings(
    bindings: tuple[FileBinding, ...],
) -> tuple[list[str], dict[str, PurePosixPath]]:
    sources = resolve_binding_paths(bindings)
    mounts = []
    paths = {}
    for binding in bindings:
        source = sources[binding.name]
        destination = binding_destination(PurePosixPath("/bindings"), binding, source)
        paths[binding.name] = destination
        access = "rw" if binding.writable else "ro"
        mounts.extend(["--volume", f"{source}:{destination}:{access}"])
    return mounts, paths


@dataclass(frozen=True)
class LocalProcessExecutor:
    config: ProcessConfig

    def run(self, invocation: Invocation) -> ExecutionResult:
        paths = resolve_binding_paths(invocation.bindings)
        _run_process(
            [*self.config.command, *expand_arguments(invocation, paths)],
            self.config.timeout_seconds,
            self.config.environment,
        )
        return ExecutionResult("process", " ".join(self.config.command))

    def run_observed(self, invocation: Invocation, control: ExecutionControl) -> ExecutionResult:
        paths = resolve_binding_paths(invocation.bindings)
        _run_process(
            [*self.config.command, *expand_arguments(invocation, paths)],
            self.config.timeout_seconds,
            self.config.environment,
            control=control,
        )
        return ExecutionResult("process", " ".join(self.config.command))


@dataclass(frozen=True)
class DockerExecutor:
    config: ContainerConfig

    def _command(self, invocation: Invocation, name: str) -> list[str]:
        mounts, paths = _container_bindings(invocation.bindings)
        command = [self.config.runtime, "run", "--rm", "--name", name, "--network", "none", *mounts]
        if self.config.gpus is not None:
            gpus = self.config.gpus
            request = "all" if gpus == "all" else f'"device={",".join(gpus)}"'
            command.extend(["--gpus", request])
        for device in self.config.devices:
            command.extend(["--device", f"{device.host}:{device.container}:rw"])
        for device in self.config.cdi_devices:
            command.extend(["--device", device])
        for key, value in self.config.environment.items():
            command.extend(["--env", f"{key}={value}"])
        return [*command, self.config.image, *expand_arguments(invocation, paths)]

    def _remove_container(self, name: str) -> None:
        try:
            result = subprocess.run(
                [self.config.runtime, "rm", "--force", name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ExecutionError("Could not clean up inference container") from error
        if result.returncode == 0:
            return
        if b"No such container" in result.stderr:
            return
        raise ExecutionError("Could not clean up inference container")

    def run(self, invocation: Invocation) -> ExecutionResult:
        name = f"inference-{uuid.uuid4().hex}"
        _run_process(
            self._command(invocation, name),
            self.config.timeout_seconds,
            None,
            cleanup=lambda: self._remove_container(name),
        )
        return ExecutionResult("docker", self.config.image)

    def run_observed(self, invocation: Invocation, control: ExecutionControl) -> ExecutionResult:
        name = f"inference-{uuid.uuid4().hex}"
        _run_process(
            self._command(invocation, name),
            self.config.timeout_seconds,
            None,
            cleanup=lambda: self._remove_container(name),
            control=control,
        )
        return ExecutionResult("docker", self.config.image)
