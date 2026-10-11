"""Inference orchestration and request-file ownership."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import Event, Lock
from typing import Generic

from label_inference.files import ResultFileError, retain_result

from monai_service.contracts import (
    ExecutionCancelled,
    ExecutionControl,
    ExecutionError,
    ExecutionResult,
    InferenceError,
    InferencePreview,
    InferenceRequest,
    InferenceResult,
    ModelDeployment,
    ModelDescriptor,
    ObservableExecutor,
    Parameters,
    ResolvedArtifact,
    StreamingWorkflow,
)


def _with_provenance(
    result: InferenceResult, artifact: ResolvedArtifact, execution: ExecutionResult
) -> InferenceResult:
    metadata = {
        **result.metadata,
        "model": artifact.identity.name,
        "model_version": artifact.identity.version,
        "model_artifact": artifact.identity.uri,
        "runtime": execution.runtime,
        "runtime_reference": execution.reference,
    }
    return InferenceResult(result.path, metadata)


class _PreviewDelivery:
    """Hold a single worker snapshot until its HTTP consumer releases it."""

    def __init__(
        self,
        workflow: StreamingWorkflow,
        workspace: Path,
        control: Event,
        receive: Callable[[InferencePreview], None],
    ) -> None:
        self.workflow = workflow
        self.workspace = workspace
        self.cancelled = control
        self.receive = receive
        self.pending: Event | None = None
        self.release_pending: Callable[[], None] | None = None

    def poll(self) -> None:
        if self.pending is not None and not self.pending.is_set():
            return
        self.pending = None
        candidate = self.workflow.read_preview(self.workspace)
        if candidate is None:
            return
        preview = candidate
        acknowledged = Event()
        release_lock = Lock()
        self.pending = acknowledged

        def release() -> None:
            with release_lock:
                if acknowledged.is_set():
                    return
                try:
                    preview.release()
                finally:
                    acknowledged.set()

        self.release_pending = release
        try:
            self.receive(InferencePreview(preview.path, preview.metadata, release))
        except BaseException:
            release()
            raise

    def wait_pending(self) -> None:
        while self.pending is not None and not self.pending.wait(0.1):
            if self.cancelled.is_set():
                if self.release_pending is not None:
                    self.release_pending()
                raise ExecutionCancelled("Inference stream was cancelled")

    def finish(self) -> None:
        while True:
            self.wait_pending()
            self.poll()
            if self.pending is None:
                return


class InferenceService(Generic[Parameters]):
    def __init__(self, deployment: ModelDeployment[Parameters], work_dir: Path) -> None:
        self.deployment = deployment
        self.work_dir = work_dir

    @property
    def descriptor(self) -> ModelDescriptor:
        return self.deployment.workflow.descriptor

    @property
    def supports_streaming(self) -> bool:
        return isinstance(self.deployment.workflow, StreamingWorkflow) and isinstance(
            self.deployment.executor, ObservableExecutor
        )

    def stream(
        self,
        request: InferenceRequest[Parameters],
        cancelled: Event,
        receive: Callable[[InferencePreview], None],
    ) -> InferenceResult:
        """Emit owned previews and return the final file after isolated execution."""
        workflow, executor = self.deployment.workflow, self.deployment.executor
        if not isinstance(workflow, StreamingWorkflow) or not isinstance(
            executor, ObservableExecutor
        ):
            raise ExecutionError("Streaming requires a supported workflow and isolated executor")
        artifact = self.deployment.artifacts.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="inference-", dir=self.work_dir) as temporary:
            workspace = Path(temporary).resolve()
            invocation = workflow.prepare_stream(request, artifact, workspace)
            delivery = _PreviewDelivery(workflow, workspace, cancelled, receive)
            try:
                execution = executor.run_observed(
                    invocation, ExecutionControl(cancelled, delivery.poll)
                )
                delivery.finish()
                result = self.deployment.workflow.read_result(request, workspace)
                result = _with_provenance(result, artifact, execution)
                output = self._transfer_result(result.path, workspace)
                return InferenceResult(output, result.metadata)
            finally:
                delivery.wait_pending()

    def infer(self, request: InferenceRequest[Parameters]) -> InferenceResult:
        """Return one validated file whose deletion belongs to the caller."""
        artifact = self.deployment.artifacts.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        output: Path | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="inference-", dir=self.work_dir) as temporary:
                workspace = Path(temporary).resolve()
                result = self._run_workflow(request, artifact, workspace)
                output = self._transfer_result(result.path, workspace)
            return InferenceResult(output, result.metadata)
        except BaseException:
            if output is not None:
                output.unlink(missing_ok=True)
            raise

    def _run_workflow(
        self,
        request: InferenceRequest[Parameters],
        artifact: ResolvedArtifact,
        workspace: Path,
    ) -> InferenceResult:
        workflow = self.deployment.workflow
        if self.deployment.executor is None:
            result = self._infer_direct(request, artifact, workspace)
            return _with_provenance(result, artifact, ExecutionResult("direct", "python"))
        invocation = workflow.prepare(request, artifact, workspace)
        execution = self.deployment.executor.run(invocation)
        result = workflow.read_result(request, workspace)
        return _with_provenance(result, artifact, execution)

    def _infer_direct(
        self,
        request: InferenceRequest[Parameters],
        artifact: ResolvedArtifact,
        workspace: Path,
    ) -> InferenceResult:
        try:
            return self.deployment.workflow.infer(request, artifact, workspace)
        except InferenceError:
            raise
        except (ValueError, RuntimeError, OSError) as error:
            raise ExecutionError(str(error)) from error

    def _transfer_result(self, source: Path, workspace: Path) -> Path:
        try:
            return retain_result(source, workspace, self.work_dir)
        except ResultFileError as error:
            raise ExecutionError(str(error)) from error
