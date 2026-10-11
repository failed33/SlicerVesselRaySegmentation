"""Application behavior without MONAI, MLflow, image libraries or a worker runtime."""

from pathlib import Path, PurePosixPath

import pytest
from pydantic import BaseModel, ConfigDict, StrictStr

from monai_service.application import InferenceService
from monai_service.contracts import (
    ArtifactIdentity,
    ExecutionError,
    ExecutionResult,
    FileArgument,
    FileBinding,
    InferenceRequest,
    InferenceResult,
    Invocation,
    ModelDeployment,
    ModelDescriptor,
    ResolvedArtifact,
)


class FixedArtifact:
    def __init__(self, path):
        self.path = path
        self.calls = 0

    def resolve(self):
        self.calls += 1
        return ResolvedArtifact(ArtifactIdentity("example", "42", "example:42"), self.path)


class DocumentParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: StrictStr = "hello"


class DocumentWorkflow:
    descriptor = ModelDescriptor("Example", "Non-image output", {}, 0, "document", "other")

    def prepare(self, request: InferenceRequest[DocumentParameters], artifact, workspace):
        return Invocation(
            (request.parameters.text,), (FileBinding("output", workspace, writable=True),)
        )

    def read_result(self, request, workspace):
        return InferenceResult(
            workspace / "result.txt", {"model_version": "untrusted", "characters": 5}
        )


class DocumentExecutor:
    def run(self, invocation):
        (invocation.bindings[0].path / "result.txt").write_text(invocation.arguments[0])
        return ExecutionResult("test", "document-writer")


def make_service(tmp_path, *, workflow=None, executor=None):
    artifacts = FixedArtifact(tmp_path)
    service = InferenceService(
        ModelDeployment(
            workflow or DocumentWorkflow(),
            artifacts,
            executor or DocumentExecutor(),
            DocumentParameters.model_validate,
        ),
        tmp_path / "requests",
    )
    return service, artifacts


def test_service_transfers_only_result_and_captures_identity_once(tmp_path):
    service, artifacts = make_service(tmp_path)
    result = service.infer(InferenceRequest(tmp_path / "input", DocumentParameters()))
    assert result.path.read_text() == "hello"
    assert list(service.work_dir.iterdir()) == [result.path]
    assert artifacts.calls == 1
    assert result.metadata == {
        "characters": 5,
        "model": "example",
        "model_version": "42",
        "model_artifact": "example:42",
        "runtime": "test",
        "runtime_reference": "document-writer",
    }
    result.path.unlink()
    assert not list(service.work_dir.iterdir())


def test_application_passes_parsed_parameters_to_workflow(tmp_path):
    service, _ = make_service(tmp_path)
    parameters = service.deployment.parse_parameters({"text": "typed document"})
    result = service.infer(InferenceRequest(tmp_path / "input", parameters))
    assert result.path.read_text() == "typed document"
    result.path.unlink()


def test_execution_failure_removes_partial_output(tmp_path):
    class FailingExecutor:
        def run(self, invocation):
            (invocation.bindings[0].path / "partial.txt").write_text("partial")
            raise ExecutionError("Worker failed")

    service, _ = make_service(tmp_path, executor=FailingExecutor())
    with pytest.raises(ExecutionError, match="Worker failed"):
        service.infer(InferenceRequest(tmp_path / "input", DocumentParameters()))
    assert not list(service.work_dir.iterdir())


@pytest.mark.parametrize("result_kind", ["missing", "external", "symlink"])
def test_unowned_result_is_rejected_without_deleting_external_file(tmp_path, result_kind):
    external = tmp_path / "external.txt"
    external.write_text("preserve")

    class InvalidResultWorkflow(DocumentWorkflow):
        def read_result(self, request, workspace):
            result = workspace / "absent.txt"
            if result_kind == "external":
                result = external
            elif result_kind == "symlink":
                result.symlink_to(external)
            return InferenceResult(result)

    service, _ = make_service(tmp_path, workflow=InvalidResultWorkflow())
    with pytest.raises(ExecutionError):
        service.infer(InferenceRequest(tmp_path / "input", DocumentParameters()))
    assert external.read_text() == "preserve"
    assert not list(service.work_dir.iterdir())


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "nested/../../escape"])
def test_file_arguments_cannot_escape_bindings(relative):
    with pytest.raises(ValueError, match="within their binding"):
        FileArgument("output", PurePosixPath(relative))


def test_invocation_rejects_ambiguous_or_missing_bindings():
    binding = FileBinding("input", Path("input"))
    with pytest.raises(ValueError, match="unique"):
        Invocation((), (binding, binding))
    with pytest.raises(ValueError, match="Unknown file binding"):
        Invocation((FileArgument("missing"),), (binding,))
