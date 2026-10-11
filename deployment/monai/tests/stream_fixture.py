"""A delayed file-producing worker served through the production HTTP seam."""

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from monai_service.adapters.execution import LocalProcessExecutor
from monai_service.app import create_app
from monai_service.config import ServiceSettings
from monai_service.contracts import (
    FileArgument,
    InferencePreview,
    InferenceRequest,
    InferenceResult,
    Invocation,
    ModelDeployment,
    ModelDescriptor,
    ResolvedArtifact,
)
from monai_service.schemas.base import NoParameters
from monai_service.schemas.execution import ProcessConfig
from monai_service.worker_io import prepare_worker_files, read_worker_preview, read_worker_result
from tests.fixtures import FixedArtifactSource, smoke_deployment


@dataclass(frozen=True)
class StreamFixtureWorkflow:
    descriptor = ModelDescriptor("Streaming test", "Delayed cumulative fixture", {"vessel": 1}, 3)

    def prepare(
        self, request: InferenceRequest, artifact: ResolvedArtifact, workspace: Path
    ) -> Invocation:
        return prepare_worker_files(request.image, workspace)

    def prepare_stream(
        self, request: InferenceRequest, artifact: ResolvedArtifact, workspace: Path
    ) -> Invocation:
        invocation = self.prepare(request, artifact, workspace)
        return replace(
            invocation,
            arguments=(
                *invocation.arguments,
                "--preview-directory",
                FileArgument("output", PurePosixPath("previews")),
            ),
        )

    def read_preview(self, workspace: Path) -> InferencePreview | None:
        return read_worker_preview(workspace)

    def read_result(self, request: InferenceRequest, workspace: Path) -> InferenceResult:
        return read_worker_result(workspace)


def fixture_deployment() -> ModelDeployment:
    return ModelDeployment(
        StreamFixtureWorkflow(),
        FixedArtifactSource(smoke_deployment().artifacts.resolve()),
        LocalProcessExecutor(
            ProcessConfig(
                mode="process",
                command=(sys.executable, str(Path(__file__).with_name("stream_worker.py"))),
                timeout_seconds=float(os.environ.get("STREAM_TEST_TIMEOUT", "30")),
            )
        ),
        NoParameters.model_validate,
    )


app = create_app(
    ServiceSettings(
        Path(os.environ.get("MONAI_SERVICE_WORK_DIR", "/tmp/stream-fixture")),
        {"stream-fixture": fixture_deployment(), "ordinary-fixture": smoke_deployment()},
    )
)
