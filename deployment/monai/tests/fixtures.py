"""Test workflows composed through the same application ports as deployed models."""

import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from monai_service.adapters.execution import LocalProcessExecutor
from monai_service.app import create_app
from monai_service.config import ServiceSettings
from monai_service.contracts import (
    ArtifactIdentity,
    Executor,
    InferenceRequest,
    InferenceResult,
    Invocation,
    ModelDeployment,
    ModelDescriptor,
    ResolvedArtifact,
)
from monai_service.schemas.base import NoParameters
from monai_service.schemas.execution import ProcessConfig
from monai_service.worker_io import (
    prepare_worker_files,
    read_worker_result,
)

SMOKE_WORKER_PATH = Path(__file__).with_name("smoke_worker.py")


@dataclass(frozen=True)
class FixedArtifactSource:
    artifact: ResolvedArtifact

    def resolve(self) -> ResolvedArtifact:
        return self.artifact


@dataclass(frozen=True)
class SmokeWorkflow:
    @property
    def descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            title="Test fixture",
            description="Test-only fixture; thresholds the input at zero.",
            labels={"background": 0, "test_foreground": 1},
            dimension=3,
        )

    def prepare(
        self, request: InferenceRequest[NoParameters], artifact: ResolvedArtifact, workspace: Path
    ) -> Invocation:
        return prepare_worker_files(request.image, workspace)

    def read_result(
        self, request: InferenceRequest[NoParameters], workspace: Path
    ) -> InferenceResult:
        return read_worker_result(workspace)


def smoke_deployment(executor: Executor | None = None) -> ModelDeployment[NoParameters]:
    return ModelDeployment(
        SmokeWorkflow(),
        FixedArtifactSource(
            ResolvedArtifact(
                ArtifactIdentity("smoke-fixture", "1", "test-fixture:smoke"),
                Path(__file__).parent,
            )
        ),
        executor
        or LocalProcessExecutor(
            ProcessConfig(
                mode="process", command=(sys.executable, str(SMOKE_WORKER_PATH)), timeout_seconds=30
            )
        ),
        NoParameters.model_validate,
    )


def create_test_app(service_settings: ServiceSettings, executor: Executor | None = None) -> FastAPI:
    return create_app(
        ServiceSettings(
            service_settings.work_dir,
            {**service_settings.registrations, "smoke-test": smoke_deployment(executor)},
        )
    )
