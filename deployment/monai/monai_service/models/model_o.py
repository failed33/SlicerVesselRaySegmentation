"""ModelO polyline schema, workflow and shared-device Dawn worker.

Callers supply ordered polylines in the submitted CT's ZYX voxel grid. Shared
preparation generates fixed-point rays and selects patches; Dawn generates the
seven ONNX inputs and consumes logits on the same GPU queue.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Literal

from model_o_inference.schemas import ModelOParameters
from pydantic import BeforeValidator, TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict

from monai_service.contracts import (
    FileArgument,
    FileBinding,
    InferencePreview,
    InferenceRequest,
    InferenceResult,
    Invocation,
    ModelDescriptor,
    ResolvedArtifact,
    TaskDefinition,
)
from monai_service.worker_io import (
    PreviewPublisher,
    prepare_worker_files,
    read_worker_preview,
    read_worker_result,
)

if TYPE_CHECKING:
    import SimpleITK as sitk
    from model_o_inference.preview import PatchProgress


@dataclass(frozen=True)
class _ModelOPreviews:
    """Adapt cumulative model images to the worker's bounded preview output."""

    publisher: PreviewPublisher

    def on_progress(self, progress: PatchProgress, snapshot: Callable[[], sitk.Image]) -> None:
        import SimpleITK as sitk

        self.publisher.publish(
            asdict(progress),
            lambda path: sitk.WriteImage(snapshot(), str(path), useCompression=True),
        )


class ModelOSettings(BaseSettings):
    """Server-owned native assets, bound into the selected worker by its executor."""

    model_config = SettingsConfigDict(
        env_prefix="MONAI_SERVICE_MODEL_O_", frozen=True, extra="forbid"
    )
    assets: Path
    queue_depth: Annotated[Literal[1, 2, 4], BeforeValidator(TypeAdapter(int).validate_python)] = 1


@dataclass(frozen=True)
class ModelOWorkflow:
    settings: ModelOSettings = field(default_factory=ModelOSettings)
    descriptor = ModelDescriptor(
        title="ModelO",
        description="ModelO vessel segmentation from CT and ordered voxel polylines on Dawn.",
        labels={"background": 0, "vessel": 1},
        dimension=3,
        input_profile="polylines",
    )

    def prepare(
        self,
        request: InferenceRequest[ModelOParameters],
        artifact: ResolvedArtifact,
        workspace: Path,
    ) -> Invocation:
        invocation = prepare_worker_files(request.image, workspace)
        parameters_path = workspace / "parameters.json"
        parameters_path.write_text(request.parameters.model_dump_json())
        return Invocation(
            arguments=(
                *invocation.arguments,
                "--model",
                FileArgument("model"),
                "--assets",
                FileArgument("assets"),
                "--parameters",
                FileArgument("parameters"),
                "--queue-depth",
                str(self.settings.queue_depth),
            ),
            bindings=(
                *invocation.bindings,
                FileBinding("model", artifact.path),
                FileBinding("assets", self.settings.assets),
                FileBinding("parameters", parameters_path),
            ),
        )

    def read_result(
        self, request: InferenceRequest[ModelOParameters], workspace: Path
    ) -> InferenceResult:
        result = read_worker_result(workspace)
        return InferenceResult(result.path, {**result.metadata, "test_only": False})

    def prepare_stream(
        self,
        request: InferenceRequest[ModelOParameters],
        artifact: ResolvedArtifact,
        workspace: Path,
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

    def infer(
        self,
        request: InferenceRequest[ModelOParameters],
        artifact: ResolvedArtifact,
        workspace: Path,
    ) -> InferenceResult:
        from model_o_inference.runtime import predict_model_o

        output = workspace / "mask.nii.gz"
        metadata = predict_model_o(
            request.image,
            artifact.path,
            output,
            request.parameters,
            self.settings.assets,
            queue_depth=self.settings.queue_depth,
        )
        return InferenceResult(output, {**metadata, "test_only": False})


def build_workflow() -> ModelOWorkflow:
    return ModelOWorkflow(settings=ModelOSettings())


TASK = TaskDefinition(
    identifier="model_o",
    key="MODEL_O",
    workflow_factory=build_workflow,
    parse_parameters=ModelOParameters.model_validate,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--preview-directory", type=Path)
    parser.add_argument("--queue-depth", type=int, choices=(1, 2, 4), default=1)
    arguments = parser.parse_args()
    parameters = ModelOParameters.model_validate_json(arguments.parameters.read_bytes())
    from model_o_inference.runtime import predict_model_o

    observer = None
    if arguments.preview_directory is not None:
        observer = _ModelOPreviews(PreviewPublisher(arguments.preview_directory))
    metadata = predict_model_o(
        arguments.input,
        arguments.model,
        arguments.output,
        parameters,
        arguments.assets,
        observer=observer,
        queue_depth=arguments.queue_depth,
    )
    serialized = json.dumps(metadata, allow_nan=False)
    if arguments.metadata is not None:
        arguments.metadata.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
