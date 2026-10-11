"""ARG ONNX task, native-plans policy, workflow and worker CLI.

The shared image pipeline preserves the inspected fine-tuning plans. The graph
is exported separately; serving does not load checkpoints or training code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, StrictStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from monai_service.contracts import (
    FileArgument,
    FileBinding,
    InferenceRequest,
    InferenceResult,
    Invocation,
    ModelDescriptor,
    ResolvedArtifact,
    TaskDefinition,
)
from monai_service.schemas.base import FrozenParameters, NoParameters


class ArgPolicy(FrozenParameters):
    """Worker policy, independent of process launch and artifact lookup."""

    device: Literal["auto", "cpu", "gpu", "npu"] = "auto"
    model_file: Annotated[StrictStr, Field(min_length=1)] = "model.onnx"

    @field_validator("model_file")
    @classmethod
    def model_is_filename(cls, value: str) -> str:
        if (
            value in {".", ".."}
            or PurePosixPath(value).name != value
            or "\\" in value
            or "\0" in value
            or not value.endswith(".onnx")
        ):
            raise ValueError("model_file must be an ONNX filename inside the model directory")
        return value


class ArgSettings(BaseSettings, ArgPolicy):
    """Server-selected policy read from the ARG environment prefix."""

    model_config = SettingsConfigDict(env_prefix="MONAI_SERVICE_ARG_", frozen=True, extra="forbid")


@dataclass(frozen=True)
class ArgWorkflow:
    policy: ArgPolicy = field(default_factory=ArgPolicy)
    descriptor = ModelDescriptor(
        title="ARG stage 1",
        description="ARG binary vessel segmentation using ONNX and native nnU-Net plans.",
        labels={"background": 0, "vessel": 1},
        dimension=3,
    )

    def prepare(
        self, request: InferenceRequest[NoParameters], artifact: ResolvedArtifact, workspace: Path
    ) -> Invocation:
        from monai_service.worker_io import prepare_worker_files

        invocation = prepare_worker_files(request.image, workspace)
        return Invocation(
            arguments=(
                *invocation.arguments,
                "--model",
                FileArgument("model"),
                "--device",
                self.policy.device,
                "--model-file",
                self.policy.model_file,
            ),
            bindings=(*invocation.bindings, FileBinding("model", artifact.path)),
        )

    def infer(
        self, request: InferenceRequest[NoParameters], artifact: ResolvedArtifact, workspace: Path
    ) -> InferenceResult:
        output = workspace / "mask.nii.gz"
        metadata = predict_arg(request.image, artifact.path, output, self.policy)
        return InferenceResult(output, {**metadata, "test_only": False})

    def read_result(
        self, request: InferenceRequest[NoParameters], workspace: Path
    ) -> InferenceResult:
        from monai_service.worker_io import read_worker_result

        result = read_worker_result(workspace)
        return InferenceResult(result.path, {**result.metadata, "test_only": False})


def build_workflow() -> ArgWorkflow:
    return ArgWorkflow(policy=ArgSettings())


TASK = TaskDefinition(
    identifier="arg_vessel",
    key="ARG",
    workflow_factory=build_workflow,
    parse_parameters=NoParameters.model_validate,
)


def _validate_dataset(path: Path) -> None:
    """The exported graph serves only a single-channel binary segmentation."""
    dataset = json.loads(path.read_text())
    if (
        not isinstance(dataset, dict)
        or not isinstance(dataset.get("channel_names"), dict)
        or set(dataset["channel_names"]) != {"0"}
        or not isinstance(dataset.get("labels"), dict)
        or set(dataset["labels"].values()) != {0, 1}
    ):
        raise ValueError("ARG dataset requires one input channel and binary labels")


def predict_arg(
    input_path: Path, model_path: Path, output_path: Path, policy: ArgPolicy
) -> dict[str, object]:
    """Run the shared ONNX pipeline with one immutable artifact directory."""
    from arg_inference.plans import load_plan
    from arg_inference.runtime import predict_file

    _validate_dataset(model_path / "dataset.json")
    plan = load_plan(model_path / "plans.json")
    result = predict_file(
        input_path,
        output_path,
        model_path / policy.model_file,
        plan,
        device=policy.device,
    )
    return {
        **result,
        "worker": "arg",
        "preprocessing": "native-plans",
        "orientation": "LPS",
        "model_file": policy.model_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "npu"), default="auto")
    parser.add_argument("--model-file", default="model.onnx")
    arguments = parser.parse_args()
    metadata = predict_arg(
        arguments.input,
        arguments.model,
        arguments.output,
        ArgPolicy(device=arguments.device, model_file=arguments.model_file),
    )
    encoded_metadata = json.dumps(metadata)
    if arguments.metadata is not None:
        arguments.metadata.write_text(encoded_metadata)
    print(encoded_metadata)


if __name__ == "__main__":
    main()
