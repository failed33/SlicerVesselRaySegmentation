"""One native Label adapter for model-independent inference services."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Generic

from fastapi import HTTPException
from label_inference.requests import parameter_errors, request_parameters
from monailabel.interfaces.tasks.infer_v2 import InferTask
from pydantic import ValidationError

from monai_service.application import InferenceService
from monai_service.contracts import (
    ArtifactUnavailable,
    ExecutionError,
    ExecutionTimeout,
    InferenceRequest,
    InvalidRequest,
    Parameters,
)


class LabelInferTask(InferTask, Generic[Parameters]):
    def __init__(self, service: InferenceService[Parameters]) -> None:
        descriptor = service.descriptor
        super().__init__(
            type=descriptor.task_type,
            labels=dict(descriptor.labels),
            dimension=descriptor.dimension,
            description=descriptor.description,
            config={
                "input_profile": descriptor.input_profile,
                "streaming": service.supports_streaming,
            },
        )
        self.service = service

    def is_valid(self) -> bool:
        return True

    def info(self) -> dict[str, Any]:
        return {**super().info(), "title": self.service.descriptor.title}

    def __call__(self, request: Mapping[str, Any]) -> tuple[str, dict[str, object]]:
        parameters = request_parameters(request)
        try:
            parsed_parameters = self.service.deployment.parse_parameters(parameters)
            result = self.service.infer(InferenceRequest(Path(request["image"]), parsed_parameters))
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=parameter_errors(error)) from error
        except InvalidRequest as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ExecutionTimeout as error:
            raise HTTPException(status_code=504, detail=str(error)) from error
        except (ArtifactUnavailable, ExecutionError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return str(result.path), dict(result.metadata)
