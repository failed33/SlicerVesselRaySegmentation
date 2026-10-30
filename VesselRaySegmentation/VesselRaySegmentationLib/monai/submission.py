"""Submit a captured volume through MONAI Label without accessing the live scene."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .client import (
    ConnectionSettings,
    InferenceResponse,
    MonaiClient,
    MonaiClientError,
)
from .images import IMAGE_SUFFIXES, VolumeSnapshot
from .progress import ReportProgress, RequestPhase, RequestProgress, ignore_progress
from .sessions import VolumeSession

_PROVENANCE_FIELDS = (
    "model",
    "model_name",
    "model_version",
    "model_artifact",
    "runtime",
    "runtime_reference",
    "test_only",
)


@dataclass(frozen=True)
class InferenceRequest:
    snapshot: VolumeSnapshot
    connection: ConnectionSettings
    model: str
    parameters: dict[str, Any]
    input_profile: str
    output_name: str = "MONAI segmentation"
    session: VolumeSession | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class InferenceResult:
    mask: np.ndarray = field(repr=False)
    request: InferenceRequest
    metadata: dict[str, Any]

    def provenance(self) -> str:
        """Serialize submitted identity and allowed model metadata without credentials."""
        snapshot = self.request.snapshot
        values = {
            "task": self.request.model,
            "source_volume_id": snapshot.source_id,
            "source_volume_name": snapshot.source_name,
            "submitted_ijk_to_ras": snapshot.geometry.ijk_to_ras().tolist(),
            "submitted_shape_zyx": snapshot.geometry.shape_zyx,
            "server": {
                key: self.metadata[key] for key in _PROVENANCE_FIELDS if key in self.metadata
            },
        }
        token = self.request.connection.bearer_token
        if token:
            values = _redact_strings(values, token)
        return json.dumps(values, allow_nan=False)


def _read_result(
    request: InferenceRequest, response: InferenceResponse, directory: Path
) -> InferenceResult:
    suffix = next((value for value in IMAGE_SUFFIXES if response.filename.endswith(value)), None)
    if suffix is None:
        raise MonaiClientError("MONAI Label returned an unsupported segmentation image format")
    output_path = directory / f"result{suffix}"
    output_path.write_bytes(response.image)
    mask = request.snapshot.read_mask(output_path)
    return InferenceResult(mask, request, response.metadata)


def run_inference(
    request: InferenceRequest, report: ReportProgress = ignore_progress
) -> InferenceResult:
    """Perform one native Label request without touching the live MRML scene."""
    report(RequestProgress(RequestPhase.VALIDATING))
    with (
        MonaiClient(request.connection, report) as client,
        TemporaryDirectory(prefix="vessel-monai-") as temporary,
    ):
        model = client.models().get(request.model)
        if model is None or model.input_profile != request.input_profile:
            raise MonaiClientError(
                "Model input requirements changed; refresh models before inference"
            )
        directory = Path(temporary)
        if request.session is not None:
            if (
                request.session.snapshot is not request.snapshot
                or request.session.connection != request.connection
            ):
                raise ValueError("MONAI image session does not match the submitted CT and endpoint")
            with request.session.use(client) as image:
                response = client.infer(image, request.model, request.parameters)
        else:
            image = directory / "ct.nrrd"
            report(RequestProgress(RequestPhase.PREPARING))
            request.snapshot.write_image(image)
            response = client.infer(image, request.model, request.parameters)
        return _read_result(request, response, directory)


def _redact_strings(value: Any, token: str) -> Any:
    if isinstance(value, str):
        return value.replace(token, "[redacted]")
    if isinstance(value, dict):
        return {key: _redact_strings(item, token) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_strings(item, token) for item in value]
    return value
