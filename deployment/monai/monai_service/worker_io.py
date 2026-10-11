"""File arguments and JSON results shared by the image-worker CLIs."""

import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from monai_service.contracts import (
    ExecutionError,
    FileArgument,
    FileBinding,
    InferencePreview,
    InferenceResult,
    Invocation,
)

_metadata_object = TypeAdapter(dict[str, JsonValue])


class PreviewRecord(BaseModel):
    """Metadata published with one complete worker snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: Annotated[int, Field(strict=True, ge=1)]
    completed: Annotated[int, Field(strict=True, ge=0)]
    total: Annotated[int, Field(strict=True, ge=0)]
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)]


class PreviewPublisher:
    """Publish at most one immutable pending preview in the output binding.

    The consumer removes ``ready`` after transmission. Skipping publication while
    occupied bounds disk usage and avoids slowing GPU inference for a slow client.
    """

    def __init__(self, directory: Path, interval_seconds: float = 5.0) -> None:
        self.directory = directory
        self.interval_seconds = interval_seconds
        self.revision = 0
        self.last_published = float("-inf")
        self.directory.mkdir(parents=True, exist_ok=True)

    def publish(self, progress: Mapping[str, object], write_image: Callable[[Path], None]) -> bool:
        now = time.monotonic()
        ready = self.directory / "ready"
        if now - self.last_published < self.interval_seconds or ready.exists():
            return False
        record = PreviewRecord.model_validate({**progress, "revision": self.revision + 1})
        with tempfile.TemporaryDirectory(prefix="snapshot-", dir=self.directory) as temporary:
            staging = Path(temporary)
            write_image(staging / "image.nii.gz")
            (staging / "progress.json").write_text(record.model_dump_json())
            staging.rename(ready)
        self.revision = record.revision
        self.last_published = time.monotonic()
        return True


def read_worker_preview(workspace: Path) -> InferencePreview | None:
    """Acquire the pending snapshot; its caller releases it after transmission."""
    ready = workspace / "output" / "previews" / "ready"
    if not ready.is_dir():
        return None
    try:
        record = PreviewRecord.model_validate_json((ready / "progress.json").read_bytes())
    except (OSError, ValidationError) as error:
        raise ExecutionError("Inference preview metadata is invalid") from error
    return InferencePreview(
        ready / "image.nii.gz", record.model_dump(), partial(shutil.rmtree, ready)
    )


def prepare_worker_files(image: Path, workspace: Path) -> Invocation:
    """Bind Label's existing input and prepare the worker's output directory."""
    output = workspace / "output"
    output.mkdir()
    return Invocation(
        arguments=(
            "--input",
            FileArgument("image"),
            "--output",
            FileArgument("output", PurePosixPath("mask.nii.gz")),
            "--metadata",
            FileArgument("output", PurePosixPath("metadata.json")),
        ),
        bindings=(FileBinding("image", image), FileBinding("output", output, writable=True)),
    )


def read_worker_result(workspace: Path) -> InferenceResult:
    """Read the worker's JSON metadata; the application transfers its result file."""
    output = workspace / "output"
    try:
        metadata = _metadata_object.validate_json((output / "metadata.json").read_bytes())
    except (OSError, ValidationError) as error:
        raise ExecutionError("Inference result metadata must be a readable JSON object") from error
    return InferenceResult(output / "mask.nii.gz", metadata)
