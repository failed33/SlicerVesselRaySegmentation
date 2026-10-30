"""Small client-observed updates, independent of image previews and model internals."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO


@dataclass(frozen=True)
class RequestProgress:
    message: str
    completed_bytes: int = 0
    total_bytes: int = 0
    bytes_per_second: float = 0
    endpoint_reachable: bool | None = None


ReportProgress = Callable[[RequestProgress], None]


def ignore_progress(_update: RequestProgress) -> None:
    pass


class UploadReader:
    """Count file bytes handed to HTTPX, keeping its native multipart encoder."""

    def __init__(self, source: BinaryIO, total: int, report: ReportProgress) -> None:
        self.source = source
        self.total = total
        self.report = report
        self.started = time.monotonic()
        self.updated = self.started
        self.report(RequestProgress("Uploading CT", total_bytes=total))

    def fileno(self) -> int:
        return self.source.fileno()

    def tell(self) -> int:
        return self.source.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.source.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        now = time.monotonic()
        if not chunk:
            self.report(RequestProgress("Upload sent. Waiting for server confirmation..."))
        elif now - self.updated >= 0.25 or self.tell() == self.total:
            completed = self.tell()
            self.report(
                RequestProgress(
                    "Uploading CT",
                    completed,
                    self.total,
                    completed / max(now - self.started, 0.001),
                )
            )
            self.updated = now
        return chunk
