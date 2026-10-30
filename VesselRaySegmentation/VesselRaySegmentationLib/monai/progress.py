"""Transport facts shared by desktop and browser, with presentation owned by the UI."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto


class RequestPhase(Enum):
    VALIDATING = auto()
    PREPARING = auto()
    CHECKING_SESSION = auto()
    SESSION_READY = auto()
    UPLOADING = auto()
    WAITING = auto()
    RECEIVING = auto()
    READING = auto()


@dataclass(frozen=True)
class RequestProgress:
    phase: RequestPhase
    completed_bytes: int = 0
    total_bytes: int = 0
    bytes_per_second: float = 0


ReportProgress = Callable[[RequestProgress], None]


def ignore_progress(_update: RequestProgress) -> None:
    pass


def upload_chunks(chunks: Iterable[bytes], total: int, report: ReportProgress) -> Iterator[bytes]:
    """Observe HTTPX's encoded body using its documented content-generator pattern."""
    started = updated = time.monotonic()
    completed = 0
    report(RequestProgress(RequestPhase.UPLOADING, total_bytes=total))
    for chunk in chunks:
        yield chunk
        completed += len(chunk)
        now = time.monotonic()
        if now - updated >= 0.25 or completed == total:
            elapsed = max(now - started, 0.001)
            report(
                RequestProgress(
                    RequestPhase.UPLOADING,
                    completed_bytes=completed,
                    total_bytes=total,
                    bytes_per_second=completed / elapsed,
                )
            )
            updated = now
    report(RequestProgress(RequestPhase.WAITING))
