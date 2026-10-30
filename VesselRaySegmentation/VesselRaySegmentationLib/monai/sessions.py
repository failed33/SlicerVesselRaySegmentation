"""One native Label upload shared by requests for an immutable CT snapshot."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .client import ConnectionSettings, MonaiClient, MonaiClientError, SessionInput
from .images import VolumeSnapshot
from .progress import RequestPhase, RequestProgress

logger = logging.getLogger(__name__)


class VolumeSession:
    def __init__(self, connection: ConnectionSettings, snapshot: VolumeSnapshot) -> None:
        self.connection = connection
        self.snapshot = snapshot
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._session_id: str | None = None
        self._uncertain = False

    def _ensure_session(self, client: MonaiClient) -> str:
        if self._session_id is not None and client.session_exists(self._session_id):
            client.report(RequestProgress(RequestPhase.SESSION_READY))
            return self._session_id
        self._session_id = None
        with TemporaryDirectory(prefix="vessel-monai-upload-") as temporary:
            path = Path(temporary) / "ct.nrrd"
            client.report(RequestProgress(RequestPhase.PREPARING))
            self.snapshot.write_image(path)
            self._session_id = client.create_session(path)
        self._uncertain = False
        return self._session_id

    @contextmanager
    def use(self, client: MonaiClient) -> Iterator[SessionInput]:
        """Serialize access and retain the upload until its request has finished."""
        with self._lock:
            if self._closed.is_set():
                raise MonaiClientError("MONAI image session is closed")
            client.report(RequestProgress(RequestPhase.CHECKING_SESSION))
            session_id = self._ensure_session(client)
            if self._closed.is_set():
                raise MonaiClientError("MONAI image session is closed")
            try:
                yield SessionInput(session_id)
            except BaseException:
                # Even an HTTP error may come from a gateway while native work
                # continues. Only successful requests prove remote completion.
                self._uncertain = True
                raise

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            session_id, self._session_id = self._session_id, None
            if session_id is not None and not self._uncertain:
                connection = replace(
                    self.connection, timeout_seconds=min(5, self.connection.timeout_seconds)
                )
                MonaiClient(connection).delete_session(session_id)

    def close_in_background(self) -> None:
        """Dispose without blocking the Slicer UI or interrupting an active reader."""
        self._closed.set()

        def release() -> None:
            try:
                self.close()
            except MonaiClientError as error:
                logger.warning("Could not delete MONAI image session: %s", error)

        threading.Thread(target=release, name="monai-session-cleanup", daemon=True).start()
