"""Serialize native Label session files and expiry within one server process."""

import threading

from monailabel.utils.sessions import SessionInfo, Sessions


class SynchronizedSessions(Sessions):
    """Retain native session policy while coordinating its read/write operations."""

    def __init__(self, store_path: str = "", expiry: int = 3600) -> None:
        self._session_lock = threading.RLock()
        super().__init__(store_path, expiry)

    def get_session(
        self, session_id: str, update_ts: bool = True, fetch_cache: bool = True
    ) -> SessionInfo | None:
        with self._session_lock:
            return super().get_session(session_id, update_ts, fetch_cache)

    def add_session(
        self,
        data_file: str,
        expiry: int = 0,
        uncompress: bool = False,
        session_id: str | None = None,
    ) -> tuple[str, SessionInfo]:
        with self._session_lock:
            return super().add_session(data_file, expiry, uncompress, session_id)

    def remove_session(self, session_id: str) -> None:
        with self._session_lock:
            super().remove_session(session_id)

    def remove_expired(self) -> int:
        with self._session_lock:
            return super().remove_expired()
