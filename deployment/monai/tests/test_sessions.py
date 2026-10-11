"""Native Label session lifecycle with synchronized expiry and refresh."""

import json
import threading
from pathlib import Path

import pytest
from label_inference.sessions import SynchronizedSessions
from monailabel.config import settings
from monailabel.interfaces.utils.app import app_instance, clear_cache
from monailabel.utils import sessions as native_sessions

from monai_service.config import ServiceSettings
from tests.fixtures import create_test_app


def test_native_lifecycle_and_expiry_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(native_sessions.time, "time", lambda: 1000)
    owner = SynchronizedSessions(str(tmp_path / "sessions"), expiry=120)
    upload = tmp_path / "ct.nrrd"
    upload.write_bytes(b"native-ct")
    identifier, info = owner.add_session(str(upload), expiry=200, session_id="test-session")
    assert identifier == "test-session" and info.expiry == 120
    assert not upload.exists() and Path(info.image).read_bytes() == b"native-ct"

    monkeypatch.setattr(native_sessions.time, "time", lambda: 1050)
    assert owner.get_session(identifier, update_ts=False).last_access_ts == 1000
    assert owner.get_session(identifier).last_access_ts == 1050
    metadata = json.loads((Path(info.path) / "meta.info").read_text())
    assert metadata["last_access_ts"] == 1050

    assert owner.remove_expired() == 0
    monkeypatch.setattr(native_sessions.time, "time", lambda: 1171)
    assert owner.remove_expired() == 1
    assert owner.get_session(identifier) is None and not Path(info.image).exists()


def test_expiry_and_refresh_cannot_interleave_after_expiry_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(native_sessions.time, "time", lambda: 1000)
    owner = SynchronizedSessions(str(tmp_path / "sessions"), expiry=61)
    upload = tmp_path / "ct.nrrd"
    upload.write_bytes(b"native-ct")
    identifier, info = owner.add_session(str(upload))
    monkeypatch.setattr(native_sessions.time, "time", lambda: 1100)
    removing, release, refresh_attempted = threading.Event(), threading.Event(), threading.Event()
    refreshed = threading.Event()
    result, errors = [], []
    remove_session = native_sessions.Sessions.remove_session

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread().name == "session-refresh":
                refresh_attempted.set()
            return lock.__enter__()

        def __exit__(self, *args):
            return lock.__exit__(*args)

    lock = owner._session_lock
    monkeypatch.setattr(owner, "_session_lock", ObservedLock())

    def pause_removal(self, session_id):
        removing.set()
        assert release.wait(2)
        remove_session(self, session_id)

    def expire():
        try:
            assert owner.remove_expired() == 1
        except BaseException as error:
            errors.append(error)

    def refresh():
        try:
            result.append(owner.get_session(identifier))
        except BaseException as error:
            errors.append(error)
        finally:
            refreshed.set()

    monkeypatch.setattr(native_sessions.Sessions, "remove_session", pause_removal)
    cleanup = threading.Thread(target=expire)
    lookup = threading.Thread(target=refresh, name="session-refresh")
    cleanup.start()
    try:
        assert removing.wait(2)
        lookup.start()
        assert refresh_attempted.wait(2)
        assert not refreshed.is_set()
    finally:
        release.set()
        cleanup.join(2)
        if lookup.ident is not None:
            lookup.join(2)
    assert not cleanup.is_alive() and not lookup.is_alive() and not errors
    assert result == [None] and not Path(info.image).exists()


@pytest.mark.parametrize("configured_path", [False, True])
def test_native_app_loader_registers_synchronized_sessions(tmp_path, monkeypatch, configured_path):
    store_path = str(tmp_path / "configured-sessions") if configured_path else ""
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSION_PATH", store_path)
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSION_EXPIRY", 123)
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSIONS", True)
    monkeypatch.setattr(settings, "MONAI_LABEL_STUDIES", str(tmp_path / "studies"))
    monkeypatch.setattr(settings, "MONAI_LABEL_DATASTORE_AUTO_RELOAD", False)
    create_test_app(ServiceSettings(work_dir=tmp_path))
    try:
        instance = app_instance()
        owner = instance.sessions()
        assert isinstance(owner, SynchronizedSessions)
        assert owner.store_path == (store_path or str(tmp_path / "sessions"))
        assert owner.expiry == 123
        assert instance._load_sessions(False) is None
    finally:
        clear_cache()


def test_disabled_sessions_keep_native_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSIONS", False)
    monkeypatch.setattr(settings, "MONAI_LABEL_STUDIES", str(tmp_path / "studies"))
    monkeypatch.setattr(settings, "MONAI_LABEL_DATASTORE_AUTO_RELOAD", False)

    def unexpected_cleanup(_sessions):
        pytest.fail("Disabled sessions attempted expiry cleanup")

    monkeypatch.setattr(SynchronizedSessions, "remove_expired", unexpected_cleanup)
    create_test_app(ServiceSettings(work_dir=tmp_path))
    try:
        instance = app_instance()
        assert instance.sessions() is None
        instance.cleanup_sessions()
        assert instance.sessions() is None
    finally:
        clear_cache()


def test_native_app_restart_cleans_persisted_expiry_with_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSION_PATH", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSION_EXPIRY", 61)
    monkeypatch.setattr(settings, "MONAI_LABEL_SESSIONS", True)
    monkeypatch.setattr(settings, "MONAI_LABEL_STUDIES", str(tmp_path / "studies"))
    monkeypatch.setattr(settings, "MONAI_LABEL_DATASTORE_AUTO_RELOAD", False)
    create_test_app(ServiceSettings(work_dir=tmp_path))
    try:
        initial = app_instance()
        upload = tmp_path / "ct.nrrd"
        upload.write_bytes(b"native-ct")
        with monkeypatch.context() as clock:
            clock.setattr(native_sessions.time, "time", lambda: 1000)
            identifier, info = initial.sessions().add_session(str(upload))
        assert Path(info.image).exists()

        clear_cache()
        restarted = app_instance()
        owner = restarted.sessions()
        assert restarted is not initial and isinstance(owner, SynchronizedSessions)
        assert len(owner) == 0 and Path(info.image).exists()
        with monkeypatch.context() as clock:
            clock.setattr(native_sessions.time, "time", lambda: 1100)
            restarted.cleanup_sessions()
        assert not Path(info.path).exists() and owner.get_session(identifier) is None
    finally:
        clear_cache()
