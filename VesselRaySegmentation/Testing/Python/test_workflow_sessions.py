"""Native session reuse, expiry and client-side request ownership."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from VesselRaySegmentationLib.monai import sessions
from VesselRaySegmentationLib.monai.client import ConnectionSettings, MonaiClientError
from VesselRaySegmentationLib.monai.images import VolumeSnapshot


@pytest.fixture
def session(monkeypatch):
    state = SimpleNamespace(uploads=0, writes=0, current=None, deleted=[], lookup_error=None)

    class Client:
        def __init__(self, connection):
            self.connection = connection

        def create_session(self, path):
            assert path.read_bytes() == b"captured-ct"
            state.uploads += 1
            state.current = f"session-{state.uploads}"
            return state.current

        def session_exists(self, session_id):
            if state.lookup_error is not None:
                raise state.lookup_error
            return session_id == state.current

        def delete_session(self, session_id):
            state.deleted.append(session_id)

    def write_image(_snapshot, path):
        state.writes += 1
        path.write_bytes(b"captured-ct")

    monkeypatch.setattr(sessions, "MonaiClient", Client)
    monkeypatch.setattr(VolumeSnapshot, "write_image", write_image)
    snapshot = VolumeSnapshot.capture(np.zeros((2, 3, 4)), np.eye(4), ("ct-id", "CT"))
    return sessions.VolumeSession(ConnectionSettings(), snapshot), state


def test_unchanged_volume_is_serialized_and_uploaded_once(session):
    owner, state = session
    for _ in range(3):
        with owner.use() as (_client, image):
            assert image.session_id == "session-1"
    assert state.uploads == state.writes == 1
    owner.close()
    owner.close()
    assert state.deleted == ["session-1"]


def test_expired_session_reuploads_original_snapshot(session):
    owner, state = session
    with owner.use():
        pass
    state.current = None
    with owner.use() as (_client, image):
        assert image.session_id == "session-2"
    assert state.uploads == state.writes == 2


def test_access_failure_does_not_reupload_or_hide_the_error(session):
    owner, state = session
    with owner.use():
        pass
    state.lookup_error = MonaiClientError("Access denied", status_code=403)
    with pytest.raises(MonaiClientError, match="Access denied"):
        with owner.use():
            pytest.fail("Inference started despite failed session lookup")
    assert state.uploads == 1


@pytest.mark.parametrize(
    "error",
    [MonaiClientError("Request timed out"), ValueError("Bad preview")]
    + [MonaiClientError("Gateway failure", status_code=status) for status in (502, 503, 504)],
)
def test_uncertain_remote_completion_leaves_upload_to_native_expiry(session, error):
    owner, state = session
    with pytest.raises(type(error), match=str(error)):
        with owner.use():
            raise error
    owner.close()
    assert state.deleted == []


def test_later_success_does_not_certify_an_earlier_failed_request_completed(session):
    owner, state = session
    with pytest.raises(MonaiClientError, match="Gateway failure"):
        with owner.use():
            raise MonaiClientError("Gateway failure", status_code=504)
    with owner.use() as (_client, image):
        assert image.session_id == "session-1"
    owner.close()
    assert state.uploads == 1 and state.deleted == []


def test_recreated_session_has_independent_completion_ownership(session):
    owner, state = session
    with pytest.raises(MonaiClientError, match="Gateway failure"):
        with owner.use():
            raise MonaiClientError("Gateway failure", status_code=504)
    state.current = None
    with owner.use() as (_client, image):
        assert image.session_id == "session-2"
    owner.close()
    assert state.deleted == ["session-2"]


def test_lookup_failure_does_not_make_an_unused_session_uncertain(session):
    owner, state = session
    with owner.use():
        pass
    state.lookup_error = MonaiClientError("Gateway failure", status_code=504)
    with pytest.raises(MonaiClientError, match="Gateway failure"):
        with owner.use():
            pytest.fail("Inference started despite failed session lookup")
    owner.close()
    assert state.deleted == ["session-1"]


def test_close_during_upload_does_not_start_inference(session, monkeypatch):
    owner, state = session
    uploading, release, deleted = threading.Event(), threading.Event(), threading.Event()
    create_session = sessions.MonaiClient.create_session
    delete_session = sessions.MonaiClient.delete_session
    submitted, errors = [], []

    def blocked_upload(client, path):
        uploading.set()
        assert release.wait(2)
        return create_session(client, path)

    def delete(client, session_id):
        delete_session(client, session_id)
        deleted.set()

    def infer():
        try:
            with owner.use():
                submitted.append(True)
        except MonaiClientError as error:
            errors.append(str(error))

    monkeypatch.setattr(sessions.MonaiClient, "create_session", blocked_upload)
    monkeypatch.setattr(sessions.MonaiClient, "delete_session", delete)
    worker = threading.Thread(target=infer)
    worker.start()
    assert uploading.wait(2)
    owner.close_in_background()
    assert state.deleted == []
    release.set()
    worker.join(2)
    assert not worker.is_alive() and deleted.wait(2)
    assert submitted == [] and errors == ["MONAI image session is closed"]
    assert state.uploads == 1 and state.deleted == ["session-1"]


def test_close_rejects_a_request_waiting_for_the_active_session(session, monkeypatch):
    owner, state = session
    active, waiting, release = threading.Event(), threading.Event(), threading.Event()
    deleted = threading.Event()
    delete_session = sessions.MonaiClient.delete_session
    submitted, errors = [], []

    def delete(client, session_id):
        delete_session(client, session_id)
        deleted.set()

    def active_request():
        with owner.use():
            active.set()
            assert release.wait(2)

    def waiting_request():
        waiting.set()
        try:
            with owner.use():
                submitted.append(True)
        except MonaiClientError as error:
            errors.append(str(error))

    monkeypatch.setattr(sessions.MonaiClient, "delete_session", delete)
    workers = [threading.Thread(target=active_request), threading.Thread(target=waiting_request)]
    workers[0].start()
    assert active.wait(2)
    workers[1].start()
    assert waiting.wait(2)
    owner.close_in_background()
    assert state.deleted == []
    release.set()
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()
    assert deleted.wait(2)
    assert submitted == [] and errors == ["MONAI image session is closed"]
    assert state.uploads == 1 and state.deleted == ["session-1"]


def test_close_waits_for_active_request_before_deleting(session):
    owner, state = session
    entered, release = threading.Event(), threading.Event()

    def infer():
        with owner.use():
            entered.set()
            assert release.wait(2)

    worker = threading.Thread(target=infer)
    worker.start()
    assert entered.wait(2)
    cleanup = threading.Thread(target=owner.close)
    cleanup.start()
    assert state.deleted == []
    release.set()
    worker.join(2)
    cleanup.join(2)
    assert not worker.is_alive() and not cleanup.is_alive()
    assert state.deleted == ["session-1"]
    with pytest.raises(MonaiClientError, match="closed"):
        with owner.use():
            pass
