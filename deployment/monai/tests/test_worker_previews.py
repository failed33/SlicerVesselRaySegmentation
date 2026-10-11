"""Bounded worker previews remain immutable until the stream releases them."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from monai_service.contracts import ExecutionError
from monai_service.worker_io import PreviewPublisher, read_worker_preview


def test_pending_preview_skips_image_creation_and_preserves_owned_file(tmp_path: Path) -> None:
    publisher = PreviewPublisher(tmp_path / "output/previews", interval_seconds=0)
    progress = {"completed": 1, "total": 5, "elapsed_seconds": 0.25}
    assert publisher.publish(progress, lambda path: path.write_bytes(b"first mask"))
    preview = read_worker_preview(tmp_path)
    assert preview is not None
    assert preview.metadata == {**progress, "revision": 1}
    skipped = Mock()
    assert not publisher.publish({**progress, "completed": 2}, skipped)
    skipped.assert_not_called()
    assert preview.path.read_bytes() == b"first mask"
    preview.release()
    assert read_worker_preview(tmp_path) is None
    assert publisher.publish(
        {**progress, "completed": 3}, lambda path: path.write_bytes(b"new mask")
    )
    newer = read_worker_preview(tmp_path)
    assert newer is not None
    assert newer.metadata["revision"] == 2
    assert newer.metadata["completed"] == 3
    assert newer.path.read_bytes() == b"new mask"
    newer.release()
    assert list(publisher.directory.iterdir()) == []


def test_preview_cadence_is_checked_before_expensive_image_creation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("monai_service.worker_io.time.monotonic", lambda: 100.0)
    publisher = PreviewPublisher(tmp_path / "output/previews", interval_seconds=5)
    progress = {"completed": 1, "total": 5, "elapsed_seconds": 0.25}
    publisher.publish(progress, lambda path: path.write_bytes(b"mask"))
    preview = read_worker_preview(tmp_path)
    assert preview is not None
    preview.release()
    skipped = Mock()
    assert not publisher.publish(progress, skipped)
    skipped.assert_not_called()
    monkeypatch.setattr("monai_service.worker_io.time.monotonic", lambda: 105.0)
    assert publisher.publish(progress, lambda path: path.write_bytes(b"next mask"))


def test_failed_snapshot_is_never_visible_to_consumer(tmp_path: Path) -> None:
    publisher = PreviewPublisher(tmp_path / "output/previews")

    def fail(path: Path) -> None:
        path.write_bytes(b"incomplete")
        assert read_worker_preview(tmp_path) is None
        raise OSError("write failed")

    with pytest.raises(OSError, match="write failed"):
        publisher.publish({"completed": 1, "total": 5, "elapsed_seconds": 1.0}, fail)
    assert list(publisher.directory.iterdir()) == []
    assert publisher.revision == 0


def test_invalid_worker_record_fails_at_file_boundary(tmp_path: Path) -> None:
    ready = tmp_path / "output/previews/ready"
    ready.mkdir(parents=True)
    (ready / "progress.json").write_text('{"revision": "not an integer"}')
    with pytest.raises(ExecutionError, match="preview metadata"):
        read_worker_preview(tmp_path)


def test_model_o_worker_flag_publishes_native_grid_preview(tmp_path, monkeypatch) -> None:
    import SimpleITK as sitk
    from model_o_inference.preview import PatchProgress

    from monai_service.models.model_o import main

    output = tmp_path / "output"
    output.mkdir()
    parameters = tmp_path / "parameters.json"
    parameters.write_text('{"polylines_zyx": [[[1, 2, 3], [4, 5, 6]]]}')
    mask = sitk.Image([4, 5, 6], sitk.sitkUInt8)
    mask.SetSpacing((0.5, 0.7, 1.2))
    mask.SetOrigin((8, 9, 10))
    mask[1, 2, 3] = 1

    def predict(*args, observer, queue_depth):
        assert queue_depth == 4
        observer.on_progress(PatchProgress(1, 2, 0.5), lambda: mask)
        sitk.WriteImage(mask, str(args[2]))
        return {"patches": 2}

    monkeypatch.setattr("model_o_inference.runtime.predict_model_o", predict)
    monkeypatch.setattr(
        "sys.argv",
        [
            "model_o",
            "--input",
            str(tmp_path / "ct.nrrd"),
            "--model",
            str(tmp_path / "model"),
            "--assets",
            str(tmp_path / "assets"),
            "--parameters",
            str(parameters),
            "--output",
            str(output / "mask.nii.gz"),
            "--metadata",
            str(output / "metadata.json"),
            "--preview-directory",
            str(output / "previews"),
            "--queue-depth",
            "4",
        ],
    )
    main()
    preview = read_worker_preview(tmp_path)
    assert preview is not None
    restored = sitk.ReadImage(str(preview.path))
    assert restored.IsSameImageGeometryAs(mask)
    assert restored[1, 2, 3] == 1
    assert preview.metadata["completed"] == 1
    assert (output / "mask.nii.gz").is_file()
    assert (output / "metadata.json").read_text() == '{"patches": 2}'
