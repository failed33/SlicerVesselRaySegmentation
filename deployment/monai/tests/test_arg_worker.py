"""Shared ONNX and hosted whole-volume checks with a small test-only graph."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import SimpleITK as sitk
from arg_inference.runtime import create_session, load_plan, preprocess, sliding_logits
from onnx import TensorProto, helper, numpy_helper

from monai_service.models.arg import ArgPolicy, predict_arg


def _write_graph(model_path: Path, *, foreground: bool = True, **metadata: object) -> None:
    logits = np.zeros((1, 2, 8, 8, 8), dtype=np.float16)
    logits[:, int(foreground)] = 1
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["stored_logits"], value=numpy_helper.from_array(logits)
            ),
            helper.make_node("Cast", ["stored_logits"], ["logits"], to=TensorProto.FLOAT),
        ],
        "test_only_binary_mixed_precision",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 8, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, 8, 8, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    properties = {
        "preprocessing": "native-plans",
        "configuration": "3d_fullres",
        "patch_size": [8, 8, 8],
        "precision": "mixed-fp16",
        "plans_sha256": hashlib.sha256((model_path / "plans.json").read_bytes()).hexdigest(),
        **metadata,
    }
    helper.set_model_props(model, {key: json.dumps(value) for key, value in properties.items()})
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, model_path / "model.onnx")


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "model"
    model_path.mkdir()
    kwargs = {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None}
    plans = {
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {
            "3d_fullres": {
                "preprocessor_name": "DefaultPreprocessor",
                "patch_size": [8, 8, 8],
                "spacing": [1, 1, 1],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": kwargs,
                "resampling_fn_probabilities_kwargs": kwargs,
            }
        },
    }
    (model_path / "plans.json").write_text(json.dumps(plans))
    (model_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "vessel": 1}})
    )
    _write_graph(model_path)
    return model_path


def _write_ct(tmp_path: Path, voxels: np.ndarray | None = None) -> Path:
    if voxels is None:
        voxels = np.arange(1, 513, dtype=np.float32).reshape(8, 8, 8)
    input_path = tmp_path / "ct.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(voxels), str(input_path))
    return input_path


def test_whole_prediction_restores_oblique_input_grid(tiny_model: Path, tmp_path: Path) -> None:
    voxels = np.zeros((8, 10, 12), dtype=np.float32)
    voxels[1:7, 2:8, 3:10] = np.arange(1, 6 * 6 * 7 + 1).reshape(6, 6, 7)
    image = sitk.GetImageFromArray(voxels)
    angle = math.pi / 12
    image.SetDirection(
        (-math.cos(angle), math.sin(angle), 0, -math.sin(angle), -math.cos(angle), 0, 0, 0, 1)
    )
    image.SetSpacing((0.8, 1.1, 1.4))
    image.SetOrigin((17.3, -8.6, 5.2))
    input_path, output_path = tmp_path / "ct.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(input_path))
    metadata = predict_arg(input_path, tiny_model, output_path, ArgPolicy(device="cpu"))
    mask, submitted = sitk.ReadImage(str(output_path)), sitk.ReadImage(str(input_path))
    assert np.array_equal(sitk.GetArrayFromImage(mask), voxels > 0)
    assert mask.GetOrigin() == pytest.approx(submitted.GetOrigin(), rel=0, abs=1e-6)
    assert mask.GetSpacing() == pytest.approx(submitted.GetSpacing(), rel=0, abs=1e-6)
    assert mask.GetDirection() == pytest.approx(submitted.GetDirection(), rel=0, abs=1e-6)
    assert mask.GetSize() == submitted.GetSize()
    assert mask.GetPixelID() == sitk.sitkUInt8
    assert metadata["preprocessing"] == "native-plans"
    assert metadata["inference_runtime"] == "onnxruntime"
    assert metadata["model_precision"] == "mixed-fp16"
    assert metadata["model_file"] == "model.onnx"
    assert metadata["device_preference"] == "cpu"
    assert metadata["providers"] == ["CPUExecutionProvider"]
    assert metadata["accumulation"] == "native-fp16"
    assert (
        metadata["model_sha256"]
        == hashlib.sha256((tiny_model / "model.onnx").read_bytes()).hexdigest()
    )
    assert (
        metadata["plans_sha256"]
        == hashlib.sha256((tiny_model / "plans.json").read_bytes()).hexdigest()
    )
    assert not {"trainer", "checkpoint", "use_mirroring"} & metadata.keys()


def test_shared_runtime_needs_no_training_or_export_imports(
    tiny_model: Path, tmp_path: Path
) -> None:
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "nnunetv2", "monai", "monailabel", "onnx"}:
            raise AssertionError(f"Serving imported {fullname}")
        return None

sys.meta_path.insert(0, BlockTrainingImports())
from arg_inference.plans import load_plan
from arg_inference.runtime import predict_file
model = Path(sys.argv[2])
result = predict_file(Path(sys.argv[1]), Path(sys.argv[3]), model / "model.onnx", load_plan(model / "plans.json"), device="cpu")
assert result["inference_runtime"] == "onnxruntime"
"""
    environment = {
        **os.environ,
        "ORT_DISABLE_TELEMETRY": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "inference" / "src"),
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(_write_ct(tmp_path)),
            str(tiny_model),
            str(tmp_path / "mask.nrrd"),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_policy_crops_raw_zeros_before_normalizing(tiny_model: Path) -> None:
    raw = np.zeros((1, 5, 5, 5), dtype=np.float32)
    raw[:, 1:4, 1:4, 1:4] = np.arange(1, 28).reshape(1, 3, 3, 3)
    prepared = preprocess(raw, (1, 1, 1), load_plan(tiny_model / "plans.json"))
    cropped = raw[:, 1:4, 1:4, 1:4]
    assert prepared.bbox == (slice(1, 4),) * 3
    assert np.allclose(prepared.data, (cropped - cropped.mean()) / cropped.std(), rtol=0, atol=1e-6)
    assert prepared.data.dtype == np.float32


def test_overlapping_tiles_use_fp16_accumulation(tiny_model: Path) -> None:
    plan = load_plan(tiny_model / "plans.json")
    session = create_session(tiny_model / "model.onnx", plan, device="cpu")
    logits = sliding_logits(np.ones((1, 13, 14, 15), dtype=np.float32), plan, session)
    assert logits.shape == (2, 13, 14, 15)
    assert logits.dtype == np.float16
    assert np.all(logits[0] == 0)
    assert np.all(logits[1] == 1)


@pytest.mark.parametrize("shape", [(1, 1, 8, 8, 8), (2, 2, 8, 8, 8)])
def test_dynamic_output_cannot_broadcast_or_drop_batches(tiny_model: Path, shape: tuple) -> None:
    plan = load_plan(tiny_model / "plans.json")
    session = SimpleNamespace(run=lambda *_: [np.ones(shape, dtype=np.float32)])
    with pytest.raises(ValueError, match="prediction shape differs"):
        sliding_logits(np.ones((1, 8, 8, 8), dtype=np.float32), plan, session)


@pytest.mark.parametrize(
    "voxels", [np.zeros((8, 8, 8), dtype=np.float32), np.full((8, 8, 8), np.nan, dtype=np.float32)]
)
def test_invalid_intensities_fail_without_output(
    tiny_model: Path, tmp_path: Path, voxels: np.ndarray
) -> None:
    input_path = _write_ct(tmp_path, voxels)
    output_path = tmp_path / "mask.nrrd"
    with pytest.raises(ValueError, match="no nonzero crop|non-finite intensities"):
        predict_arg(input_path, tiny_model, output_path, ArgPolicy())
    assert not output_path.exists()


@pytest.mark.parametrize("dimension", [2, 4])
def test_non_3d_ct_is_rejected(tiny_model: Path, tmp_path: Path, dimension: int) -> None:
    input_path = tmp_path / "ct.nrrd"
    image = sitk.Image([4] * dimension, sitk.sitkFloat32)
    sitk.WriteImage(image, str(input_path))
    with pytest.raises(ValueError, match="scalar 3D CT"):
        predict_arg(input_path, tiny_model, tmp_path / "mask.nrrd", ArgPolicy())


def test_vector_ct_is_rejected_before_cast(tiny_model: Path, tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.ones((4, 4, 4, 2), dtype=np.float32), isVector=True)
    input_path = tmp_path / "vector.nrrd"
    sitk.WriteImage(image, str(input_path))
    with pytest.raises(ValueError, match="scalar 3D CT"):
        predict_arg(input_path, tiny_model, tmp_path / "mask.nrrd", ArgPolicy())


@pytest.mark.parametrize(
    "name,value",
    [
        ("preprocessing", "reference-global-zscore"),
        ("configuration", "2d"),
        ("plans_sha256", "wrong"),
    ],
)
def test_incompatible_graph_metadata_is_rejected(tiny_model: Path, name: str, value: str) -> None:
    _write_graph(tiny_model, **{name: value})
    with pytest.raises(ValueError, match="metadata differs|hashes disagree"):
        create_session(
            tiny_model / "model.onnx", load_plan(tiny_model / "plans.json"), device="cpu"
        )


def test_unspecified_precision_is_reported_without_inventing_fp16(
    tiny_model: Path, tmp_path: Path
) -> None:
    model = onnx.load(tiny_model / "model.onnx")
    properties = {item.key: item.value for item in model.metadata_props if item.key != "precision"}
    helper.set_model_props(model, properties)
    onnx.save(model, tiny_model / "model.onnx")
    metadata = predict_arg(
        _write_ct(tmp_path),
        tiny_model,
        tmp_path / "mask.nrrd",
        ArgPolicy(),
    )
    assert metadata["model_precision"] == "unspecified"


@pytest.mark.parametrize("precision", ["not-json", "16"])
def test_malformed_precision_metadata_is_rejected(tiny_model: Path, precision: str) -> None:
    model = onnx.load(tiny_model / "model.onnx")
    properties = {item.key: item.value for item in model.metadata_props}
    properties["precision"] = precision
    helper.set_model_props(model, properties)
    onnx.save(model, tiny_model / "model.onnx")
    with pytest.raises(ValueError, match="precision metadata must be a JSON string"):
        create_session(
            tiny_model / "model.onnx", load_plan(tiny_model / "plans.json"), device="cpu"
        )


def test_missing_graph_metadata_is_rejected(tiny_model: Path) -> None:
    model = onnx.load(tiny_model / "model.onnx")
    model.ClearField("metadata_props")
    onnx.save(model, tiny_model / "model.onnx")
    with pytest.raises(ValueError, match="hashes disagree"):
        create_session(
            tiny_model / "model.onnx", load_plan(tiny_model / "plans.json"), device="cpu"
        )


def test_graph_patch_shape_must_match_plans(tiny_model: Path) -> None:
    plans_path = tiny_model / "plans.json"
    plans = json.loads(plans_path.read_text())
    plans["configurations"]["3d_fullres"]["patch_size"] = [16, 8, 8]
    plans_path.write_text(json.dumps(plans))
    _write_graph(tiny_model)
    with pytest.raises(ValueError, match="shape differs"):
        create_session(tiny_model / "model.onnx", load_plan(plans_path), device="cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("spacing", [0, 1, 1]),
        ("patch_size", [8, 8]),
        (
            "resampling_fn_data_kwargs",
            {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
        ),
    ],
)
def test_unsupported_plans_are_rejected(tiny_model: Path, field: str, value: object) -> None:
    path = tiny_model / "plans.json"
    plans = json.loads(path.read_text())
    plans["configurations"]["3d_fullres"][field] = value
    path.write_text(json.dumps(plans))
    with pytest.raises(ValueError, match="ARG preprocessing contract|positive 3D"):
        load_plan(path)


def test_malformed_plans_are_rejected(tiny_model: Path) -> None:
    path = tiny_model / "plans.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Malformed ARG preprocessing plans"):
        load_plan(path)


@pytest.mark.parametrize(
    "dataset",
    [
        {"channel_names": {"0": "CT", "1": "CT"}, "labels": {"background": 0, "vessel": 1}},
        {"channel_names": {"0": "CT"}, "labels": {"background": 0, "vessel": 2}},
    ],
)
def test_incompatible_dataset_is_rejected(tiny_model: Path, tmp_path: Path, dataset: dict) -> None:
    (tiny_model / "dataset.json").write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="one input channel and binary labels"):
        predict_arg(
            _write_ct(tmp_path),
            tiny_model,
            tmp_path / "mask.nrrd",
            ArgPolicy(),
        )


@pytest.mark.parametrize(
    "model_file", ["../model.onnx", ".", "..", "sub/model.onnx", "checkpoint.pth", "model\\.onnx"]
)
def test_model_filename_cannot_escape_artifact(model_file: str) -> None:
    with pytest.raises(ValueError, match="ONNX filename inside"):
        ArgPolicy(model_file=model_file)


@pytest.mark.skipif(
    not os.environ.get("ARG_MODEL_PATH"),
    reason="Set ARG_MODEL_PATH for published ONNX artifact proof",
)
def test_published_artifact_loads_without_forward() -> None:
    model = Path(os.environ["ARG_MODEL_PATH"])
    plan = load_plan(model / "plans.json")
    session = create_session(model / "model.onnx", plan, device="cpu")
    assert session.get_inputs()[0].shape == [1, 1, *plan.patch]


@pytest.mark.parametrize(
    "mode",
    [
        "process",
        pytest.param(
            "container",
            marks=pytest.mark.skipif(
                not os.environ.get("MONAI_ARG_IMAGE"), reason="Local ARG image required"
            ),
        ),
    ],
)
def test_live_http_registry_and_onnx_worker_updates(tiny_model, tmp_path, live_service, mode):
    """Two ONNX releases take effect without restarting the actual Label server."""
    import shutil

    from mlflow import MlflowClient
    from requests_toolbelt.multipart.decoder import MultipartDecoder

    registry_uri = f"sqlite:///{tmp_path / 'registry.db'}"
    registry = MlflowClient(tracking_uri=registry_uri, registry_uri=registry_uri)
    registry.create_registered_model("arg_test")
    registry.create_model_version("arg_test", tiny_model.as_uri())
    second_model = tmp_path / "second-model"
    shutil.copytree(tiny_model, second_model)
    _write_graph(second_model, foreground=False)
    registry.create_model_version("arg_test", second_model.as_uri())
    registry.set_registered_model_alias("arg_test", "selected", "1")
    image = sitk.GetImageFromArray(np.arange(1, 513, dtype=np.float32).reshape(8, 8, 8))
    image.SetOrigin((12, -24, 48))
    input_path = tmp_path / "http-input.nii.gz"
    sitk.WriteImage(image, str(input_path))
    environment = {
        "MONAI_SERVICE_ENABLE_ARG": "true",
        "MONAI_SERVICE_ARG_WORKER__MODE": mode,
        "MONAI_SERVICE_ARG_MODEL_NAME": "arg_test",
        "MONAI_SERVICE_ARG_MODEL_ALIAS": "selected",
        "MONAI_SERVICE_ARG_DEVICE": "cpu",
        "MONAI_SERVICE_ARG_WORKER__TIMEOUT_SECONDS": "45",
        "MONAI_SERVICE_ARG_WORKER__ENVIRONMENT": json.dumps({"ORT_DISABLE_TELEMETRY": "1"}),
        "MONAI_SERVICE_MLFLOW_TRACKING_URI": registry_uri,
        "MONAI_SERVICE_MLFLOW_REGISTRY_URI": registry_uri,
    }
    if mode == "container":
        environment["MONAI_SERVICE_ARG_WORKER__IMAGE"] = os.environ["MONAI_ARG_IMAGE"]
    else:
        environment["MONAI_SERVICE_ARG_WORKER__COMMAND"] = json.dumps(
            [sys.executable, "-m", "monai_service.models.arg"]
        )
    with live_service(environment) as client:
        assert "arg_vessel" in client.get("/info/").json()["models"]
        for version, expected_label in (("1", 1), ("2", 0)):
            registry.set_registered_model_alias("arg_test", "selected", version)
            response = client.post(
                "/infer/arg_vessel", files={"file": ("ct.nii.gz", input_path.read_bytes())}
            )
            assert response.status_code == 200, response.text
            parts = {
                part.headers[b"Content-Disposition"].split(b'name="')[1].split(b'"')[0]: part
                for part in MultipartDecoder.from_response(response).parts
            }
            result_path = tmp_path / f"mask-{version}.nii.gz"
            result_path.write_bytes(parts[b"image"].content)
            result = sitk.ReadImage(str(result_path))
            assert np.all(sitk.GetArrayFromImage(result) == expected_label)
            assert result.GetOrigin() == pytest.approx(image.GetOrigin())
            metadata = json.loads(parts[b"params"].content)
            assert metadata["model_version"] == version
            assert metadata["model"] == "arg_test"
            assert metadata["runtime"] == ("docker" if mode == "container" else "process")
            assert metadata["inference_runtime"] == "onnxruntime"
            assert metadata["providers"] == ["CPUExecutionProvider"]
            assert metadata["model_file"] == "model.onnx"
            assert metadata["preprocessing"] == "native-plans"
            assert metadata.get("test_only") is not True
        assert len([p for p in (tmp_path / "service" / "models").iterdir() if p.is_dir()]) == 2
