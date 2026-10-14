"""Production Slang shader fixtures with a CPU sampling/feature oracle."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from model_o_inference.constants import PATCH_SIZE, RAY_LENGTH
from model_o_inference.lattice import FIXED_POINT_SCALE
from model_o_inference.shader_contract import config_record, output_specs


def interpolate(volume: np.ndarray, positions: np.ndarray) -> np.ndarray:
    lower = np.floor(positions).astype(np.int64)
    fraction = positions - lower
    result = np.zeros(len(positions), np.float64)
    for corner in product((0, 1), repeat=3):
        index = np.clip(lower + corner, 0, np.array(volume.shape) - 1)
        weight = np.prod(np.where(np.array(corner), fraction, 1 - fraction), axis=1)
        result += volume[tuple(index.T)] * weight
    return result


def reference(
    volume: np.ndarray, geometry: np.ndarray, center: np.ndarray, h0: float
) -> dict[str, np.ndarray]:
    """Independent float64 sampling with training-compatible per-patch truncation."""
    lattice = geometry[:, 1:].view(np.int32).astype(np.int64)
    origins = lattice[:, :3]
    steps = lattice[:, 3:]
    corner = center - PATCH_SIZE // 2
    outputs = {s.name: np.zeros(s.shape, s.dtype) for s in output_specs(len(geometry))}
    outputs["origin_zyx_norm"][:] = (origins / FIXED_POINT_SCALE - corner) / PATCH_SIZE
    outputs["direction_zyx_norm"][:] = steps / FIXED_POINT_SCALE / PATCH_SIZE
    for ray in range(len(geometry)):
        positions = origins[ray] + np.arange(1, RAY_LENGTH + 1)[:, None] * steps[ray]
        voxels = np.rint(positions.astype(np.float64) / FIXED_POINT_SCALE).astype(np.int64)
        local = voxels - corner
        valid = np.all((voxels >= 0) & (voxels < volume.shape), axis=1) & np.all(
            (local >= 0) & (local < PATCH_SIZE), axis=1
        )
        count = int(np.flatnonzero(~valid)[0]) if not valid.all() else RAY_LENGTH
        if count == 0:
            continue
        hu = interpolate(volume, positions[:count] / FIXED_POINT_SCALE)
        outputs["voxel_indices"][ray, :count] = np.ravel_multi_index(
            local[:count].T, (PATCH_SIZE,) * 3
        )
        outputs["ray_mask"][ray, :count] = True
        outputs["ray_hu"][ray, :count] = hu
        scale = max(abs(float(geometry[ray, 0:1].view(np.float32)[0])), 1.0)
        outputs["ray_features"][ray, :count] = np.column_stack(
            (hu / scale, np.arange(count) / max(count - 1, 1))
        )
    patch = outputs["ct_patch"][0]
    for z in range(PATCH_SIZE):
        for y in range(PATCH_SIZE):
            absolute = corner + [z, y, 0]
            if np.any(absolute[:2] < 0) or np.any(absolute[:2] >= volume.shape[:2]):
                continue
            lo, hi = max(0, -corner[2]), min(PATCH_SIZE, volume.shape[2] - corner[2])
            if hi > lo:
                patch[z, y, lo:hi] = volume[
                    absolute[0], absolute[1], corner[2] + lo : corner[2] + hi
                ]
    if h0 > 1e-6:
        patch /= h0
    return outputs


def make_tensor_bridge(output: Path, rays: int) -> None:
    """Make ORT read every production buffer, including packed bool/int64."""
    import onnx
    from onnx import TensorProto, helper

    inputs, nodes, outputs = [], [], []
    types = {"float32": TensorProto.FLOAT, "bool": TensorProto.BOOL, "int64": TensorProto.INT64}
    for spec in output_specs(rays):
        inputs.append(helper.make_tensor_value_info(spec.name, types[str(spec.dtype)], spec.shape))
        cast = spec.name + "_float"
        total = spec.name + "_total"
        nodes.append(helper.make_node("Cast", [spec.name], [cast], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("ReduceSum", [cast], [total], keepdims=0))
        outputs.append(helper.make_tensor_value_info(total, TensorProto.FLOAT, []))
    model = helper.make_model(
        helper.make_graph(nodes, "ModelO_buffer_contract", inputs, outputs),
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    (output / "tensor_bridge.onnx").write_bytes(model.SerializeToString())


def generate(output: Path) -> dict[str, Any]:
    """Include signed ties, large-coordinate Q16 fractions, tails and padding."""
    output.mkdir(parents=True, exist_ok=True)
    z, y, x = np.indices((4096, 13, 14))
    volume = (130 + 5 * z - 3 * y + 0.25 * x + 20 * np.sin(z / 3)).astype(np.float32)
    patterns = np.array(
        [
            [4.5, 5.5, 6.5, 0.25, 0, 0],
            [0, 0, 0, -0.25, 0, 0],
            [-0.5, 2, 2, 0, 0.25, 0],
            [-1.5, 3, 3, 0.25, 0, 0],
            [11.5, 12.5, 13.5, 0, 0, 0],
            [5, 6, 7, 0, 0, 0],
            [10, 11, 12, -0.25, -0.5, -0.75],
            [-0.5 + 1 / FIXED_POINT_SCALE, 2, 2, 0, 0, 0],
            [-0.5 - 1 / FIXED_POINT_SCALE, 2, 2, 0, 0, 0],
            [300.5 + 1 / FIXED_POINT_SCALE, 5, 6, 0, 0, 0],
            [3072.5 + 1 / FIXED_POINT_SCALE, 5, 6, 0, 0, 0],
            [3072.5 - 1 / FIXED_POINT_SCALE, 5, 6, 0, 0, 0],
            [3072.5, 5, 6, 0, 0, 0],
            [3073.5, 5, 6, 0, 0, 0],
        ]
    )
    lattice = np.rint(patterns[np.arange(65) % len(patterns)] * FIXED_POINT_SCALE).astype(np.int32)
    hu = np.tile(np.array([0, -200, 1, 120, -1, 300, 50], np.float32), 10)[:65]
    geometry = np.column_stack((hu.view(np.uint32), lattice.view(np.uint32)))
    geometry.tofile(output / "geometry.bin")
    make_tensor_bridge(output, len(geometry))
    manifest: dict[str, Any] = {
        "rays": len(geometry),
        "volume_shape": list(volume.shape),
        "geometry": "geometry.bin",
        "cases": [],
    }
    # A one-Q16-unit offset around a half voxel produces exactly +/-1 HU.
    # Converting the whole coordinate to float32 erases that offset above 256.
    precision = np.where(z % 2 == 0, 32768.0, -32768.0).astype(np.float32)
    for label, data in (
        ("varying", volume),
        ("constant", np.full_like(volume, -200)),
        ("q16-precision", precision),
    ):
        data.tofile(output / f"{label}.bin")
        for iteration, (center, h0) in enumerate(
            (
                (np.array([6, 6, 7]), 180.0),
                (np.array([35, 35, 35]), 0.0),
                (np.array([300, 6, 7]), 180.0),
                (np.array([3072, 6, 7]), 180.0),
            )
        ):
            name = f"{label}-{iteration}"
            depth, height, width = data.shape
            corner_z, corner_y, corner_x = center - PATCH_SIZE // 2
            config = config_record(
                (depth, height, width),
                len(geometry),
                h0,
                (int(corner_z), int(corner_y), int(corner_x)),
            )
            config.tofile(output / f"{name}-config.bin")
            specs = []
            expected = reference(data, geometry, center, h0)
            for spec in output_specs(len(geometry)):
                values = expected[spec.name]
                filename = f"{name}-{spec.name}.bin"
                values.tofile(output / filename)
                specs.append(
                    {
                        "name": spec.name,
                        "dtype": str(spec.dtype),
                        "shape": list(spec.shape),
                        "bytes": spec.nbytes,
                        "expected": filename,
                        "atol": 0.0003 if spec.name == "ray_features" else 0.0001,
                        "rtol": 0.00003,
                    }
                )
            manifest["cases"].append(
                {
                    "name": name,
                    "volume": f"{label}.bin",
                    "config": f"{name}-config.bin",
                    "center": center.tolist(),
                    "h0": h0,
                    "outputs": specs,
                }
            )
    logits = np.array(
        [-80, -1e-7, -np.finfo(np.float32).tiny, -0.0, 0.0, np.finfo(np.float32).tiny, 1e-7, 80],
        np.float32,
    )
    logits.tofile(output / "logits.bin")
    (logits > 0).tofile(output / "mask.bin")
    (output / "model_o.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_native(host: Any, directory: Path) -> dict[str, Any]:
    import onnxruntime as ort
    from dawn_ort import DeviceTensor, GpuSession
    from model_o_inference.dawn_inputs import DawnPatchInputs, PackedRayCase

    manifest = json.loads((directory / "model_o.json").read_text())
    geometry = np.fromfile(directory / manifest["geometry"], np.uint32).reshape(-1, 7)
    results = []
    with ExitStack() as resources:
        ct = host.buffer(int(np.prod(manifest["volume_shape"])) * 4)
        resources.callback(ct.close)
        first = manifest["cases"][0]
        volume = np.fromfile(directory / first["volume"], np.float32).reshape(
            manifest["volume_shape"]
        )
        case = PackedRayCase(volume, (1, 1, 1), geometry, np.array(first["center"]), first["h0"])
        builder = DawnPatchInputs(host, case, ct_buffer=ct, initialize=False)
        resources.callback(builder.close)
        assert builder.storage["volume_zyx"] is ct
        addresses = {name: tensor.owner.address for name, tensor in builder.gpu_inputs.items()}
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = GpuSession(
            host,
            (directory / "tensor_bridge.onnx").read_bytes(),
            options=options,
            provider_options={"enableInt64": "1"},
        )
        resources.callback(session.close)
        session.native_session.disable_fallback()
        totals = {}
        for spec in builder.specs:
            buffer = host.buffer(4)
            resources.callback(buffer.close)
            totals[spec.name + "_total"] = DeviceTensor(buffer, (), np.float32)
        session.bind(builder.gpu_inputs, outputs=totals)
        for spec in manifest["cases"]:
            host.write(ct, np.fromfile(directory / spec["volume"], np.float32))
            builder.write_geometry(geometry)
            builder.generate(np.array(spec["center"]), spec["h0"])
            assert addresses == {
                name: tensor.owner.address for name, tensor in builder.gpu_inputs.items()
            }
            session.submit()
            errors = {}
            for output in spec["outputs"]:
                dtype = np.dtype(output["dtype"])
                actual = host.read(builder.input_buffers[output["name"]], dtype)
                expected = np.fromfile(directory / output["expected"], dtype)
                total = float(host.read(totals[output["name"] + "_total"].owner)[0])
                np.testing.assert_allclose(
                    total,
                    actual.astype(np.float64).sum(),
                    rtol=2e-4,
                    atol=0.002,
                    err_msg="ORT did not read the production buffer " + output["name"],
                )
                assert np.isfinite(actual).all(), output["name"]
                if dtype.kind in "biu":
                    np.testing.assert_array_equal(
                        actual, expected, err_msg=spec["name"] + output["name"]
                    )
                else:
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=output["atol"],
                        rtol=output["rtol"],
                        err_msg=spec["name"] + output["name"],
                    )
                errors[output["name"]] = float(
                    np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
                )
            results.append(
                {
                    "name": spec["name"],
                    "status": "passed",
                    "max_absolute_error": errors,
                }
            )
        session.close()
        builder.close()
        assert not ct.closed, "Borrowed CT was released by the input builder"
        trained = run_trained_native(host, directory, manifest, ct) if "model" in manifest else None
        from model_o_inference.stitching import mask_from_logits

        values = np.fromfile(directory / "logits.bin", np.float32)
        logits = host.buffer(values.nbytes)
        resources.callback(logits.close)
        for nonfinite in (None, np.nan, np.inf, -np.inf, None):
            data = values.copy()
            if nonfinite is not None:
                data[-1] = nonfinite
            host.write(logits, data)
            actual = host.read(logits)
            if nonfinite is None:
                np.testing.assert_array_equal(
                    mask_from_logits(actual), np.fromfile(directory / "mask.bin", bool)
                )
            else:
                try:
                    mask_from_logits(actual)
                except RuntimeError as error:
                    assert "non-finite" in str(error)
                else:
                    raise AssertionError("Nonfinite logits were accepted")
        host.raise_errors()
    return {
        "status": "passed",
        "cases": results,
        "logit_threshold_and_nonfinite_rejection": "passed",
        "caller_buffers_reused": True,
        "borrowed_ct_retained": True,
        "ort_reads_all_seven_inputs": True,
        "trained_model": trained,
    }


def attach_model(directory: Path, path: Path) -> None:
    """Stage a private trained export and CPU references beneath ignored build/."""
    import os

    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    import onnxruntime as ort
    from model_o_inference.graph import prepare_model_o

    manifest_path = directory / "model_o.json"
    manifest = json.loads(manifest_path.read_text())
    model = prepare_model_o(str(path), manifest["rays"])
    (directory / "selected.onnx").write_bytes(model)
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    # End-to-end provider arithmetic differs more than the sampled inputs.
    # Keep this tolerance separate from the strict shader checks and require
    # exact binary masks, including at the decision boundary.
    manifest["model"] = {
        "file": "selected.onnx",
        "sha256": hashlib.sha256(model).hexdigest(),
        "atol": 0.0005,
        "rtol": 0.003,
    }
    for case in manifest["cases"]:
        feeds = {
            output["name"]: np.fromfile(
                directory / output["expected"], np.dtype(output["dtype"])
            ).reshape(output["shape"])
            for output in case["outputs"]
        }
        output = session.run(["mask_logits"], feeds)[0]
        assert isinstance(output, np.ndarray), "ModelO must return a dense logit tensor"
        logits = output.reshape(-1)
        assert np.isfinite(logits).all(), "CPU trained model returned nonfinite logits"
        name = case["name"] + "-model-logits.bin"
        logits.tofile(directory / name)
        case["model_logits"] = name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def run_trained_native(
    host: Any, directory: Path, manifest: dict[str, Any], ct: Any
) -> dict[str, Any]:
    from model_o_inference.dawn_inputs import PackedRayCase
    from model_o_inference.runtime import _gpu_pipeline

    model_path = directory / manifest["model"]["file"]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest["model"]["sha256"]
    geometry = np.fromfile(directory / manifest["geometry"], np.uint32).reshape(-1, 7)
    cases = []
    with ExitStack() as resources:
        first = manifest["cases"][0]
        volume = np.fromfile(directory / first["volume"], np.float32).reshape(
            manifest["volume_shape"]
        )
        pipeline = _gpu_pipeline(
            resources,
            host,
            model_path,
            PackedRayCase(volume, (1, 1, 1), geometry, np.array(first["center"]), first["h0"]),
            ct,
        )
        pipeline.session.native_session.disable_fallback()
        for spec in manifest["cases"]:
            volume = np.fromfile(directory / spec["volume"], np.float32).reshape(
                manifest["volume_shape"]
            )
            host.write(ct, volume)
            case = PackedRayCase(volume, (1, 1, 1), geometry, np.array(spec["center"]), spec["h0"])
            with ExitStack() as pending:
                mask = pipeline.collect(pipeline.submit(case, pending))
            expected = np.fromfile(directory / spec["model_logits"], np.float32)
            actual = host.read(pipeline.logits)
            np.testing.assert_allclose(
                actual, expected, atol=manifest["model"]["atol"], rtol=manifest["model"]["rtol"]
            )
            np.testing.assert_array_equal(mask.reshape(-1), expected > 0)
            cases.append(
                {
                    "name": spec["name"],
                    "max_absolute_error": float(np.abs(actual - expected).max()),
                    "mask_differences": 0,
                    "foreground_voxels": int((expected > 0).sum()),
                }
            )
    return {
        "status": "passed",
        "model_sha256": manifest["model"]["sha256"],
        "cases": cases,
        "cpu_fallback_disabled": True,
        "production_binary_pipeline": True,
        "logit_tolerance": {key: manifest["model"][key] for key in ("atol", "rtol")},
    }
