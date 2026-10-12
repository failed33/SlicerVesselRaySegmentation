"""Production Slang shader fixtures with a CPU sampling/feature oracle."""

from __future__ import annotations

import json
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from model_o_inference.constants import PATCH_SIZE, RAY_LENGTH
from model_o_inference.lattice import FIXED_POINT_SCALE, round_fixed_to_voxel
from model_o_inference.shader_contract import config_record, output_specs
from model_o_inference.shaders import load_shader


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
    """Independent float64 interpolation/statistics, preserving the deployed policy."""
    lattice = geometry[:, 1:].view(np.int32).astype(np.int64)
    origins = lattice[:, :3]
    steps = lattice[:, 3:]
    corner = center - PATCH_SIZE // 2
    outputs = {s.name: np.zeros(s.shape, s.dtype) for s in output_specs(len(geometry))}
    outputs["origin_zyx_norm"][:] = (origins / FIXED_POINT_SCALE - corner) / PATCH_SIZE
    outputs["direction_zyx_norm"][:] = steps / FIXED_POINT_SCALE / PATCH_SIZE
    padding = -corner
    padding_index = (
        np.ravel_multi_index(padding, (PATCH_SIZE,) * 3)
        if np.all((padding >= 0) & (padding < PATCH_SIZE))
        else 0
    )
    outputs["voxel_indices"][:] = padding_index
    outputs["feature_check_mask"] = np.ones(outputs["ray_features"].shape, dtype=bool)
    for ray in range(len(geometry)):
        positions = origins[ray] + np.arange(1, RAY_LENGTH + 1)[:, None] * steps[ray]
        voxels = round_fixed_to_voxel(positions)
        valid = np.all((voxels >= 0) & (voxels < volume.shape), axis=1)
        count = int(np.flatnonzero(~valid)[0]) if not valid.all() else RAY_LENGTH
        if count == 0:
            continue
        hu = interpolate(volume, positions[:count] / FIXED_POINT_SCALE)
        local = voxels[:count] - corner
        scatter = np.all((local >= 0) & (local < PATCH_SIZE), axis=1)
        indices = np.zeros(count, dtype=np.int64)
        indices[scatter] = np.ravel_multi_index(local[scatter].T, (PATCH_SIZE,) * 3)
        outputs["voxel_indices"][ray, :count] = indices
        outputs["ray_mask"][ray, :count] = scatter
        outputs["ray_hu"][ray, :count] = hu
        scale = max(abs(float(geometry[ray, 0:1].view(np.float32)[0])), 1.0)
        prefix = np.cumsum(hu)
        counts = np.arange(1, count + 1)
        mu_left = prefix / counts
        mu_right = (hu.sum() - prefix) / np.maximum(count - counts, 1)
        variance = np.var(hu)
        if variance <= 1e-12:
            outputs["feature_check_mask"][ray, :count, 4] = False
        eta = (
            np.minimum(
                counts * (count - counts) * (mu_left - mu_right) ** 2 / (count * variance), 1
            )
            if variance > 1e-12
            else np.zeros(count)
        )
        outputs["ray_features"][ray, :count] = np.column_stack(
            (
                hu / scale,
                np.r_[0, np.diff(hu)] / scale,
                mu_left / scale,
                (mu_left - mu_right) / scale,
                eta,
                (np.maximum.accumulate(hu) - hu) / scale,
                np.arange(count) / max(count - 1, 1),
            )
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
    """Include workgroup tails, zero-valid rays, border ties, and CT padding."""
    output.mkdir(parents=True, exist_ok=True)
    z, y, x = np.indices((12, 13, 14))
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
    for label, data in (("varying", volume), ("constant", np.full_like(volume, -200))):
        data.tofile(output / f"{label}.bin")
        for iteration, (center, h0) in enumerate(
            ((np.array([6, 6, 7]), 180.0), (np.array([35, 35, 35]), 0.0))
        ):
            name = f"{label}-{iteration}"
            config = config_record(data.shape, len(geometry), h0, tuple(center - PATCH_SIZE // 2))
            config.tofile(output / f"{name}-config.bin")
            specs = []
            expected = reference(data, geometry, center, h0)
            expected["feature_check_mask"].tofile(output / f"{name}-feature-check-mask.bin")
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
                        **(
                            {"compare_mask": f"{name}-feature-check-mask.bin"}
                            if spec.name == "ray_features"
                            else {}
                        ),
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
    logits = np.linspace(-80, 80, PATCH_SIZE**3, dtype=np.float32)
    logits.tofile(output / "logits.bin")
    probabilities = (1 / (1 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    probabilities.tofile(output / "probabilities.bin")
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
                check_mask = (
                    np.fromfile(directory / output["compare_mask"], bool)
                    if "compare_mask" in output
                    else np.ones(actual.shape, bool)
                )
                assert np.isfinite(actual).all(), output["name"]
                if "compare_mask" in output:
                    assert np.all((actual[~check_mask] >= 0) & (actual[~check_mask] <= 1))
                if dtype.kind in "biu":
                    np.testing.assert_array_equal(
                        actual, expected, err_msg=spec["name"] + output["name"]
                    )
                else:
                    np.testing.assert_allclose(
                        actual[check_mask],
                        expected[check_mask],
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
                    "known_flat_profile_eta_limit": "CPU parity excludes eta on flat profiles; values remain finite in [0,1].",
                }
            )
        session.close()
        builder.close()
        assert not ct.closed, "Borrowed CT was released by the input builder"
        shader = load_shader("sigmoid")
        logits = host.buffer(PATCH_SIZE**3 * 4)
        probability = host.buffer(logits.size)
        invalid = host.buffer(4)
        for buffer in (logits, probability, invalid):
            resources.callback(buffer.close)
        consumer = host.kernel(
            shader.code,
            shader.bindings({"logits": logits, "probability": probability, "invalid": invalid}),
            entry_point=shader.entry_point,
        )
        resources.callback(consumer.close)
        values = np.fromfile(directory / "logits.bin", np.float32)
        for nonfinite in (None, np.nan, np.inf, -np.inf, None):
            data = values.copy()
            if nonfinite is not None:
                data[-1] = nonfinite
            host.write(logits, data)
            host.write(invalid, np.zeros(1, np.uint32))
            consumer.dispatch(shader.workgroups(PATCH_SIZE**3))
            actual = host.read(probability)
            flag = int(host.read(invalid, np.uint32)[0])
            assert flag == int(nonfinite is not None)
            if nonfinite is None:
                np.testing.assert_allclose(
                    actual,
                    np.fromfile(directory / "probabilities.bin", np.float32),
                    atol=1e-6,
                    rtol=1e-6,
                )
        host.raise_errors()
    return {
        "status": "passed",
        "cases": results,
        "sigmoid_nonfinite_and_reset": "passed",
        "caller_buffers_reused": True,
        "borrowed_ct_retained": True,
        "ort_reads_all_seven_inputs": True,
    }
