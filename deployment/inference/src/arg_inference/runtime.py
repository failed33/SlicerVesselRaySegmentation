"""Shared ONNX serving for the inspected scalar binary ARG native-plans model.

The image pipeline preserves nnU-Net 2.8.1 resampling and FP16 accumulation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
import SimpleITK as sitk
from numpy.typing import NDArray
from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates
from skimage.transform import resize

from arg_inference.plans import Plan, load_plan
from arg_inference.providers import Device, provider_node_counts
from arg_inference.providers import create_session as create_provider_session

Array = NDArray[np.floating]


@dataclass(frozen=True)
class Prepared:
    data: Array
    bbox: tuple[slice, ...]
    shape_before: tuple[int, ...]
    shape_cropped: tuple[int, ...]
    original_spacing: tuple[float, ...]


def separate_axis(current: tuple[float, ...], target: tuple[float, ...]) -> int | None:
    for spacing in [current, target]:
        if max(spacing) / min(spacing) > 3:
            axes = np.flatnonzero(np.asarray(spacing) == max(spacing))
            return int(axes[0]) if len(axes) == 1 else None
    return None


def resample(
    data: Array,
    shape: tuple[int, ...],
    current: tuple[float, ...],
    target: tuple[float, ...],
) -> Array:
    if data.shape[1:] == shape:
        return data
    result = np.zeros((data.shape[0], *shape), dtype=data.dtype)
    old_shape = data.shape[1:]
    axis = separate_axis(current, target)
    for channel, source in enumerate(data.astype(float, copy=False)):
        if axis is None:
            result[channel] = resize(source, shape, order=1, mode="edge", anti_aliasing=False)
            continue
        intermediate_shape = list(shape)
        intermediate_shape[axis] = old_shape[axis]
        intermediate = np.zeros(intermediate_shape, dtype=float)
        plane_shape = tuple(size for index, size in enumerate(shape) if index != axis)
        for index in range(old_shape[axis]):
            plane: list[slice | int] = [slice(None)] * 3
            plane[axis] = index
            intermediate[tuple(plane)] = resize(
                source[tuple(plane)],
                plane_shape,
                order=1,
                mode="edge",
                anti_aliasing=False,
            )
        if old_shape[axis] == shape[axis]:
            result[channel] = intermediate
        else:
            coordinates = np.mgrid[tuple(slice(size) for size in shape)].astype(float)
            for dim in range(3):
                coordinates[dim] = (
                    intermediate.shape[dim] / shape[dim] * (coordinates[dim] + 0.5) - 0.5
                )
            result[channel] = map_coordinates(intermediate, coordinates, order=0, mode="nearest")
    return result


def preprocess(data: Array, spacing: tuple[float, ...], plan: Plan) -> Prepared:
    data = data.astype(np.float32).transpose((0, *(index + 1 for index in plan.transpose)))
    spacing = tuple(spacing[index] for index in plan.transpose)
    before = data.shape[1:]
    nonzero = cast(NDArray[np.bool_], binary_fill_holes(data[0] != 0))
    indices = np.where(nonzero)
    if not indices[0].size:
        raise ValueError("ARG input has no nonzero crop")
    bbox = tuple(slice(int(index.min()), int(index.max()) + 1) for index in indices)
    cropped = data[(slice(None), *bbox)].copy()
    cropped_shape = cropped.shape[1:]
    # Native float32 reductions, cropped data, all voxels including internal zero.
    mean, std = cropped[0].mean(), cropped[0].std()
    cropped[0] -= mean
    cropped[0] /= max(std, 1e-8)
    new_shape = tuple(
        round(old / new * size) for old, new, size in zip(spacing, plan.spacing, cropped_shape)
    )
    return Prepared(
        resample(cropped, new_shape, spacing, plan.spacing),
        bbox,
        before,
        cropped_shape,
        spacing,
    )


def gaussian(patch: tuple[int, ...]) -> Array:
    impulse = np.zeros(patch)
    impulse[tuple(size // 2 for size in patch)] = 1
    weight = gaussian_filter(impulse, [size / 8 for size in patch], mode="constant", cval=0)
    weight /= weight.max() / 10
    weight = weight.astype(np.float16)
    weight[weight == 0] = weight[weight != 0].min()
    return weight


def steps(size: int, patch: int) -> list[int]:
    count = int(np.ceil((size - patch) / (patch * 0.5))) + 1
    interval = (size - patch) / (count - 1) if count > 1 else 0
    return [int(np.round(interval * index)) for index in range(count)]


def sliding_logits(
    data: Array,
    plan: Plan,
    session: ort.InferenceSession,
    precision: Literal["native", "fp32"] = "native",
) -> Array:
    padding = [max(0, patch - size) for size, patch in zip(data.shape[1:], plan.patch)]
    offsets = [value // 2 for value in padding]
    padded = np.pad(data, [(0, 0), *((value // 2, value - value // 2) for value in padding)])
    accumulation_dtype = np.float16 if precision == "native" else np.float32
    logits = np.zeros((2, *padded.shape[1:]), dtype=accumulation_dtype)
    counts = np.zeros(padded.shape[1:], dtype=accumulation_dtype)
    weight = gaussian(plan.patch)
    positions = [steps(size, patch) for size, patch in zip(padded.shape[1:], plan.patch)]
    for start in itertools.product(*positions):
        slices = tuple(slice(index, index + size) for index, size in zip(start, plan.patch))
        patch = np.ascontiguousarray(padded[(slice(None), *slices)][None])
        output = cast(Array, session.run(["logits"], {"image": patch})[0])
        if output.shape != (1, 2, *plan.patch):
            raise ValueError("ARG ONNX prediction shape differs from the planned binary patch")
        prediction = output[0]
        prediction *= weight
        # Explicitly match torch.half += float32 prediction and half count arithmetic.
        logits[(slice(None), *slices)] = (
            logits[(slice(None), *slices)].astype(np.float32) + prediction
        ).astype(accumulation_dtype)
        counts[slices] += weight
    np.divide(logits, counts, out=logits)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite sliding-window logits")
    return logits[
        (
            slice(None),
            *(slice(offset, offset + size) for offset, size in zip(offsets, data.shape[1:])),
        )
    ]


def restore(logits: Array, prepared: Prepared, plan: Plan) -> NDArray[np.uint8]:
    restored_logits = resample(
        logits, prepared.shape_cropped, plan.spacing, prepared.original_spacing
    )
    result = np.zeros(prepared.shape_before, dtype=np.uint8)
    result[prepared.bbox] = restored_logits.argmax(axis=0).astype(np.uint8)
    return result.transpose(plan.inverse)


def _validate_session(session: ort.InferenceSession, plan: Plan) -> None:
    metadata = session.get_modelmeta().custom_metadata_map
    expected = metadata.get("plans_sha256")
    if expected != json.dumps(plan.sha256):
        raise ValueError("ONNX model and preprocessing plans hashes disagree")
    for name, expected in (
        ("preprocessing", "native-plans"),
        ("configuration", "3d_fullres"),
    ):
        if metadata.get(name) != json.dumps(expected):
            raise ValueError(f"ONNX model {name} metadata differs from the ARG contract")
    for nodes, name, channels in (
        (session.get_inputs(), "image", 1),
        (session.get_outputs(), "logits", 2),
    ):
        if len(nodes) != 1 or nodes[0].name != name or nodes[0].type != "tensor(float)":
            raise ValueError(f"ARG ONNX requires one FP32 {name} tensor")
        shape = nodes[0].shape
        if len(shape) != 5 or any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(shape, (1, channels, *plan.patch))
        ):
            raise ValueError(f"ARG ONNX {name} shape differs from the planned patch")


def _model_precision(session: ort.InferenceSession) -> str:
    precision = session.get_modelmeta().custom_metadata_map.get("precision")
    if precision is None:
        return "unspecified"
    try:
        value = json.loads(precision)
    except json.JSONDecodeError as error:
        raise ValueError("ONNX model precision metadata must be a JSON string") from error
    if not isinstance(value, str):
        raise ValueError("ONNX model precision metadata must be a JSON string")
    return value


def create_session(
    model: Path, plan: Plan, threads: int = 4, *, device: Device = "auto"
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = create_provider_session(str(model), device=device, options=options)
    _validate_session(session, plan)
    _model_precision(session)
    return session


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def predict_file(
    input_path: Path, output: Path, model: Path, plan: Plan, *, device: Device = "auto"
) -> dict[str, object]:
    reference = sitk.ReadImage(str(input_path))
    if reference.GetDimension() != 3 or reference.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("ARG input must be a scalar 3D CT")
    reference = sitk.Cast(reference, sitk.sitkFloat32)
    oriented = sitk.DICOMOrient(reference, "LPS")
    data = sitk.GetArrayFromImage(oriented)[None]
    if not np.isfinite(data).all():
        raise ValueError("ARG input contains non-finite intensities")
    session = create_session(model, plan, device=device)
    prepared = preprocess(data, tuple(oriented.GetSpacing()[::-1]), plan)
    mask = sitk.GetImageFromArray(
        restore(sliding_logits(prepared.data, plan, session), prepared, plan)
    )
    mask.CopyInformation(oriented)
    original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        reference.GetDirection()
    )
    mask = sitk.DICOMOrient(mask, original_orientation)
    if mask.GetSize() != reference.GetSize() or any(
        not np.allclose(actual, expected, rtol=0, atol=1e-6)
        for actual, expected in [
            (mask.GetSpacing(), reference.GetSpacing()),
            (mask.GetOrigin(), reference.GetOrigin()),
            (mask.GetDirection(), reference.GetDirection()),
        ]
    ):
        raise ValueError("ARG output physical grid differs from input")
    mask.CopyInformation(reference)
    sitk.WriteImage(mask, str(output), useCompression=True)
    return {
        "shape": list(reference.GetSize()),
        "preprocessed_shape": list(prepared.data.shape),
        "providers": session.get_providers(),
        "provider_node_counts": provider_node_counts(session),
        "accumulation": "native-fp16",
        "inference_runtime": "onnxruntime",
        "available_devices": [
            {
                "provider": ep.ep_name,
                "type": ep.device.type.name,
                "id": ep.device.device_id,
            }
            for ep in ort.get_ep_devices()
        ],
        "device_preference": device,
        "model_precision": _model_precision(session),
        "model_sha256": _sha256(model),
        "plans_sha256": plan.sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "npu", "gpu", "cpu"), default="auto")
    args = parser.parse_args()
    start = time.perf_counter()
    result = predict_file(
        args.input, args.output, args.model, load_plan(args.plans), device=args.device
    )
    result["seconds"] = time.perf_counter() - start
    print(json.dumps(result))


if __name__ == "__main__":
    main()
