"""Native shared-device ModelO inference from CT and ordered ZYX polylines."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import ExitStack, closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .preview import PatchProgress, PreviewObserver
from .schemas import ModelOParameters

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    import numpy as np
    import SimpleITK as sitk
    from dawn_ort import DawnHost, GpuSession
    from dawn_ort.host import Buffer
    from numpy.typing import NDArray

    from .dawn_inputs import DawnPatchInputs, PackedRayCase
    from .planning import PatchPlan
    from .preparation import RayContext
    from .stitching import MaskMerger


@dataclass(frozen=True)
class _Volume:
    reference: sitk.Image
    array: NDArray[np.float32]
    spacing: tuple[float, float, float]


@dataclass(frozen=True)
class _PatchExecution:
    image: _Volume
    model_path: Path
    observer: PreviewObserver | None = None


@dataclass
class _PatchRun:
    execution: _PatchExecution
    context: RayContext
    merger: MaskMerger
    ct_buffer: Buffer
    total: int
    queue_depth: Literal[1, 2, 4]
    timings: dict[str, float]
    started: float
    completed: int = 0
    last_log: float = 0.0


def _mask_image(mask_array: NDArray[np.bool_], reference: sitk.Image) -> sitk.Image:
    import numpy as np
    import SimpleITK as sitk

    mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask.CopyInformation(reference)
    return mask


def _observe_patch(execution: _PatchExecution, merger: MaskMerger, progress: PatchProgress) -> None:
    if execution.observer is not None:
        execution.observer.on_progress(
            progress, lambda: _mask_image(merger.finalize(), execution.image.reference)
        )


def _read_ct(input_path: Path) -> _Volume:
    import numpy as np
    import SimpleITK as sitk

    reference = sitk.ReadImage(str(input_path))
    if reference.GetDimension() != 3 or reference.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("ModelO input must be a scalar 3D CT")
    reference = sitk.Cast(reference, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(reference)
    if not np.isfinite(array).all():
        raise ValueError("ModelO input contains non-finite intensities")
    spacing_xyz = reference.GetSpacing()
    spacing = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return _Volume(reference, array, spacing)


def _ray_case(image: _Volume, context: RayContext, plan: PatchPlan, capacity: int) -> PackedRayCase:
    """Gather Q16 geometry directly, padding with the existing invalid-ray sentinel."""
    import numpy as np

    from .dawn_inputs import PackedRayCase, pack_ray_geometry
    from .lattice import FIXED_POINT_SCALE

    count = len(plan.ray_indices)
    lattice = np.zeros((capacity, 6), dtype=np.int32)
    lattice[:, :3] = -2 * FIXED_POINT_SCALE
    origin_hu = np.ones(capacity, dtype=np.float32)
    lattice[:count] = context.ray_lattice_fp[plan.ray_indices]
    origin_hu[:count] = context.ray_origin_hu[plan.ray_indices]
    return PackedRayCase(
        image.array, image.spacing, pack_ray_geometry(lattice, origin_hu), plan.center_zyx, plan.h0
    )


@dataclass(frozen=True)
class _GpuPatchInference:
    host: DawnHost
    builder: DawnPatchInputs
    session: GpuSession
    logits: Buffer

    def submit(self, case: PackedRayCase, resources: ExitStack) -> Buffer:
        self.builder.write_geometry(case.packed_geometry)
        self.builder.generate(case.center, case.h0)
        self.session.submit()
        # Preserve this prediction before the next patch reuses the output.
        readback = self.host.stage_readback(self.logits)
        resources.callback(readback.close)
        return readback

    def collect(self, readback: Buffer) -> NDArray[np.bool_]:
        from .constants import PATCH_SIZE
        from .stitching import mask_from_logits

        logits = self.host.read_staging(readback).reshape((PATCH_SIZE,) * 3)
        return mask_from_logits(logits)


def _gpu_pipeline(
    resources: ExitStack, host: DawnHost, model_path: Path, case: PackedRayCase, ct_buffer: Buffer
) -> _GpuPatchInference:
    import numpy as np
    from dawn_ort import DeviceTensor

    from .constants import OUTPUT_NAME, PATCH_SIZE
    from .dawn_inputs import DawnPatchInputs
    from .graph import create_model_o_session

    logits = host.buffer(PATCH_SIZE**3 * np.dtype(np.float32).itemsize)
    resources.callback(logits.close)
    builder = DawnPatchInputs(host, case, ct_buffer=ct_buffer, initialize=False)
    resources.callback(builder.close)
    session = create_model_o_session(host, str(model_path), case.n_rays)
    resources.callback(session.close)
    session.bind(
        builder.gpu_inputs,
        outputs={
            OUTPUT_NAME: DeviceTensor(
                logits, (1, 1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), np.float32
            )
        },
    )
    return _GpuPatchInference(host, builder, session, logits)


def _upload_ct(resources: ExitStack, host: DawnHost, image: _Volume) -> Buffer:
    ct_buffer = host.buffer(image.array.nbytes)
    resources.callback(ct_buffer.close)
    host.write(ct_buffer, image.array)
    return ct_buffer


def _predict_windows(
    pipeline: _GpuPatchInference,
    cases: Iterator[tuple[PatchPlan, PackedRayCase]],
    queue_depth: Literal[1, 2, 4],
    timings: dict[str, float],
) -> Generator[tuple[PatchPlan, NDArray[np.bool_]], None, None]:
    """Copy all window outputs before mapping; the caller must close this iterator."""
    while True:
        submitted = time.perf_counter()
        window = tuple(islice(cases, queue_depth))
        if not window:
            return
        with ExitStack() as resources:
            pending = [(plan, pipeline.submit(case, resources)) for plan, case in window]
            timings["patch_submission"] += time.perf_counter() - submitted
            for plan, readback in pending:
                collected = time.perf_counter()
                patch_mask = pipeline.collect(readback)
                timings["patch_collection"] += time.perf_counter() - collected
                yield plan, patch_mask


def _report_patch(
    execution: _PatchExecution, merger: MaskMerger, progress: PatchProgress, last_log: float
) -> float:
    _observe_patch(execution, merger, progress)
    if (
        progress.completed == 1
        or progress.completed == progress.total
        or progress.elapsed_seconds - last_log >= 10.0
    ):
        logger.info(
            "ModelO patches %s/%s completed in %.2f s",
            progress.completed,
            progress.total,
            progress.elapsed_seconds,
        )
        return progress.elapsed_seconds
    return last_log


def _capacity_groups(plans: list[PatchPlan]) -> dict[int, list[PatchPlan]]:
    """Reduce invalid-ray padding with at most three sequential session sizes."""
    maximum = max(len(plan.ray_indices) for plan in plans)
    upper = 1 << (maximum - 1).bit_length()
    # Coarse bounds limit model reloads; cap the last group at its actual maximum.
    limits = (max(1, upper // 4), max(1, upper // 2), maximum)
    groups: dict[int, list[PatchPlan]] = {}
    for plan in plans:
        capacity = next(limit for limit in limits if len(plan.ray_indices) <= limit)
        groups.setdefault(capacity, []).append(plan)
    return dict(sorted(groups.items()))


def _infer_patches(
    host: DawnHost,
    context: RayContext,
    plans: list[PatchPlan],
    execution: _PatchExecution,
    *,
    queue_depth: Literal[1, 2, 4] = 1,
) -> tuple[NDArray[np.bool_], int, str, dict[str, float]]:
    """Measure CPU wall time; collection includes queued GPU completion and mapping."""
    from .stitching import MaskMerger

    if type(queue_depth) is not int or queue_depth not in (1, 2, 4):
        raise ValueError("ModelO queue depth must be 1, 2 or 4")
    image = execution.image
    groups = _capacity_groups(plans)
    capacity = max(groups)
    logger.info(
        "ModelO initializing %s GPU capacity groups: %s patches, maximum %s rays, queue depth %s",
        len(groups),
        len(plans),
        capacity,
        queue_depth,
    )
    merger = MaskMerger((image.array.shape[0], image.array.shape[1], image.array.shape[2]))
    started = time.perf_counter()
    provider = ""
    timings = dict.fromkeys(
        ("ct_upload", "group_setup", "patch_submission", "patch_collection", "patch_merge"), 0.0
    )
    with ExitStack() as request_resources:
        ct_buffer = _upload_ct(request_resources, host, image)
        timings["ct_upload"] = time.perf_counter() - started
        run = _PatchRun(
            execution, context, merger, ct_buffer, len(plans), queue_depth, timings, started
        )
        for group_index, (group_capacity, group_plans) in enumerate(groups.items(), start=1):
            logger.info(
                "ModelO GPU group %s/%s: %s patches, %s-ray capacity",
                group_index,
                len(groups),
                len(group_plans),
                group_capacity,
            )
            provider = _infer_group(host, group_capacity, group_plans, run)
    return merger.finalize(), capacity, provider, timings


def _infer_group(host: DawnHost, capacity: int, plans: list[PatchPlan], run: _PatchRun) -> str:
    started = time.perf_counter()
    image = run.execution.image
    first_case = _ray_case(image, run.context, plans[0], capacity)
    with ExitStack() as resources:
        pipeline = _gpu_pipeline(
            resources, host, run.execution.model_path, first_case, run.ct_buffer
        )
        provider = _gpu_provider(pipeline)
        setup_seconds = time.perf_counter() - started
        run.timings["group_setup"] += setup_seconds
        logger.info("ModelO GPU group ready in %.2f s", setup_seconds)
        cases = (
            (plan, first_case if index == 0 else _ray_case(image, run.context, plan, capacity))
            for index, plan in enumerate(plans)
        )
        predictions = resources.enter_context(
            closing(_predict_windows(pipeline, cases, run.queue_depth, run.timings))
        )
        for plan, patch_mask in predictions:
            _merge_prediction(run, plan, patch_mask)
    return provider


def _merge_prediction(run: _PatchRun, plan: PatchPlan, patch_mask: NDArray[np.bool_]) -> None:
    started = time.perf_counter()
    center = plan.center_zyx
    run.merger.aggregate(patch_mask, (int(center[0]), int(center[1]), int(center[2])))
    run.timings["patch_merge"] += time.perf_counter() - started
    run.completed += 1
    progress = PatchProgress(run.completed, run.total, time.perf_counter() - run.started)
    run.last_log = _report_patch(run.execution, run.merger, progress, run.last_log)


def _gpu_provider(pipeline: _GpuPatchInference) -> str:
    providers = pipeline.session.native_session.get_providers()
    for name in providers:
        if name.lower() == "webgpuexecutionprovider":
            return name
    raise RuntimeError("ModelO session did not select the Dawn WebGPU provider")


def _prepare_patches(
    image: _Volume, parameters: ModelOParameters
) -> tuple[RayContext, list[PatchPlan], dict[str, float]]:
    from .constants import model_o_geometry
    from .planning import build_patch_plans
    from .preparation import prepare_ray_context

    started = time.perf_counter()
    logger.info("ModelO preparing rays for %s polylines at 1 mm", len(parameters.polylines_zyx))
    context = prepare_ray_context(
        polylines_zyx=parameters.polylines_zyx,
        volume=image.array,
        geometry=model_o_geometry(),
        spacing_zyx=image.spacing,
    )
    preparation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    plans = build_patch_plans(
        context, overlap=parameters.overlap, polylines_zyx=parameters.polylines_zyx
    )
    planning_seconds = time.perf_counter() - started
    logger.info(
        "ModelO prepared %s origins (%s rays) in %.2f s; planned %s patches in %.2f s, overlap=%s",
        len(context.sample_positions),
        context.n_rays,
        preparation_seconds,
        len(plans),
        planning_seconds,
        parameters.overlap,
    )
    return (
        context,
        plans,
        {"ray_preparation": preparation_seconds, "patch_planning": planning_seconds},
    )


def _patch_metadata(
    context: RayContext, plans: list[PatchPlan], capacity: int, queue_depth: Literal[1, 2, 4]
) -> dict[str, object]:
    from .constants import PATCH_SIZE, SAMPLE_SPACING_MM

    groups = _capacity_groups(plans)
    return {
        "patches": len(plans),
        "patch_size": PATCH_SIZE,
        "sample_spacing_mm": SAMPLE_SPACING_MM,
        "ray_samples": len(context.sample_positions),
        "rays": context.n_rays,
        "encoded_rays": capacity,
        "queue_depth": queue_depth,
        "capacity_groups": [
            {"capacity": size, "patches": len(group)} for size, group in groups.items()
        ],
        "gathered_ray_slots": sum(len(plan.ray_indices) for plan in plans),
        "padded_ray_slots": sum(size * len(group_plans) for size, group_plans in groups.items()),
    }


def predict_model_o(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    parameters: ModelOParameters,
    assets_path: Path,
    *,
    observer: PreviewObserver | None = None,
    queue_depth: Literal[1, 2, 4] = 1,
) -> dict[str, object]:
    """Run patch inference, optionally observing lazy cumulative mask previews."""
    import SimpleITK as sitk

    from .constants import LOGIT_THRESHOLD

    started = time.perf_counter()
    logger.info("ModelO reading submitted CT")
    image = _read_ct(input_path)
    read_seconds = time.perf_counter() - started
    context, plans, timings = _prepare_patches(image, parameters)
    inference_started = time.perf_counter()
    import onnxruntime as ort
    from dawn_ort import DawnHost
    from dawn_ort.platforms import platform_config

    provider = assets_path / platform_config(sys.platform).provider_library
    with ExitStack() as resources:
        ort.register_execution_provider_library("dawn", str(provider.resolve()))
        resources.callback(ort.unregister_execution_provider_library, "dawn")
        host = DawnHost(assets_path / "webgpu.py")
        resources.callback(host.close)
        mask_array, capacity, provider_name, gpu_timings = _infer_patches(
            host,
            context,
            plans,
            _PatchExecution(image, model_path / "modelo.onnx", observer),
            queue_depth=queue_depth,
        )
        backend = host.backend
    timings.update(
        gpu_timings, ct_read=read_seconds, inference=time.perf_counter() - inference_started
    )
    mask = _mask_image(mask_array, image.reference)
    sitk.WriteImage(mask, str(output_path), useCompression=True)
    timings["total"] = time.perf_counter() - started
    logger.info("ModelO finished %s patches in %.2f s", len(plans), timings["total"])
    return {
        "worker": "model_o",
        "model_file": "modelo.onnx",
        "provider": provider_name,
        "backend": backend,
        "input_space": "input-voxel-zyx",
        "input_shape_zyx": list(image.array.shape),
        "spacing_zyx": list(image.spacing),
        **_patch_metadata(context, plans, capacity, queue_depth),
        "overlap": parameters.overlap,
        "timings_seconds": timings,
        "timings_kind": "wall_clock",
        "logit_threshold": LOGIT_THRESHOLD,
        "steps": ["gpu_ray_inputs", "gpu_model_o", "logit_threshold", "binary_union"],
    }
