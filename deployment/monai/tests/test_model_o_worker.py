"""Hosted ModelO polyline contract and shared ray-preparation wiring."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from model_o_inference.runtime import _read_ct, predict_model_o
from pydantic import ValidationError

from monai_service.contracts import ArtifactIdentity, InferenceRequest, ResolvedArtifact
from monai_service.models.model_o import (
    TASK,
    ModelOParameters,
    ModelOSettings,
    ModelOWorkflow,
)


@pytest.fixture
def polyline_data() -> dict[str, object]:
    return {
        "polylines_zyx": [
            [[3.25, 6, 5], [4.25, 6, 5], [5.25, 6.5, 5]],
            [[7, 8, 9], [6, 8, 9]],
        ]
    }


def _host_double():
    return SimpleNamespace(
        backend="Vulkan",
        buffer=lambda _size: SimpleNamespace(close=lambda: None),
        write=lambda *_args: None,
        close=lambda: None,
    )


def _pipeline_double(predict):
    return SimpleNamespace(
        submit=lambda case, _resources: predict(case),
        collect=lambda probability: probability,
        session=SimpleNamespace(
            native_session=SimpleNamespace(get_providers=lambda: ["WebGpuExecutionProvider"])
        ),
    )


def test_polyline_schema_preserves_order_and_freezes_nested_values(
    polyline_data: dict[str, object],
) -> None:
    parameters = TASK.parse_parameters(polyline_data)
    restored = ModelOParameters.model_validate_json(parameters.model_dump_json())
    assert restored.polylines_zyx == (
        ((3.25, 6.0, 5.0), (4.25, 6.0, 5.0), (5.25, 6.5, 5.0)),
        ((7.0, 8.0, 9.0), (6.0, 8.0, 9.0)),
    )
    with pytest.raises(ValidationError, match="frozen"):
        parameters.polylines_zyx = ()
    with pytest.raises(TypeError):
        parameters.polylines_zyx[0][0][0] = 99


@pytest.mark.parametrize("polylines", [[], [[]], [[[1, 2, 3]]]])
def test_polyline_contract_requires_a_line_with_at_least_two_points(polylines: object) -> None:
    with pytest.raises(ValidationError):
        ModelOParameters(polylines_zyx=polylines)


@pytest.mark.parametrize(
    "coordinate",
    [
        [3, 6],
        [3, 6, 5, 1],
        [True, 6, 5],
        ["3", 6, 5],
        [float("nan"), 6, 5],
        [float("inf"), 6, 5],
        [float("-inf"), 6, 5],
        [10**400, 6, 5],
    ],
)
def test_polyline_coordinates_are_finite_numbers_in_three_dimensions(coordinate: object) -> None:
    with pytest.raises(ValidationError):
        ModelOParameters(polylines_zyx=[[coordinate, [4, 6, 5]]])


def test_repeated_polyline_points_keep_the_sampler_contract() -> None:
    parameters = ModelOParameters(polylines_zyx=[[[3, 6, 5], [3, 6, 5]]])
    assert parameters.polylines_zyx == (((3.0, 6.0, 5.0), (3.0, 6.0, 5.0)),)


@pytest.mark.parametrize("extra", ["tree", "ray_lattice_fp", "ray_origin_hu", "sample_hu"])
def test_request_rejects_retired_tree_and_internal_ray_metadata(
    polyline_data: dict[str, object], extra: str
) -> None:
    with pytest.raises(ValidationError) as failure:
        ModelOParameters.model_validate({**polyline_data, extra: []})
    assert any(error["type"] == "extra_forbidden" for error in failure.value.errors())
    assert set(ModelOParameters.model_json_schema()["properties"]) == {"polylines_zyx", "overlap"}


@pytest.mark.parametrize("overlap", [None, 0.0, 0.25, 0.5, 0.99])
def test_overlap_preserves_requested_sparse_or_dense_policy(polyline_data, overlap) -> None:
    parameters = ModelOParameters.model_validate({**polyline_data, "overlap": overlap})
    assert parameters.overlap == overlap
    assert ModelOParameters.model_validate_json(parameters.model_dump_json()).overlap == overlap


@pytest.mark.parametrize("overlap", [-0.1, 1.0, True, "0.5", float("nan"), float("inf")])
def test_overlap_rejects_invalid_ranges_and_types(polyline_data, overlap) -> None:
    with pytest.raises(ValidationError):
        ModelOParameters.model_validate({**polyline_data, "overlap": overlap})


def test_overlap_defaults_to_half_and_patch_size_is_not_a_request_option(polyline_data) -> None:
    assert ModelOParameters.model_validate(polyline_data).overlap == 0.5
    with pytest.raises(ValidationError, match="patch_size"):
        ModelOParameters.model_validate({**polyline_data, "patch_size": 64})


def test_workflow_serializes_polylines_for_the_worker(
    polyline_data: dict[str, object], tmp_path: Path
) -> None:
    parameters = ModelOParameters.model_validate(polyline_data)
    workflow = ModelOWorkflow(ModelOSettings(assets=tmp_path / "dawn-assets"))
    input_path = tmp_path / "ct.nrrd"
    artifact = ResolvedArtifact(
        ArtifactIdentity("model_o", "7", "models:/model_o/7"), tmp_path / "model"
    )
    invocation = workflow.prepare(InferenceRequest(input_path, parameters), artifact, tmp_path)
    bindings = {binding.name: binding for binding in invocation.bindings}
    assert workflow.descriptor.input_profile == "polylines"
    assert json.loads(bindings["parameters"].path.read_text()) == {**polyline_data, "overlap": 0.5}
    assert bindings["image"].path == input_path
    assert bindings["model"].path == artifact.path
    assert bindings["assets"].path == workflow.settings.assets
    depth_index = invocation.arguments.index("--queue-depth")
    assert invocation.arguments[depth_index + 1] == "1"


@pytest.mark.parametrize("queue_depth", [1, 2, 4])
def test_workflow_passes_selected_queue_depth(polyline_data, tmp_path, queue_depth) -> None:
    parameters = ModelOParameters.model_validate(polyline_data)
    workflow = ModelOWorkflow(ModelOSettings(assets=tmp_path, queue_depth=queue_depth))
    artifact = ResolvedArtifact(ArtifactIdentity("model_o", "7", "models:/model_o/7"), tmp_path)
    invocation = workflow.prepare(
        InferenceRequest(tmp_path / "ct.nrrd", parameters), artifact, tmp_path
    )
    depth_index = invocation.arguments.index("--queue-depth")
    assert invocation.arguments[depth_index + 1] == str(queue_depth)


@pytest.mark.parametrize("queue_depth", [0, 3, 8, "bad"])
def test_workflow_rejects_unsupported_queue_depth(queue_depth) -> None:
    with pytest.raises(ValidationError):
        ModelOSettings(assets=Path("unused"), queue_depth=queue_depth)


@pytest.mark.parametrize("queue_depth", ["1", "2", "4"])
def test_workflow_queue_depth_environment_uses_selected_integer(monkeypatch, queue_depth) -> None:
    monkeypatch.setenv("MONAI_SERVICE_MODEL_O_QUEUE_DEPTH", queue_depth)
    assert ModelOSettings(assets=Path("unused")).queue_depth == int(queue_depth)


@pytest.mark.parametrize("queue_depth", ["0", "3", "bad"])
def test_workflow_rejects_unsupported_queue_depth_environment(monkeypatch, queue_depth) -> None:
    monkeypatch.setenv("MONAI_SERVICE_MODEL_O_QUEUE_DEPTH", queue_depth)
    with pytest.raises(ValidationError):
        ModelOSettings(assets=Path("unused"))


@pytest.mark.parametrize("overlap", [None, 0.25, 0.5])
def test_worker_passes_submitted_ct_and_polylines_to_shared_preparation(
    polyline_data: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlap
) -> None:
    from model_o_inference import planning, preparation
    from model_o_inference.constants import model_o_geometry

    voxels = np.arange(12**3, dtype=np.float32).reshape((12,) * 3)
    image = sitk.GetImageFromArray(voxels)
    image.SetSpacing((0.8, 1.1, 1.4))
    input_path, output_path = tmp_path / "ct.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(input_path))
    parameters = ModelOParameters.model_validate({**polyline_data, "overlap": overlap})
    prepare = preparation.prepare_ray_context
    captured = []

    class PlanningReached(Exception):
        """Stop after real preprocessing, before any GPU inference is attempted."""

    def record_preparation(*, polylines_zyx, volume, geometry, spacing_zyx):
        assert polylines_zyx == parameters.polylines_zyx
        np.testing.assert_array_equal(volume, voxels)
        assert volume.dtype == np.float32
        assert geometry == model_o_geometry()
        assert spacing_zyx == (1.4, 1.1, 0.8)
        context = prepare(polylines_zyx, volume, geometry, spacing_zyx)
        captured.append(context)
        return context

    def reach_planner(context, *, overlap, polylines_zyx):
        assert context is captured[0]
        assert overlap == parameters.overlap
        assert polylines_zyx == parameters.polylines_zyx
        assert context.n_rays > 0
        assert set(context.sample_edges) == {0, 1}
        raise PlanningReached

    monkeypatch.setattr(preparation, "prepare_ray_context", record_preparation)
    monkeypatch.setattr(planning, "build_patch_plans", reach_planner)
    with pytest.raises(PlanningReached):
        predict_model_o(
            input_path, tmp_path / "model", output_path, parameters, tmp_path / "assets"
        )
    assert len(captured) == 1
    assert not output_path.exists()


@pytest.mark.parametrize("kind", ["2d", "vector", "nonfinite"])
def test_worker_rejects_non_scalar_or_nonfinite_ct(kind: str, tmp_path: Path) -> None:
    if kind == "2d":
        image = sitk.Image([4, 4], sitk.sitkFloat32)
    elif kind == "vector":
        image = sitk.GetImageFromArray(np.zeros((4, 4, 4, 2), dtype=np.float32), isVector=True)
    else:
        voxels = np.zeros((4,) * 3, dtype=np.float32)
        voxels[0, 0, 0] = np.nan
        image = sitk.GetImageFromArray(voxels)
    input_path = tmp_path / "ct.nrrd"
    sitk.WriteImage(image, str(input_path))
    with pytest.raises(ValueError, match="scalar 3D CT|non-finite intensities"):
        _read_ct(input_path)


def test_sparse_selected_patch_gpu_geometry_matches_dense_reference(
    polyline_data: dict[str, object], tmp_path: Path
) -> None:
    from model_o_inference.planning import build_patch_plans
    from model_o_inference.runtime import _prepare_patches, _ray_case

    image = sitk.GetImageFromArray(np.full((12,) * 3, 250, dtype=np.float32))
    image.SetSpacing((0.8, 1.1, 1.4))
    input_path = tmp_path / "ct.nrrd"
    sitk.WriteImage(image, str(input_path))
    volume = _read_ct(input_path)
    context, sparse, _timings = _prepare_patches(
        volume, ModelOParameters.model_validate(polyline_data)
    )
    dense = build_patch_plans(context)
    capacity = max(len(plan.ray_indices) for plan in dense)
    for plan in sparse:
        reference = next(
            candidate
            for candidate in dense
            if np.array_equal(candidate.center_zyx, plan.center_zyx)
            and np.array_equal(candidate.ray_indices, plan.ray_indices)
            and candidate.h0 == plan.h0
        )
        actual = _ray_case(volume, context, plan, capacity)
        expected = _ray_case(volume, context, reference, capacity)
        for field in ("packed_geometry", "center"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        assert actual.h0 == expected.h0


def test_gpu_patch_progress_logs_first_periodic_and_last_completion(monkeypatch, caplog) -> None:
    from model_o_inference import runtime
    from model_o_inference.planning import PatchPlan

    plans = [PatchPlan(np.array([6, 6, 6]), np.array([index]), 250.0) for index in range(4)]
    pipeline = _pipeline_double(lambda _case: np.zeros((48,) * 3, dtype=np.float32))
    monkeypatch.setattr(runtime, "_gpu_pipeline", lambda *_args: pipeline)
    monkeypatch.setattr(runtime, "_ray_case", lambda _image, _context, plan, _capacity: plan)
    clock = [0.0]
    pipeline.collect = lambda probability: (clock.__setitem__(0, clock[0] + 5), probability)[1]
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: clock[0])
    caplog.set_level("INFO", logger=runtime.__name__)
    volume = SimpleNamespace(array=np.zeros((12,) * 3, dtype=np.float32))

    probability, capacity, provider, _timings = runtime._infer_patches(
        _host_double(), None, plans, runtime._PatchExecution(volume, Path("unused.onnx"))
    )
    assert probability.shape == (12,) * 3
    assert capacity == 1
    assert provider == "WebGpuExecutionProvider"
    messages = [record.message for record in caplog.records if "ModelO patches" in record.message]
    assert [message.split()[2] for message in messages] == ["1/4", "3/4", "4/4"]


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([1, 1], {1: 2}),
        ([1024, 1024], {1024: 2}),
        ([1024, 2048, 4096], {1024: 1, 2048: 1, 4096: 1}),
        ([17408, 56320], {32768: 1, 56320: 1}),
        ([1024, 56320, 8192, 32768, 17408, 3072], {16384: 3, 32768: 2, 56320: 1}),
    ],
)
def test_capacity_groups_bound_session_count_and_cap_at_actual_maximum(counts, expected) -> None:
    from model_o_inference.planning import PatchPlan
    from model_o_inference.runtime import _capacity_groups

    plans = [PatchPlan(np.array([32, 32, 32]), np.arange(count), 250.0) for count in counts]
    groups = _capacity_groups(plans)
    assert {capacity: len(group) for capacity, group in groups.items()} == expected
    assert list(groups) == sorted(expected)
    assert max(groups) == max(counts)
    assert len(groups) <= 3
    assert sorted(id(plan) for group in groups.values() for plan in group) == sorted(map(id, plans))
    assert all(
        len(plan.ray_indices) <= capacity for capacity, group in groups.items() for plan in group
    )


def test_smaller_capacity_keeps_real_ray_geometry_and_invalid_padding() -> None:
    from model_o_inference.lattice import FIXED_POINT_SCALE
    from model_o_inference.planning import PatchPlan
    from model_o_inference.runtime import _ray_case

    context = SimpleNamespace(
        ray_lattice_fp=np.array([[3, 6, 5, 1, 2, 3], [4, 6, 5, 2, 3, 4]]) * FIXED_POINT_SCALE,
        ray_origin_hu=np.array([250, 275], dtype=np.float32),
    )
    volume = SimpleNamespace(array=np.zeros((12,) * 3, dtype=np.float32), spacing=(1.4, 1.1, 0.8))
    plan = PatchPlan(np.array([6, 6, 6]), np.array([1, 0]), 250.0)
    small, large = (_ray_case(volume, context, plan, capacity) for capacity in (4, 8))
    np.testing.assert_array_equal(small.packed_geometry, large.packed_geometry[:4])
    lattice = small.packed_geometry[:, 1:].view(np.int32)
    np.testing.assert_array_equal(lattice[:2], context.ray_lattice_fp[[1, 0]])
    np.testing.assert_array_equal(lattice[2:, :3], np.full((2, 3), -2 * FIXED_POINT_SCALE))
    np.testing.assert_array_equal(lattice[2:, 3:], np.zeros((2, 3), dtype=np.int32))
    np.testing.assert_array_equal(small.packed_geometry[2:, 0].view(np.float32), np.ones(2))
    assert small.volume is large.volume is volume.array
    assert small.h0 == large.h0 == plan.h0
    np.testing.assert_array_equal(small.center, large.center)


@pytest.mark.parametrize("preview_stride", [None, 0, 1, 2])
@pytest.mark.parametrize("queue_depth", [1, 2, 4])
def test_grouped_inference_reuses_host_closes_groups_and_preserves_max_union(
    monkeypatch, caplog, preview_stride, queue_depth
) -> None:
    from model_o_inference import runtime
    from model_o_inference.constants import THRESHOLD
    from model_o_inference.planning import PatchPlan
    from model_o_inference.stitching import MaxMerger

    counts = [1024, 56320, 8192, 32768, 17408, 3072]
    plans = [
        PatchPlan(np.array([32, 20 + index * 2, 32]), np.arange(count), (index + 1) / 10)
        for index, count in enumerate(counts)
    ]
    host, context = _host_double(), object()
    reference = sitk.GetImageFromArray(np.zeros((64,) * 3, dtype=np.float32))
    reference.SetSpacing((0.8, 1.1, 1.4))
    reference.SetOrigin((123.5, -80.25, 9.75))
    reference.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    volume = SimpleNamespace(array=sitk.GetArrayFromImage(reference), reference=reference)
    active, closed, predicted = [], [], []
    progress_updates, previews, finalized = [], [], []
    original_finalize = MaxMerger.finalize

    def finalize(merger):
        finalized.append(True)
        return original_finalize(merger)

    def on_progress(progress, snapshot):
        progress_updates.append(progress)
        if preview_stride and progress.completed % preview_stride == 0:
            previews.append(snapshot())

    observer = None if preview_stride is None else SimpleNamespace(on_progress=on_progress)

    def ray_case(image, ray_context, plan, capacity):
        assert image is volume and ray_context is context
        return plan, capacity

    def pipeline(resources, group_host, _model, first_case, _ct_buffer):
        assert group_host is host and not active
        capacity = first_case[1]
        active.append(capacity)

        def close():
            assert active.pop() == capacity
            closed.append(capacity)

        def predict(case):
            plan, case_capacity = case
            assert active == [capacity] and case_capacity == capacity
            assert len(plan.ray_indices) <= capacity
            predicted.append(plan)
            return np.full((48,) * 3, plan.h0, dtype=np.float32)

        resources.callback(close)
        return _pipeline_double(predict)

    monkeypatch.setattr(runtime, "_ray_case", ray_case)
    monkeypatch.setattr(runtime, "_gpu_pipeline", pipeline)
    monkeypatch.setattr(MaxMerger, "finalize", finalize)
    caplog.set_level("INFO", logger=runtime.__name__)
    probability, capacity, provider, timings = runtime._infer_patches(
        host,
        context,
        plans,
        runtime._PatchExecution(volume, Path("unused.onnx"), observer),
        queue_depth=queue_depth,
    )
    assert all(value >= 0 for value in timings.values())
    # Only requested previews and the final result copy the accumulated volume.
    assert len(finalized) == len(previews) + 1
    assert len(previews) == (len(plans) // preview_stride if preview_stride else 0)
    if observer is not None:
        assert [progress.completed for progress in progress_updates] == list(range(1, 7))
        assert all(progress.total == 6 for progress in progress_updates)
        elapsed = [progress.elapsed_seconds for progress in progress_updates]
        assert elapsed == sorted(elapsed) and elapsed[0] >= 0
    for preview in previews:
        assert preview.GetPixelID() == sitk.sitkUInt8
        for getter in ("GetSize", "GetSpacing", "GetDirection", "GetOrigin"):
            assert getattr(preview, getter)() == getattr(reference, getter)()
    arrays = [sitk.GetArrayFromImage(preview) for preview in previews]
    assert all(np.all(previous <= current) for previous, current in zip(arrays, arrays[1:]))
    if arrays:
        np.testing.assert_array_equal(arrays[-1], (probability > THRESHOLD).astype(np.uint8))
        # Editing a returned image cannot change the worker's aggregation buffer.
        previews[-1][32, 32, 32] = 0
        assert probability[32, 32, 32] > THRESHOLD
    expected = MaxMerger(volume.array.shape)
    for plan in plans:
        expected.aggregate(np.full((48,) * 3, plan.h0, dtype=np.float32), tuple(plan.center_zyx))
    np.testing.assert_array_equal(probability, expected.finalize())
    plan_indices = {id(plan): index for index, plan in enumerate(plans)}
    assert [plan_indices[id(plan)] for plan in predicted] == [0, 2, 5, 3, 4, 1]
    assert closed == [16384, 32768, 56320] and not active
    assert capacity == 56320 and provider == "WebGpuExecutionProvider"
    assert any("ModelO patches 6/6" in record.message for record in caplog.records)


def test_grouped_inference_releases_current_group_when_prediction_fails(monkeypatch) -> None:
    from model_o_inference import runtime
    from model_o_inference.planning import PatchPlan

    plans = [PatchPlan(np.array([6, 6, 6]), np.arange(count), 250.0) for count in (1024, 4096)]
    closed = []

    def pipeline(resources, _host, _model, _case, _ct_buffer):
        resources.callback(lambda: closed.append(True))

        def predict(_case):
            raise RuntimeError("prediction failed")

        return _pipeline_double(predict)

    monkeypatch.setattr(runtime, "_gpu_pipeline", pipeline)
    monkeypatch.setattr(runtime, "_ray_case", lambda *_args: None)
    with pytest.raises(RuntimeError, match="prediction failed"):
        runtime._infer_patches(
            _host_double(),
            None,
            plans,
            runtime._PatchExecution(
                SimpleNamespace(array=np.zeros((12,) * 3, dtype=np.float32)),
                Path("unused.onnx"),
            ),
        )
    assert closed == [True]


@pytest.mark.parametrize("queue_depth", [1, 2, 4])
def test_preview_observer_failure_stops_patches_and_releases_gpu_group(
    monkeypatch, queue_depth
) -> None:
    from model_o_inference import runtime
    from model_o_inference.planning import PatchPlan

    plans = [PatchPlan(np.array([6, 6, 6]), np.array([index]), 250.0) for index in range(3)]
    predicted, closed = [], []

    def on_progress(_progress, _snapshot):
        raise RuntimeError("preview failed")

    def pipeline(resources, *_args):
        resources.callback(lambda: closed.append(True))

        def predict(case):
            predicted.append(case)
            return np.full((48,) * 3, 0.75, dtype=np.float32)

        return _pipeline_double(predict)

    monkeypatch.setattr(runtime, "_gpu_pipeline", pipeline)
    monkeypatch.setattr(runtime, "_ray_case", lambda _image, _context, plan, _capacity: plan)
    execution = runtime._PatchExecution(
        SimpleNamespace(array=np.zeros((12,) * 3, dtype=np.float32)),
        Path("unused.onnx"),
        SimpleNamespace(on_progress=on_progress),
    )
    with pytest.raises(RuntimeError, match="preview failed"):
        runtime._infer_patches(_host_double(), None, plans, execution, queue_depth=queue_depth)
    assert len(predicted) == min(queue_depth, len(plans))
    assert closed == [True]


def test_runtime_reports_sparse_planning_counts_timings_and_output_grid(
    polyline_data: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dawn_ort
    import onnxruntime as ort
    from model_o_inference import runtime

    image = sitk.GetImageFromArray(np.full((12,) * 3, 250, dtype=np.float32))
    image.SetSpacing((0.8, 1.1, 1.4))
    input_path, output_path = tmp_path / "ct.nrrd", tmp_path / "mask.nrrd"
    sitk.WriteImage(image, str(input_path))
    monkeypatch.setattr(ort, "register_execution_provider_library", lambda *_args: None)
    monkeypatch.setattr(ort, "unregister_execution_provider_library", lambda *_args: None)
    monkeypatch.setattr(dawn_ort, "DawnHost", lambda _path: _host_double())

    def pipeline(_resources, _host, _model, _case, _ct_buffer):
        return _pipeline_double(lambda _case: np.full((48,) * 3, 0.75, dtype=np.float32))

    monkeypatch.setattr(runtime, "_gpu_pipeline", pipeline)
    baseline = predict_model_o(
        input_path,
        tmp_path / "model",
        tmp_path / "baseline.nrrd",
        ModelOParameters.model_validate(polyline_data),
        tmp_path / "assets",
    )
    previews = []
    observer = SimpleNamespace(
        on_progress=lambda _progress, snapshot: previews.append(snapshot()),
    )
    metadata = predict_model_o(
        input_path,
        tmp_path / "model",
        output_path,
        ModelOParameters.model_validate(polyline_data),
        tmp_path / "assets",
        observer=observer,
    )
    assert {key: value for key, value in metadata.items() if key != "timings_seconds"} == {
        key: value for key, value in baseline.items() if key != "timings_seconds"
    }
    assert metadata["patches"] == 2
    assert metadata["patch_size"] == 48
    assert metadata["overlap"] == 0.5
    assert metadata["sample_spacing_mm"] == 1.0
    assert metadata["rays"] == metadata["ray_samples"] * 1024
    assert metadata["gathered_ray_slots"] == metadata["rays"]
    assert metadata["capacity_groups"] == [
        {"capacity": 2048, "patches": 1},
        {"capacity": 3072, "patches": 1},
    ]
    assert metadata["padded_ray_slots"] == 5120
    assert metadata["padded_ray_slots"] < metadata["patches"] * metadata["encoded_rays"]
    assert set(metadata["timings_seconds"]) == {
        "ct_read",
        "ray_preparation",
        "patch_planning",
        "inference",
        "total",
        "ct_upload",
        "group_setup",
        "patch_submission",
        "patch_collection",
        "patch_merge",
    }
    assert metadata["queue_depth"] == 1
    assert metadata["timings_kind"] == "wall_clock"
    assert all(value >= 0 for value in metadata["timings_seconds"].values())
    mask = sitk.ReadImage(str(output_path))
    assert mask.GetSize() == image.GetSize()
    assert mask.GetSpacing() == image.GetSpacing()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), np.ones((12,) * 3))
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(mask),
        sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "baseline.nrrd"))),
    )
    assert len(previews) == 2
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(previews[-1]), sitk.GetArrayFromImage(mask)
    )


@pytest.mark.skipif(
    not (os.environ.get("MODELO_MODEL_PATH") and os.environ.get("MODELO_NATIVE_ASSETS")),
    reason="Set MODELO_MODEL_PATH and MODELO_NATIVE_ASSETS for real native ModelO inference",
)
def test_real_native_model_o_preserves_binary_output_and_physical_geometry(tmp_path: Path) -> None:
    """Use the selected ONNX graph and native assets with the Dawn runtime available."""
    selected_model = Path(os.environ["MODELO_MODEL_PATH"])
    assets = Path(os.environ["MODELO_NATIVE_ASSETS"])
    assert selected_model.is_file(), "MODELO_MODEL_PATH must identify an ONNX file"
    assert assets.is_dir(), "MODELO_NATIVE_ASSETS must identify the native asset directory"
    model = tmp_path / "model"
    model.mkdir()
    shutil.copyfile(selected_model, model / "modelo.onnx")

    zz, yy, xx = np.indices((64,) * 3, dtype=np.float32)
    voxels = zz * 0.5 + yy * 0.25 + xx * 0.125 - 60
    voxels[(zz - 32) ** 2 + (yy - 32) ** 2 < 16] += 300
    image = sitk.GetImageFromArray(voxels)
    image.SetSpacing((0.9, 1.1, 0.7))
    image.SetOrigin((123.5, -80.25, 9.75))
    angle = np.pi / 6
    image.SetDirection((np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1))
    input_path, output_path = tmp_path / "ct.nrrd", tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(input_path))
    parameters = ModelOParameters(
        polylines_zyx=[
            [[32, 32, 30], [32, 32, 30.2]],
            [[32, 32, 36], [32, 32, 36.2]],
        ]
    )

    metadata = predict_model_o(input_path, model, output_path, parameters, assets)
    mask = sitk.ReadImage(str(output_path))
    assert mask.GetSize() == image.GetSize()
    assert mask.GetPixelID() == sitk.sitkUInt8
    assert set(np.unique(sitk.GetArrayFromImage(mask))).issubset({0, 1})
    for getter in ("GetSpacing", "GetDirection", "GetOrigin"):
        np.testing.assert_allclose(
            getattr(mask, getter)(), getattr(image, getter)(), atol=1e-5, rtol=0
        )
    assert metadata["provider"].lower() == "webgpuexecutionprovider"
    assert metadata["patches"] == 2
    assert metadata["rays"] == 2048
    assert metadata["encoded_rays"] == 1024
    assert metadata["input_space"] == "input-voxel-zyx"
    assert metadata["input_shape_zyx"] == [64, 64, 64]
    np.testing.assert_allclose(metadata["spacing_zyx"], [0.7, 1.1, 0.9], atol=1e-5, rtol=0)
