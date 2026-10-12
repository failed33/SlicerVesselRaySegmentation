"""Bounded GPU submissions retain outputs and release pending readbacks on failure."""

from __future__ import annotations

from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from model_o_inference import runtime
from model_o_inference.planning import PatchPlan
from model_o_inference.shaders import load_shader


class _Buffer:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values.copy()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Host:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.buffers: list[_Buffer] = []
        self.stages: list[_Buffer] = []
        self.fail_stage: int | None = None
        self.fail_read: int | None = None
        self.reads = 0

    def buffer(self, size: int) -> _Buffer:
        buffer = _Buffer(np.zeros(size // 4, dtype=np.uint32))
        self.buffers.append(buffer)
        return buffer

    def write(self, buffer: _Buffer, values: np.ndarray) -> None:
        self.events.append("write")
        buffer.values = values.copy().reshape(-1)

    def stage_readback(self, buffer: _Buffer) -> _Buffer:
        if len(self.stages) + 1 == self.fail_stage:
            raise RuntimeError("stage failed")
        self.events.append("stage")
        staging = _Buffer(buffer.values)
        self.stages.append(staging)
        return staging

    def read_staging(self, staging: _Buffer, dtype=np.float32) -> np.ndarray:
        self.events.append("read")
        self.reads += 1
        if self.reads == self.fail_read:
            # Native mapping can fail before read_staging enters its cleanup block.
            raise RuntimeError("map failed")
        values = staging.values.astype(dtype).copy()
        staging.close()
        return values


def _pipeline(host: _Host, *, bad_logits: bool = False, bad_probability: bool = False):
    probability = host.buffer(48**3 * 4)
    invalid = host.buffer(4)
    current = [0.0]

    def generate(_center, h0):
        host.events.append("generate")
        current[0] = h0

    def dispatch(_workgroups):
        host.events.append("sigmoid")
        probability.values = np.full(48**3, current[0], dtype=np.float32)
        if bad_probability:
            probability.values[0] = np.nan
        if bad_logits:
            invalid.values[0] = 1

    builder = SimpleNamespace(
        write_geometry=lambda _geometry: host.events.append("geometry"), generate=generate
    )
    session = SimpleNamespace(
        submit=lambda: host.events.append("ort"),
        native_session=SimpleNamespace(get_providers=lambda: ["WebGpuExecutionProvider"]),
    )
    return runtime._GpuPatchInference(
        host,
        builder,
        session,
        SimpleNamespace(dispatch=dispatch),
        probability,
        invalid,
        load_shader("sigmoid").workgroups(48**3),
    )


def _cases(count: int):
    for index in range(count):
        plan = PatchPlan(np.array([24, 24, 24]), np.array([index]), (index + 1) / 10)
        case = SimpleNamespace(
            packed_geometry=np.zeros((1, 7), dtype=np.uint32), center=plan.center_zyx, h0=plan.h0
        )
        yield plan, case


@pytest.mark.parametrize("queue_depth", [1, 2, 4])
def test_window_copies_each_output_before_reuse_and_collects_partial_tail(queue_depth) -> None:
    host = _Host()
    pipeline = _pipeline(host)
    timings = {"patch_submission": 0.0, "patch_collection": 0.0}
    saved = []
    with closing(
        runtime._predict_windows(pipeline, _cases(7), queue_depth, timings)
    ) as predictions:
        for plan, probability in predictions:
            saved.append(probability)
            np.testing.assert_array_equal(probability, np.full((48,) * 3, plan.h0, np.float32))
    expected = []
    submit_events = ["geometry", "generate", "write", "ort", "sigmoid", "stage", "stage"]
    for start in range(0, 7, queue_depth):
        count = min(queue_depth, 7 - start)
        expected.extend(submit_events * count + ["read", "read"] * count)
    assert host.events == expected
    assert len(host.stages) == 14 and all(stage.closed for stage in host.stages)
    assert all(value >= 0 for value in timings.values())
    for index, probability in enumerate(saved):
        np.testing.assert_array_equal(probability, np.full((48,) * 3, (index + 1) / 10, np.float32))
        assert not np.shares_memory(probability, pipeline.probability.values)
    assert not np.shares_memory(saved[0], saved[-1])


@pytest.mark.parametrize("queue_depth", [1, 2, 4])
@pytest.mark.parametrize(
    "failure", ["second_stage", "first_map", "second_map", "logits", "probability"]
)
def test_submission_and_collection_failures_close_all_pending_stages(queue_depth, failure) -> None:
    host = _Host()
    pipeline = _pipeline(
        host, bad_logits=failure == "logits", bad_probability=failure == "probability"
    )
    if failure == "second_stage":
        host.fail_stage = 2
    elif failure in ("first_map", "second_map"):
        host.fail_read = 1 if failure == "first_map" else 2
    timings = {"patch_submission": 0.0, "patch_collection": 0.0}
    with pytest.raises(RuntimeError, match="stage failed|map failed|non-finite"):
        with closing(
            runtime._predict_windows(pipeline, _cases(7), queue_depth, timings)
        ) as predictions:
            next(predictions)
    assert host.stages and all(stage.closed for stage in host.stages)
    expected_stages = 1 if failure == "second_stage" else queue_depth * 2
    assert len(host.stages) == expected_stages


@pytest.mark.parametrize("queue_depth", [2, 4])
@pytest.mark.parametrize("failure", ["observer", "merge"])
def test_consumer_failure_closes_uncollected_stages_before_group_and_ct(
    monkeypatch, queue_depth, failure
) -> None:
    host = _Host()
    plans = [plan for plan, _case in _cases(7)]
    closed = []

    def group(resources, _host, _model, _case, ct_buffer):
        pipeline = _pipeline(host)

        def close_group():
            assert all(stage.closed for stage in host.stages)
            assert not ct_buffer.closed
            closed.append("group")
            pipeline.probability.close()
            pipeline.invalid.close()

        resources.callback(close_group)
        return pipeline

    def fail(*_args):
        raise RuntimeError("consumer failed")

    monkeypatch.setattr(runtime, "_gpu_pipeline", group)
    monkeypatch.setattr(
        runtime,
        "_ray_case",
        lambda _image, _context, plan, _capacity: SimpleNamespace(
            packed_geometry=np.zeros((1, 7), dtype=np.uint32), center=plan.center_zyx, h0=plan.h0
        ),
    )
    if failure == "merge":
        from model_o_inference.stitching import MaxMerger

        monkeypatch.setattr(MaxMerger, "aggregate", fail)
    execution = runtime._PatchExecution(
        SimpleNamespace(array=np.zeros((48,) * 3, dtype=np.float32)),
        Path("unused.onnx"),
        SimpleNamespace(on_progress=fail) if failure == "observer" else None,
    )
    with pytest.raises(RuntimeError, match="consumer failed"):
        runtime._infer_patches(host, None, plans, execution, queue_depth=queue_depth)
    assert closed == ["group"]
    assert len(host.stages) == queue_depth * 2
    assert all(stage.closed for stage in host.stages)
    assert all(buffer.closed for buffer in host.buffers)
    assert host.reads == 2


def test_invalid_logits_flag_is_reset_before_each_prediction() -> None:
    host = _Host()
    pipeline = _pipeline(host)
    pipeline.invalid.values[0] = 1
    with ExitStack() as resources:
        _plan, case = next(_cases(1))
        pending = pipeline.submit(case, resources)
        np.testing.assert_array_equal(
            pipeline.collect(pending), np.full((48,) * 3, 0.1, np.float32)
        )
