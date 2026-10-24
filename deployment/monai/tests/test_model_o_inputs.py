"""Patch input ownership and exact packed geometry, without a GPU device."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from dawn_ort.host import Buffer, DawnHost, Kernel
from model_o_inference import inputs
from model_o_inference.dawn_inputs import DawnPatchInputs
from model_o_inference.inputs import PackedRayCase, RayCase, pack_ray_geometry


class _Kernel:
    def __init__(self) -> None:
        self.dispatches: list[int] = []
        self.closed = False

    def dispatch(self, workgroups: int) -> None:
        self.dispatches.append(workgroups)

    def close(self) -> None:
        self.closed = True


class _Host:
    def __init__(self) -> None:
        self.buffers: set[Buffer] = set()
        self.allocated: list[Buffer] = []
        self.writes: list[tuple[Buffer, np.ndarray]] = []
        self.released: list[int] = []
        self.w = SimpleNamespace(wgpuBufferRelease=self.released.append)
        self.kernels: list[_Kernel] = []
        self.fail_kernel = False

    def buffer(self, size: int) -> Buffer:
        buffer = Buffer(cast(DawnHost, self), len(self.allocated) + 1, size, (size + 15) // 16 * 16)
        self.allocated.append(buffer)
        return buffer

    def write(self, buffer: Buffer, values: np.ndarray) -> None:
        assert buffer.size == values.nbytes
        assert not buffer.closed
        self.writes.append((buffer, np.array(values, copy=True)))

    def kernel(
        self, _source: str, buffers: dict[int, Buffer], *, entry_point: str = "main"
    ) -> Kernel:
        assert entry_point == "main"
        assert all(not buffer.closed for buffer in buffers.values())
        if self.fail_kernel:
            raise RuntimeError("kernel compilation failed")
        kernel = _Kernel()
        self.kernels.append(kernel)
        return cast(Kernel, kernel)


def _case(n_rays: int = 2) -> RayCase:
    return RayCase(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        (0.7, 1.1, 1.4),
        np.arange(n_rays * 3, dtype=np.float64).reshape(n_rays, 3) / 3 - 2,
        np.tile([0.1, -0.3, 0.9], (n_rays, 1)),
        np.linspace(-1024.25, 333.5, n_rays, dtype=np.float32),
        np.array([1, 2, 3]),
        250.0,
    )


def test_capacity_builders_borrow_one_request_ct_upload_and_never_close_it() -> None:
    host = _Host()
    first = _case(2)
    ct_buffer = host.buffer(first.volume.nbytes)
    host.write(ct_buffer, first.volume)
    for case in (first, _case(4)):
        builder = DawnPatchInputs(cast(DawnHost, host), case, ct_buffer=ct_buffer, initialize=False)
        assert builder.storage["volume_zyx"] is ct_buffer
        assert host.kernels[-1].dispatches == []
        assert not any(buffer is builder.storage["ray_geometry"] for buffer, _values in host.writes)
        builder.write_geometry(case.packed_geometry)
        builder.generate(case.center, case.h0)
        assert len(host.kernels[-1].dispatches) == 1
        builder.close()
        builder.close()
        assert host.kernels[-1].closed
        assert not ct_buffer.closed
        assert host.buffers == {ct_buffer}
    writes = [values for buffer, values in host.writes if buffer is ct_buffer]
    assert len(writes) == 1
    np.testing.assert_array_equal(writes[0], first.volume)
    ct_buffer.close()
    assert not host.buffers


def test_legacy_case_still_uploads_owned_volume_and_generates_example_inputs() -> None:
    host = _Host()
    case = _case()
    builder = DawnPatchInputs(cast(DawnHost, host), case)
    assert len(host.kernels[0].dispatches) == 1
    np.testing.assert_array_equal(host.writes[0][1], case.volume)
    np.testing.assert_array_equal(host.writes[1][1], case.packed_geometry)
    assert builder.gpu_inputs["ray_mask"].dtype == np.dtype(bool)
    assert builder.gpu_inputs["voxel_indices"].dtype == np.dtype(np.int64)
    builder.close()
    builder.close()
    assert host.kernels[0].closed
    assert not host.buffers
    assert len(host.released) == len(host.allocated)


def test_packed_geometry_matches_legacy_hu_and_q16_words() -> None:
    case = _case()
    expected = np.column_stack([case.origin_hu.view(np.uint32), case.lattice.view(np.uint32)])
    actual = pack_ray_geometry(case.lattice, case.origin_hu)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint32
    assert actual.flags.c_contiguous


def test_direct_packed_case_preserves_q16_bits_and_uploads_without_quantization(
    monkeypatch,
) -> None:
    host = _Host()
    case = _case()
    lattice = np.array(
        [[-2147483648, 2147483647, -131072, 1, -1, 32768], [0, 65536, -65536, 0, 0, 0]],
        dtype=np.int32,
    )
    hu = np.array([-1024.25, -0.0], dtype=np.float32)
    geometry = pack_ray_geometry(lattice, hu)
    packed = PackedRayCase(case.volume, case.spacing, geometry, case.center, case.h0)
    monkeypatch.setattr(
        inputs,
        "quantize_to_fixed",
        lambda *_args: pytest.fail("Already packed Q16 geometry must not be requantized"),
    )
    builder = DawnPatchInputs(cast(DawnHost, host), packed, initialize=False)
    assert packed.packed_geometry is geometry
    builder.write_geometry(packed.packed_geometry)
    np.testing.assert_array_equal(host.writes[-1][1][:, 1:].view(np.int32), lattice)
    np.testing.assert_array_equal(host.writes[-1][1][:, 0], hu.view(np.uint32))
    changed = geometry.copy()
    changed[0, 1] = np.uint32(123456)
    builder.write_geometry(changed)
    np.testing.assert_array_equal(host.writes[-1][1], changed)
    builder.generate(case.center, case.h0)
    config = host.writes[-1][1]
    assert [config[name][0] for name in ("depth", "height", "width", "rayCount")] == [2, 3, 4, 2]
    builder.close()


@pytest.mark.parametrize("failure", ["closed", "other_host", "wrong_size"])
def test_borrowed_ct_requires_a_live_matching_buffer_before_allocations(failure: str) -> None:
    host = _Host()
    case = _case()
    owner = _Host() if failure == "other_host" else host
    buffer = owner.buffer(case.volume.nbytes + (4 if failure == "wrong_size" else 0))
    if failure == "closed":
        buffer.close()
    allocated = len(host.allocated)
    with pytest.raises(ValueError, match="Borrowed CT buffer"):
        DawnPatchInputs(cast(DawnHost, host), case, ct_buffer=buffer)
    assert len(host.allocated) == allocated
    assert buffer.closed == (failure == "closed")
    buffer.close()


@pytest.mark.parametrize("borrow_ct", [False, True])
def test_failed_builder_closes_owned_buffers_and_preserves_borrowed_ct(borrow_ct: bool) -> None:
    host = _Host()
    case = _case()
    ct_buffer = host.buffer(case.volume.nbytes) if borrow_ct else None
    host.fail_kernel = True
    with pytest.raises(RuntimeError, match="kernel compilation failed"):
        DawnPatchInputs(cast(DawnHost, host), case, ct_buffer=ct_buffer)
    assert host.buffers == ({ct_buffer} if ct_buffer is not None else set())
    if ct_buffer is not None:
        assert not ct_buffer.closed
        ct_buffer.close()


@pytest.mark.parametrize("dtype,shape", [(np.float32, (2, 7)), (np.uint32, (3, 7))])
def test_packed_geometry_upload_rejects_wrong_dtype_or_capacity(dtype, shape) -> None:
    host = _Host()
    builder = DawnPatchInputs(cast(DawnHost, host), _case(), initialize=False)
    writes = len(host.writes)
    with pytest.raises(ValueError, match="Packed geometry"):
        builder.write_geometry(np.zeros(shape, dtype=dtype))
    assert len(host.writes) == writes
    builder.close()
