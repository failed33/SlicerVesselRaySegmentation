"""Exercise the installed numerical application core identically on both hosts."""

from __future__ import annotations


def application_summary() -> dict[str, object]:
    import model_o_inference.preparation as preparation
    import numpy as np
    from model_o_inference.geometry import RayGeometryConfig
    from model_o_inference.preparation import prepare_ray_context

    z, y, x = np.indices((5, 5, 5))
    volume = (100 * z + 10 * y + x).astype(np.float32)
    geometry = RayGeometryConfig(8, 1.5, 0.1, 1.0, 0.25, 1.0)
    context = prepare_ray_context([[[2, 2, 1], [2, 2, 3]]], volume, geometry, (1, 1, 1))
    positions = np.array([[2, 2, 1], [2, 2, 2], [2, 2, 3]], dtype=np.float64)
    np.testing.assert_array_equal(context.sample_positions, positions)
    np.testing.assert_array_equal(context.sample_hu, [221, 222, 223])
    np.testing.assert_array_equal(context.sample_edges, [0, 0, 0])
    np.testing.assert_array_equal(context.ray_origin_hu, np.repeat([221, 222, 223], 8))
    np.testing.assert_array_equal(
        context.ray_lattice_fp[:, :3], np.repeat(positions, 8, axis=0) * 65536
    )
    assert context.n_rays == 24
    assert context.ray_lattice_fp.dtype == np.int32
    assert np.isfinite(context.ray_lattice_fp).all()
    return {
        "status": "passed",
        "sample_positions_zyx": context.sample_positions.tolist(),
        "sample_hu": context.sample_hu.tolist(),
        "ray_count": context.n_rays,
        "lattice_shape": list(context.ray_lattice_fp.shape),
        "module_path": str(preparation.__file__),
    }
