"""Ray-sampling geometry shared by all segmentation producers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OriginSamplingPolicy = Literal["max_spacing", "legacy_serving"]


@dataclass(frozen=True)
class RayGeometryConfig:
    """Geometry and origin placement for ray sampling.

    Pattern/learned producers receive caller-supplied values (UI knobs,
    sweep scripts, fast-test smoke configs). The ModelO factory locks
    these to its training manifest via
    :func:`model_o_inference.constants.model_o_geometry`
    -- that factory lives in the model-o package to keep ``core`` a leaf.
    """

    num_rays: int
    rays_oversample_factor: float
    min_tangent_weight: float
    sample_spacing_mm: float
    step_size_mm: float
    max_travel_mm: float
    origin_sampling_policy: OriginSamplingPolicy = "legacy_serving"
