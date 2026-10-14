"""Constants for the ModelO segmentation producer.

All values here are fixed by the trained-model architecture or by the
training-data generation config. They are deliberately not user-tunable
for ModelO -- one config drives both training and inference.

Source of truth for the geometry values:
``artifacts/learnable_affinity/dataset-generation/build_training_dataset.py``
(``DATASET_VERSION = "5.0.0"``, ``NUM_RAYS = 1024``, ``STEP_MM = 0.25``,
``MAX_MM = 30.0``, ``SAMPLE_SPACING_MM = 1.0``) plus
``artifacts/learnable_affinity/dataset-generation/generate_ray_profiles.py``
(``int(num_rays * 1.5)`` Fibonacci oversample, also documented in
``dataset-generation/README.md`` v5.0.0 section).
"""

from __future__ import annotations

from typing import Final

from .geometry import RayGeometryConfig

# Patch size -- fixed by training architecture
# (``deepsets_bridge.py:217`` -> ``patch_size: int = 48``).
PATCH_SIZE: Final[int] = 48

# Ray-geometry constants, frozen to the v5.0.0 training-data manifest values.
# ``RAY_LENGTH = MAX_TRAVEL_MM / STEP_SIZE_MM = 30.0 / 0.25 = 120``.
STEP_SIZE_MM: Final[float] = 0.25
MAX_TRAVEL_MM: Final[float] = 30.0
RAY_LENGTH: Final[int] = 120
FIXED_POINT_SCALE: Final[int] = 1 << 16
FEATURE_CHANNELS: Final[tuple[str, ...]] = ("hu_normalized", "position")
FEATURE_COUNT: Final[int] = len(FEATURE_CHANNELS)
SHADER_CONSTANTS: Final[dict[str, int]] = {
    "PATCH_SIZE": PATCH_SIZE,
    "RAY_LENGTH": RAY_LENGTH,
    "FEATURE_COUNT": FEATURE_COUNT,
    "FIXED_POINT_SCALE": FIXED_POINT_SCALE,
    "WORKGROUP_SIZE": 64,
}
SAMPLE_SPACING_MM: Final[float] = 1.0
NUM_RAYS: Final[int] = 1024

# Tangent-weighted Fibonacci selection knobs. ``RAYS_OVERSAMPLE_FACTOR``
# matches the ``int(num_rays * 1.5)`` oversample in
# ``dataset-generation/generate_ray_profiles.py:603``.
# ``MIN_TANGENT_WEIGHT`` mirrors the production-default in
# ``SegmentationConfig.min_tangent_weight`` (0.1) -- this is the
# rejection floor on ``|sin(angle)|`` between ray direction and vessel
# tangent, used by ``weight_directions_by_tangent``.
RAYS_OVERSAMPLE_FACTOR: Final[float] = 1.5
MIN_TANGENT_WEIGHT: Final[float] = 0.1

# Binary output uses the mathematical sigmoid(logit) > 0.5 boundary directly.
LOGIT_THRESHOLD: Final[float] = 0.0

# ONNX I/O names -- verbatim from
# ``artifacts/learnable_affinity/experiments/multi_channel_3d_refinement/model_o_interval_inside/export_onnx.py``
# (lines 299-308 -> ``input_names`` / ``output_names`` of the
# ``torch.onnx.export`` call). Order matches that ``input_names`` list exactly.
#
# Per-tensor shapes (verified against ``_build_dummy_inputs`` in the same
# export script, lines 247-262, AND against
# ``deepsets_bridge.DeepSetsBridgeSegModel.forward`` at lines 250-258):
#
#   ray_features        (n_rays, L, 2)        float32   normalized HU, position
#   ray_mask            (n_rays, L)           bool/u32  True/nonzero for non-padding
#   origin_zyx_norm     (n_rays, 3)           float32   per-ray, batch.py:143
#   direction_zyx_norm  (n_rays, 3)           float32   per-ray, batch.py:144
#   voxel_indices       (n_rays, L)           int64/u32 flat into [0, P^3)
#   ray_hu              (n_rays, L)           float32   absolute HU at samples
#   ct_patch            (1, P, P, P)          float32   B=1 enforced
INPUT_NAMES: Final[tuple[str, ...]] = (
    "ray_features",
    "ray_mask",
    "origin_zyx_norm",
    "direction_zyx_norm",
    "voxel_indices",
    "ray_hu",
    "ct_patch",
)
OUTPUT_NAME: Final[str] = "mask_logits"  # (1, 1, P, P, P) float32


def model_o_geometry() -> RayGeometryConfig:
    """Construct the training-locked ray geometry for ModelO inference.

    The values originate in this module (the training-manifest source of
    truth). Slicer and serving adapters use this same factory; the geometry
    datatype remains independent of model execution.
    """
    return RayGeometryConfig(
        num_rays=int(NUM_RAYS),
        rays_oversample_factor=float(RAYS_OVERSAMPLE_FACTOR),
        min_tangent_weight=float(MIN_TANGENT_WEIGHT),
        sample_spacing_mm=float(SAMPLE_SPACING_MM),
        step_size_mm=float(STEP_SIZE_MM),
        max_travel_mm=float(MAX_TRAVEL_MM),
    )
