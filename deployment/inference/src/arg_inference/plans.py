"""Validate the inspected ARG preprocessing plans without inference dependencies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Plan:
    spacing: tuple[float, ...]
    patch: tuple[int, ...]
    transpose: tuple[int, ...]
    inverse: tuple[int, ...]
    sha256: str


def load_plan(path: Path) -> Plan:
    raw = json.loads(path.read_text())
    try:
        return _parse_plan(raw, path)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Malformed ARG preprocessing plans") from error


def _parse_plan(raw: dict, path: Path) -> Plan:
    cfg = raw["configurations"]["3d_fullres"]
    supported = (
        cfg["preprocessor_name"] == "DefaultPreprocessor"
        and cfg["normalization_schemes"] == ["ZScoreNormalization"]
        and cfg["use_mask_for_norm"] == [False]
        and raw["image_reader_writer"] == "SimpleITKIO"
    )
    for kind in ["data", "probabilities"]:
        supported &= cfg[f"resampling_fn_{kind}"] == "resample_data_or_seg_to_shape"
        supported &= cfg[f"resampling_fn_{kind}_kwargs"] == {
            "is_seg": False,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
        }
    if not supported:
        raise ValueError("Plans differ from the inspected ARG preprocessing contract")
    _validate_dimensions(cfg["spacing"], cfg["patch_size"])
    _validate_transposes(raw["transpose_forward"], raw["transpose_backward"])
    return Plan(
        tuple(cfg["spacing"]),
        tuple(cfg["patch_size"]),
        tuple(raw["transpose_forward"]),
        tuple(raw["transpose_backward"]),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _validate_dimensions(spacing: list[float], patch: list[int]) -> None:
    invalid_spacing = len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in spacing
    )
    invalid_patch = len(patch) != 3 or any(type(value) is not int or value <= 0 for value in patch)
    if invalid_spacing or invalid_patch:
        raise ValueError("ARG plans require positive 3D spacing/patch")


def _validate_transposes(forward: list[int], backward: list[int]) -> None:
    if sorted(forward) != [0, 1, 2] or sorted(backward) != [0, 1, 2]:
        raise ValueError("ARG plans require 3D transpose permutations")
    if any(backward[forward[index]] != index for index in range(3)):
        raise ValueError("ARG plans require inverse transposes")
