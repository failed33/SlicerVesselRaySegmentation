"""Create an input-dependent synthetic ARG artifact, never a trained-model proof."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import SimpleITK as sitk
from numpy.typing import NDArray
from onnx import TensorProto, helper


@dataclass(frozen=True)
class ArgCase:
    input_dir: Path
    model_dir: Path
    expected: NDArray[np.uint8]

    @property
    def image(self) -> Path:
        return self.input_dir / "ct.nii.gz"


def _write_plans(model_dir: Path) -> None:
    resampling = {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None}
    plans = {
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {
            "3d_fullres": {
                "preprocessor_name": "DefaultPreprocessor",
                "patch_size": [8, 8, 8],
                "spacing": [1.4, 1.1, 0.8],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": resampling,
                "resampling_fn_probabilities_kwargs": resampling,
            }
        },
    }
    (model_dir / "plans.json").write_text(json.dumps(plans))
    (model_dir / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "vessel": 1}})
    )


def _write_graph(model_dir: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["image"], ["half_image"], to=TensorProto.FLOAT16),
            helper.make_node("Neg", ["half_image"], ["background"]),
            helper.make_node("Concat", ["background", "half_image"], ["half_logits"], axis=1),
            helper.make_node("Cast", ["half_logits"], ["logits"], to=TensorProto.FLOAT),
        ],
        "synthetic_input_dependent_arg",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 8, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, 8, 8, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    metadata = {
        "preprocessing": "native-plans",
        "configuration": "3d_fullres",
        "patch_size": [8, 8, 8],
        "precision": "mixed-fp16",
        "plans_sha256": hashlib.sha256((model_dir / "plans.json").read_bytes()).hexdigest(),
    }
    helper.set_model_props(model, {key: json.dumps(value) for key, value in metadata.items()})
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, model_dir / "model.onnx")


def make_arg_case(root: Path) -> ArgCase:
    input_dir, model_dir = root / "input", root / "model"
    input_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    _write_plans(model_dir)
    _write_graph(model_dir)
    voxels = np.zeros((16, 18, 20), dtype=np.float32)
    crop = np.arange(1, 12 * 14 * 16 + 1, dtype=np.float32).reshape(12, 14, 16)
    # Keep the two classes away from FP16 weighted-accumulation ties at crop edges.
    crop[crop > crop.mean()] += 10_000
    voxels[2:14, 2:16, 2:18] = crop
    image = sitk.GetImageFromArray(voxels)
    angle = math.pi / 12
    image.SetDirection(
        (-math.cos(angle), math.sin(angle), 0, -math.sin(angle), -math.cos(angle), 0, 0, 0, 1)
    )
    image.SetSpacing((0.8, 1.1, 1.4))
    image.SetOrigin((17.3, -8.6, 5.2))
    sitk.WriteImage(image, str(input_dir / "ct.nii.gz"))
    return ArgCase(input_dir, model_dir, (voxels > crop.mean()).astype(np.uint8))
