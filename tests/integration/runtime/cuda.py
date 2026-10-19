"""Accept ModelO CUDA sampling and inference on real CUDA or MIG hardware."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
from cuda_ort import initialize_cuda
from cuda_ort.host import CudaHost
from model_o import attach_model, generate
from model_o_inference.cuda_inputs import CudaPatchInputs
from model_o_inference.cuda_runtime import CudaPatchInference
from model_o_inference.inputs import PackedRayCase
from model_o_inference.stitching import mask_from_logits


def run(directory: Path) -> dict:
    initialize_cuda()
    manifest = json.loads((directory / "model_o.json").read_text())
    geometry = np.fromfile(directory / manifest["geometry"], np.uint32).reshape(-1, 7)
    results = []
    with ExitStack() as resources:
        host = CudaHost()
        resources.callback(host.close)
        volume = np.empty(manifest["volume_shape"], np.float32)
        device_volume = host.empty(tuple(volume.shape), volume.dtype)
        first = manifest["cases"][0]
        case = PackedRayCase(volume, (1, 1, 1), geometry, np.array(first["center"]), first["h0"])
        builder = CudaPatchInputs(host, case, device_volume)
        resources.callback(builder.close)
        pipeline = CudaPatchInference(host, directory / "selected.onnx", case, device_volume)
        resources.callback(pipeline.close)
        for spec in manifest["cases"]:
            volume[:] = np.fromfile(directory / spec["volume"], np.float32).reshape(volume.shape)
            host.upload(device_volume, volume)
            case = PackedRayCase(volume, (1, 1, 1), geometry, np.array(spec["center"]), spec["h0"])
            config = builder.generate(case)
            host.synchronize()
            errors = {}
            for output in spec["outputs"]:
                actual = builder.gpu_inputs[output["name"]].numpy()
                expected = np.fromfile(
                    directory / output["expected"], np.dtype(output["dtype"])
                ).reshape(output["shape"])
                if actual.dtype.kind in "biu":
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(
                        actual, expected, atol=output["atol"], rtol=output["rtol"]
                    )
                errors[output["name"]] = float(
                    np.abs(actual.astype(np.float64) - expected.astype(np.float64)).max()
                )
            with ExitStack() as pending_resources:
                pending = pipeline.submit(case, pending_resources)
                mask = pipeline.collect(pending)
                logits = pending.logits.numpy().reshape(-1)
            reference = np.fromfile(directory / spec["model_logits"], np.float32)
            np.testing.assert_allclose(
                logits, reference, atol=manifest["model"]["atol"], rtol=manifest["model"]["rtol"]
            )
            np.testing.assert_array_equal(mask.reshape(-1), mask_from_logits(reference))
            results.append(
                {
                    "name": spec["name"],
                    "inputs_max_absolute_error": errors,
                    "logits_max_absolute_error": float(np.abs(logits - reference).max()),
                    "mask_disagreement_voxels": 0,
                }
            )
    return {
        "status": "passed",
        "backend": "cuda",
        "provider": "CUDAExecutionProvider",
        "cases": results,
        "model_sha256": manifest["model"]["sha256"],
        "reference": "independent float64 sampling and prepared CPU ONNX logits",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)
    attach_model(args.output, args.model)
    report = run(args.output)
    (args.output / "cuda-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
