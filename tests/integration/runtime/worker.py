"""Executed as Python by SlicerWeb jobs.run after application.py is loaded."""

from __future__ import annotations

from typing import Any


async def run_runtime(
    runtime_adapter_url: str, runtime_options: dict[str, bool]
) -> dict[str, object]:
    import json
    import os

    import js
    from application import application_summary

    if os.environ.get("SLICERWEB_JOB") != "1" or hasattr(js, "document"):
        raise AssertionError("Runtime acceptance must execute inside SlicerWeb's job worker")
    import_runtime = js.Function("url", "return import(url)")
    runtime = await import_runtime(runtime_adapter_url)
    from model_o_inference.shaders import load_shader

    options: dict[str, Any] = dict(runtime_options)
    options["model_o_shaders"] = {
        name: {"code": load_shader(name).code, "metadata": load_shader(name).metadata}
        for name in ("patch_inputs",)
    }
    result = json.loads(str(await runtime.run(json.dumps(options))))
    import numpy as np
    from model_o_inference.stitching import mask_from_logits

    fixture = result["model_o_shaders"].pop("threshold_fixture")
    np.testing.assert_array_equal(
        mask_from_logits(np.array(fixture["logits"], np.float32)),
        np.array(fixture["mask"], bool),
    )
    if result["model_o_shaders"]["trained_model"] is not None:
        trained = result["model_o_shaders"]["trained_model"]
        fixture = trained.pop("threshold_fixture")
        np.testing.assert_array_equal(
            mask_from_logits(np.array(fixture["logits"], np.float32)),
            np.array(fixture["mask"], bool),
        )
        trained["python_binary_mask"] = "passed"
    for value in (np.nan, np.inf, -np.inf):
        try:
            mask_from_logits(np.array([value], np.float32))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Python accepted nonfinite ModelO logits")
    result["model_o_shaders"]["python_logit_threshold_and_nonfinite_rejection"] = "passed"
    result["application"] = application_summary()
    result["python_host"] = "slicerweb-pyodide-job-worker"
    return result
