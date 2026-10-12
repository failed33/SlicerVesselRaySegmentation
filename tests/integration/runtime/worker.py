"""Executed as Python by SlicerWeb jobs.run after application.py is loaded."""

from __future__ import annotations


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

    options = dict(runtime_options)
    options["model_o_shaders"] = {
        name: {"code": load_shader(name).code, "metadata": load_shader(name).metadata}
        for name in ("patch_inputs", "sigmoid")
    }
    result = json.loads(str(await runtime.run(json.dumps(options))))
    result["application"] = application_summary()
    result["python_host"] = "slicerweb-pyodide-job-worker"
    return result
