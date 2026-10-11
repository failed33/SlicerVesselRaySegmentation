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
    result = json.loads(str(await runtime.run(json.dumps(runtime_options))))
    result["application"] = application_summary()
    result["python_host"] = "slicerweb-pyodide-job-worker"
    return result
