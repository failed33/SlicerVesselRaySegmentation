"""Entry point executed by Slicer's --python-script option after module discovery."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import slicer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module import test_hosted_workflow, test_installed_module

report = Path(os.environ["TEST_REPORT"])
try:
    from slicer.packaging import load_requirements, pip_install

    modules = list(Path(os.environ["PACKAGE_ROOT"]).glob("lib/Slicer-*/qt-scripted-modules"))
    requirements = modules[0] / "VesselRaySegmentationLib/Resources/requirements.txt"
    pip_install([str(req) for req in load_requirements(requirements)], show_progress=False)
    result = test_installed_module(os.environ["PACKAGE_ROOT"])
    endpoint_python = os.environ.get("VRS_TEST_ENDPOINT_PYTHON")
    if endpoint_python:
        from endpoint import start_endpoint

        with start_endpoint(
            report.parent / "endpoint", python=Path(endpoint_python), hold_request=4
        ) as endpoint:

            async def wait_for_event(kind):
                deadline = asyncio.get_running_loop().time() + 45
                while sum(event["kind"] == kind for event in endpoint.events()) < 4:
                    if asyncio.get_running_loop().time() > deadline:
                        raise TimeoutError("The final ModelO request did not enter inference")
                    await asyncio.sleep(0.01)

            async def wait_for_request():
                await wait_for_event("inference-start")

            async def release_request():
                endpoint.release_inference()
                await wait_for_event("inference")

            result["hosted_workflow"] = asyncio.run(
                test_hosted_workflow(
                    endpoint.endpoint, endpoint.token, wait_for_request, release_request
                )
            )
            events = endpoint.events()
            uploads = [event for event in events if event.get("method") == "PUT"]
            assert len(uploads) == 1, events
            result["hosted_workflow"]["uploads"] = len(uploads)
            starts = [event for event in events if event["kind"] == "inference-start"]
            finishes = [event for event in events if event["kind"] == "inference"]
            closed = result["hosted_workflow"]["scene_closed_at"]
            assert starts[-1]["timestamp"] < closed < finishes[-1]["timestamp"], events
    package_root = Path(os.environ["PACKAGE_ROOT"]).resolve()
    candidates = list(package_root.glob("share/Slicer-*/VesselRaySegmentation/desktop-runtime"))
    override = os.environ.get("VRS_DESKTOP_RUNTIME")
    if override or candidates:
        if len(candidates) > 1 and not override:
            raise AssertionError("Multiple packaged desktop runtimes")
        runtime = Path(override).resolve() if override else candidates[0]
        executable = runtime / (
            "python/python.exe" if sys.platform == "win32" else "python/bin/python3"
        )
        runtime_report = report.with_name(report.stem + "-runtime.json")
        command = [
            str(executable),
            "-E",
            str(Path(__file__).resolve().parents[1] / "runtime/native.py"),
            "--fixtures",
            os.environ["VRS_RUNTIME_FIXTURES"],
            "--native-assets",
            str(runtime / "assets"),
            "--report",
            str(runtime_report),
            "--package-root",
            str(runtime / "python"),
        ]
        if os.environ.get("VRS_RUNTIME_ALLOW_SOFTWARE") == "1":
            command.append("--allow-software")
        if os.environ.get("VRS_RUNTIME_REQUIRE_F16") == "1":
            command.append("--require-f16")
        subprocess.run(command, check=True, timeout=300)
        runtime_result = json.loads(runtime_report.read_text())
        runtime_result["python_host"] = "desktop-slicer-bundled-worker"
        result["runtime"] = runtime_result
    screenshot = os.environ.get("VRS_GUI_SCREENSHOT")
    if screenshot:
        window = slicer.util.mainWindow()
        if window is None:
            raise RuntimeError("Screenshot acceptance requires Slicer's main window")
        slicer.util.selectModule("VesselRaySegmentation")
        window.resize(1400, 1100)
        slicer.app.processEvents()
        if not window.grab().save(screenshot):
            raise RuntimeError("Could not save the Slicer workflow screenshot")
    report.write_text(json.dumps(result, indent=2) + "\n")
    slicer.util.exit(0)
except Exception:
    report.write_text(
        json.dumps({"status": "failed", "error": traceback.format_exc()}, indent=2) + "\n"
    )
    traceback.print_exc()
    slicer.util.exit(1)
