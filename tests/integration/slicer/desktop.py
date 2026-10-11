"""Entry point executed by Slicer's --python-script option after module discovery."""

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import slicer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module import test_installed_module

report = Path(os.environ["TEST_REPORT"])
try:
    result = test_installed_module(os.environ["PACKAGE_ROOT"])
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
        result["runtime"] = json.loads(runtime_report.read_text())
        result["runtime"]["python_host"] = "desktop-slicer-bundled-worker"
    report.write_text(json.dumps(result, indent=2) + "\n")
    slicer.util.exit(0)
except Exception:
    report.write_text(
        json.dumps({"status": "failed", "error": traceback.format_exc()}, indent=2) + "\n"
    )
    traceback.print_exc()
    slicer.util.exit(1)
