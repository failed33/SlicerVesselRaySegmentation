"""Entry point executed by Slicer's --python-script option after module discovery."""

import json
import os
import sys
import traceback
from pathlib import Path

import slicer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_module import test_installed_module

report = Path(os.environ["TEST_REPORT"])
try:
    result = test_installed_module(os.environ["PACKAGE_ROOT"])
    report.write_text(json.dumps(result, indent=2) + "\n")
    slicer.util.exit(0)
except Exception:
    report.write_text(
        json.dumps({"status": "failed", "error": traceback.format_exc()}, indent=2) + "\n"
    )
    traceback.print_exc()
    slicer.util.exit(1)
