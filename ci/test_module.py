"""Run the installed module's identical tests in desktop Slicer and SlicerWeb."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import slicer


def test_installed_module(package_root: str) -> dict[str, object]:
    import VesselRaySegmentationDemo as module

    loaded = Path(module.__file__).resolve()
    if not loaded.is_relative_to(Path(package_root).resolve()):
        raise AssertionError(f"Module loaded outside the installed package: {loaded}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.VesselRaySegmentationDemoTest)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    print(log.getvalue(), flush=True)
    if result.testsRun != 2 or not result.wasSuccessful():
        raise AssertionError(log.getvalue())
    return {
        "status": "passed",
        "tests_run": result.testsRun,
        "module_path": str(loaded),
        "slicer_version": str(slicer.app.applicationVersion),
    }
