"""Check the actual CMake/CPack output without Slicer, credentials, or network."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.packaging.build_scripted import build_package


class ScriptedPackageTest(unittest.TestCase):
    def test_archive_contains_only_declared_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            archive = build_package(output, "5.13")
            with zipfile.ZipFile(archive) as package:
                files = {item.filename for item in package.infolist() if not item.is_dir()}
            module_root = Path(__file__).resolve().parents[2] / "VesselRaySegmentation"
            declared = {"VesselRaySegmentation.py", "Resources/Icons/VesselRaySegmentation.png"}
            declared.update(
                str(path.relative_to(module_root))
                for path in (module_root / "VesselRaySegmentationLib").rglob("*.py")
            )
            self.assertEqual(
                files,
                {
                    *("lib/Slicer-5.13/qt-scripted-modules/" + name for name in declared),
                    "share/Slicer-5.13/VesselRaySegmentation/LICENSE.txt",
                },
            )
            source = (
                Path(__file__).resolve().parents[2]
                / "VesselRaySegmentation/VesselRaySegmentation.py"
            )
            installed = (
                output / "unpacked/lib/Slicer-5.13/qt-scripted-modules/VesselRaySegmentation.py"
            )
            self.assertEqual(installed.read_bytes(), source.read_bytes())

    def test_rebuild_discards_obsolete_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            build_package(output, "5.13")
            roots = (
                output / "staged",
                output / "install/ext/VesselRaySegmentation",
                output / "unpacked",
            )
            for root in roots:
                (root / "retired.py").write_text("obsolete experiment")
            archive = build_package(output, "5.13")
            for root in roots:
                self.assertFalse((root / "retired.py").exists())
            with zipfile.ZipFile(archive) as package:
                self.assertNotIn("retired.py", package.namelist())

    def test_optional_runtime_payloads_preserve_package_and_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "inference.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("model_o_inference/__init__.py", '"""Shared Python."""\n')
            runtime = root / "runtime"
            runtime.mkdir()
            (runtime / "manifest.json").write_text('{"python":"3.12.12"}\n')
            executable = runtime / "python"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            build_package(root / "package", "5.13", inference_wheel=wheel, desktop_runtime=runtime)
            installed = root / "package/unpacked"
            self.assertEqual(
                (
                    installed / "lib/Slicer-5.13/qt-scripted-modules/model_o_inference/__init__.py"
                ).read_text(),
                '"""Shared Python."""\n',
            )
            delivered = installed / "share/Slicer-5.13/VesselRaySegmentation/desktop-runtime/python"
            self.assertEqual(delivered.read_bytes(), executable.read_bytes())
            self.assertTrue(delivered.stat().st_mode & 0o111)
            # Omitting an optional payload on rebuild must also remove it from CMake's cache.
            build_package(root / "package", "5.13")
            self.assertFalse(delivered.exists())
