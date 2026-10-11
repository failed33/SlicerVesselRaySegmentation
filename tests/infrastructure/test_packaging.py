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
            self.assertEqual(
                files,
                {
                    "lib/Slicer-5.13/qt-scripted-modules/VesselRaySegmentationDemo.py",
                    "lib/Slicer-5.13/qt-scripted-modules/Resources/Icons/VesselRaySegmentationDemo.png",
                    "share/Slicer-5.13/VesselRaySegmentation/LICENSE.txt",
                },
            )
            source = (
                Path(__file__).resolve().parents[2]
                / "VesselRaySegmentationDemo/VesselRaySegmentationDemo.py"
            )
            installed = (
                output / "unpacked/lib/Slicer-5.13/qt-scripted-modules/VesselRaySegmentationDemo.py"
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
