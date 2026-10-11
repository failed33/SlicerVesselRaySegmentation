"""The browser must execute the compiled bytes delivered by the wheel."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from stage_browser import stage


class PackagedRuntimeTest(unittest.TestCase):
    def write_wheel(self, root: Path, name: str, data: bytes, digest: str) -> Path:
        wheel = root / "extension.whl"
        prefix = "slicer_home/share/Slicer-5.13/VesselRaySegmentation/wasm-runtime/"
        with ZipFile(wheel, "w") as archive:
            archive.writestr(prefix + "manifest.json", json.dumps({"files": {name: digest}}))
            archive.writestr(prefix + name, data)
        return wheel

    def test_stages_the_delivered_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"compiled wasm fixture"
            wheel = self.write_wheel(root, "runtime.wasm", data, hashlib.sha256(data).hexdigest())
            result = stage(wheel, root / "served")
            self.assertEqual((root / "served/runtime.wasm").read_bytes(), data)
            self.assertEqual(
                result["manifest"]["files"]["runtime.wasm"], hashlib.sha256(data).hexdigest()
            )

    def test_rejects_modified_runtime_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = self.write_wheel(
                root, "runtime.wasm", b"changed", hashlib.sha256(b"original").hexdigest()
            )
            with self.assertRaisesRegex(ValueError, "hash differs"):
                stage(wheel, root / "served")

    def test_rejects_asset_path_outside_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = self.write_wheel(
                root, "../escape.wasm", b"data", hashlib.sha256(b"data").hexdigest()
            )
            with self.assertRaisesRegex(ValueError, "Invalid runtime asset path"):
                stage(wheel, root / "served")


if __name__ == "__main__":
    unittest.main()
