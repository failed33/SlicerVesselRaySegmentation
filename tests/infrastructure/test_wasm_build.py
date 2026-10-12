"""Check WASM build identity and packaging failures without downloads or compilers."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.packaging.build_wasm import (
    INFERENCE,
    RUNTIME_FILES,
    build_command,
    ensure_commonjs_boundary,
    ensure_typescript_boundary,
    package,
    patch_paths,
    read_pins,
    source_identity,
    validate_output,
)


class WasmBuildTest(unittest.TestCase):
    def test_typescript_boundary_preserves_options_and_rejects_unexpected_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            path = source / "js/tsconfig.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"compilerOptions": {"strict": True}, "include": ["lib"]}))
            ensure_typescript_boundary(source)
            expected = {
                "compilerOptions": {"strict": True, "typeRoots": ["./node_modules/@types"]},
                "include": ["lib"],
            }
            self.assertEqual(json.loads(path.read_text()), expected)
            ensure_typescript_boundary(source)
            expected["compilerOptions"]["typeRoots"] = ["../unrelated/@types"]
            path.write_text(json.dumps(expected))
            with self.assertRaisesRegex(ValueError, "Unexpected ORT base typeRoots"):
                ensure_typescript_boundary(source)

    @unittest.skipUnless(shutil.which("node"), "Node is required for the upstream generator probe")
    def test_upstream_generators_do_not_inherit_parent_esm_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / "package.json").write_text('{"type":"module"}\n')
            source = parent / "source"
            source.mkdir()
            generator = source / "generator.js"
            generator.write_text('console.log(typeof require("node:os").platform);\n')
            inherited = subprocess.run(["node", str(generator)], capture_output=True)
            self.assertNotEqual(inherited.returncode, 0)
            ensure_commonjs_boundary(source)
            self.assertEqual(
                subprocess.check_output(["node", str(generator)], text=True).strip(), "function"
            )
            ensure_commonjs_boundary(source)
            (source / "package.json").write_text('{"type":"module"}\n')
            with self.assertRaisesRegex(ValueError, "Unexpected ORT root"):
                ensure_commonjs_boundary(source)

    def test_source_identity_uses_canonical_pins_and_patch_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inference = Path(temporary)
            (inference / "native").mkdir()
            (inference / "native/pins.cmake").write_text(
                "set(ORT_COMMIT " + "1" * 40 + ")\nset(DAWN_COMMIT " + "2" * 40 + ")\n"
            )
            (inference / "wasm").mkdir()
            (inference / "wasm/toolchain.json").write_text(
                json.dumps({"emscripten_version": "4.0.23", "node_major": 24})
            )
            for path in patch_paths(inference):
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text("first patch content")
            first = source_identity(inference)
            patch_paths(inference)[0].write_text("changed patch content")
            second = source_identity(inference)
            self.assertEqual(first["ORT_COMMIT"], "1" * 40)
            self.assertEqual(first["DAWN_COMMIT"], "2" * 40)
            self.assertNotEqual(first["patches"], second["patches"])

    def test_missing_or_ambiguous_canonical_pin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "native").mkdir()
            path = root / "native/pins.cmake"
            for content in ("", f"set(ORT_COMMIT {'a' * 40})\n" * 2):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_pins(root)

    def test_output_cannot_overwrite_repository_or_escape_build_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                validate_output(root / "build/runtime", root), (root / "build/runtime").resolve()
            )
            for output in (root, root / "build", root.parent / "elsewhere"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    validate_output(output, root)
            (root / "build").mkdir()
            (root / "build/escape").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output(root / "build/escape/runtime", root)

    def test_cpp_provider_build_uses_exact_dawn_override(self) -> None:
        command = build_command(Path("/source"), Path("/output"), 2, "4.0.23")
        self.assertIn("--use_webgpu", command)
        self.assertNotIn("--use_jsep", command)
        self.assertNotIn("--enable_wasm_jspi", command)
        self.assertIn("FETCHCONTENT_SOURCE_DIR_DAWN=/output/sources/dawn", command)
        self.assertEqual(command[command.index("--emsdk_version") + 1], "4.0.23")

    def test_missing_artifact_and_invalid_wasm_fail_before_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources/ort"
            dist = source / "js/web/dist"
            dist.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Missing compiled runtime"):
                package(source, root, {})
            for name in RUNTIME_FILES:
                (dist / name).write_bytes(b"not a WASM file")
            with self.assertRaisesRegex(ValueError, "not a WebAssembly"):
                package(source, root, {})
            self.assertFalse((root / "dist").exists())

    def test_esm_hunk_applies_and_rejects_changed_upstream_context(self) -> None:
        patch = INFERENCE / "wasm/ort-webgpu-esm.patch"
        excerpt = """      format: 'iife',
    });
  }

  if (BUNDLE_MODE === 'prod') {
    // ort.all[.min].[m]js
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "js/web/script/build.ts"
            source.parent.mkdir(parents=True)
            source.write_text(excerpt)
            command = ["git", "apply", "--include=js/web/script/build.ts", "--check", str(patch)]
            subprocess.run(command, cwd=root, check=True)
            source.write_text(excerpt.replace("BUNDLE_MODE", "RENAMED_OPTION"))
            rejected = subprocess.run(command, cwd=root, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
