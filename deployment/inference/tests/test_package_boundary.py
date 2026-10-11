"""Keep shared numerical code separate from native and serving dependencies."""

import ast
import subprocess
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "src"
_NUMERICAL_MODULES = (
    "geometry",
    "constants",
    "lattice",
    "sampling",
    "preparation",
    "planning",
    "stitching",
    "shader_contract",
    "shaders",
)


def _absolute_imports(path: Path) -> list[str]:
    imports = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            assert node.module is not None
            imports.append(node.module)
    return imports


def test_core_imports_only_standard_library_and_runtime_dependencies():
    allowed = sys.stdlib_module_names | {"numpy", "onnxruntime"}
    violations = [
        (path.name, name)
        for path in (_SOURCE / "dawn_ort").glob("*.py")
        for name in _absolute_imports(path)
        if name.split(".")[0] not in allowed
    ]
    assert not violations, violations


def test_numerical_modules_import_only_standard_library_numpy_and_scipy():
    allowed = sys.stdlib_module_names | {"numpy", "scipy"}
    violations = [
        (module, name)
        for module in _NUMERICAL_MODULES
        for name in _absolute_imports(_SOURCE / "model_o_inference" / f"{module}.py")
        if name.split(".")[0] not in allowed
    ]
    assert not violations, violations


def test_metadata_imports_without_site_packages(tmp_path: Path):
    # -S removes all installed site packages; this executes the real modules.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            """
import sys
sys.path.insert(0, sys.argv[1])
import arg_inference
from arg_inference.plans import Plan, load_plan
from model_o_inference.constants import model_o_geometry
from model_o_inference.geometry import RayGeometryConfig
from model_o_inference.shaders import SHADER_PATH
assert model_o_geometry().num_rays > 0
assert SHADER_PATH.is_file()
assert not {'numpy', 'scipy', 'onnxruntime', 'SimpleITK', 'pydantic', 'mlflow', 'torch'} & sys.modules.keys()
""",
            str(_SOURCE),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_graph_preparation_does_not_import_native_runtime(tmp_path: Path):
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'onnxruntime', 'dawn_ort'}:
            raise AssertionError(f'Graph preparation imported {fullname}')
        return None
sys.meta_path.insert(0, RejectNative())
from model_o_inference.graph import prepare_model_o
assert callable(prepare_model_o)
""",
            str(_SOURCE),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_model_runtime_has_no_source_checkout_or_serving_dependencies():
    forbidden = {"VesselPathTracerLib", "VesselBranchSplitterLib", "monai_service", "examples"}
    violations = [
        (str(path.relative_to(_SOURCE)), name)
        for package in ("arg_inference", "model_o_inference", "model_artifacts", "label_inference")
        for path in (_SOURCE / package).rglob("*.py")
        for name in _absolute_imports(path)
        if name.split(".")[0] in forbidden
    ]
    assert not violations, violations
