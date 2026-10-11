"""Keep application and model dependencies outside the distributable core."""

import ast
import sys
from pathlib import Path


def test_core_imports_only_standard_library_and_runtime_dependencies():
    root = Path(__file__).resolve().parents[1] / "src/dawn_ort"
    allowed = sys.stdlib_module_names | {"numpy", "onnxruntime"}
    violations = []
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module is not None
                names = [node.module]
            else:
                continue
            violations.extend(
                (path.name, name) for name in names if name.split(".")[0] not in allowed
            )
    assert not violations, violations


def test_model_runtime_has_no_source_checkout_or_serving_dependencies():
    source = Path(__file__).resolve().parents[1] / "src"
    forbidden = {"VesselPathTracerLib", "VesselBranchSplitterLib", "monai_service", "examples"}
    violations = []
    for package in ("arg_inference", "model_o_inference", "model_artifacts", "label_inference"):
        for path in (source / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    assert node.module is not None
                    names = [node.module]
                else:
                    continue
                violations.extend(
                    (str(path.relative_to(source)), name)
                    for name in names
                    if name.split(".")[0] in forbidden
                )
    assert not violations, violations
