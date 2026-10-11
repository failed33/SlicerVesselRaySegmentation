"""Control-plane imports must not require a model inference environment."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCE = PROJECT.parent / "inference/src"


def _check_imports(body: str, blocked: tuple[str, ...], preload: str = "") -> None:
    script = f"""
import importlib.abc
import sys
{preload}
blocked = {blocked!r}

class BlockComputeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ModuleNotFoundError(f'Compute dependency is unavailable: {{fullname}}')
        return None

sys.meta_path.insert(0, BlockComputeImports())
{body}
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT,
        env={
            **os.environ,
            "ORT_DISABLE_TELEMETRY": "1",
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "MLFLOW_DISABLE_AGENT_HINT": "1",
            "PYTHONPATH": os.pathsep.join([str(PROJECT), str(INFERENCE_SOURCE)]),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_request_schema_and_file_helpers_load_without_compute_libraries() -> None:
    _check_imports(
        """
from label_inference.files import ResultFileError, retain_result
from label_inference.requests import request_parameters
from model_o_inference.schemas import ModelOParameters

request = {'model': 'model_o', 'image': '/ct.nrrd',
           'polylines_zyx': [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]}
parameters = ModelOParameters.model_validate(request_parameters(request))
assert parameters.polylines_zyx[0][1] == (4.0, 5.0, 6.0)
assert callable(retain_result)
assert issubclass(ResultFileError, RuntimeError)
""",
        ("numpy", "scipy", "torch", "onnx", "onnxruntime", "dawn_ort", "monai", "holoscan"),
    )


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow is required")
def test_registry_identity_loads_without_model_runtimes() -> None:
    _check_imports(
        """
from model_artifacts import ModelSource, SelectedModel

source = ModelSource('arg', 'champion', 'sqlite:///registry.db')
selected = SelectedModel(source.name, '7', source.tracking_uri, source.tracking_uri)
assert selected.artifact_uri == 'models:/arg/7'
assert len(selected.cache_key) == 64
assert 'arg_inference' not in sys.modules
assert 'model_o_inference.runtime' not in sys.modules
""",
        ("torch", "onnxruntime", "scipy", "dawn_ort", "monai.deploy", "holoscan"),
    )


@pytest.mark.skipif(
    importlib.util.find_spec("monailabel") is None, reason="Native MONAI Label is required"
)
def test_label_configuration_does_not_import_sdk_or_model_execution() -> None:
    # Native Label owns its Torch/MONAI dependencies. Initialize those first so
    # this check measures imports added by our control-plane implementation.
    _check_imports(
        """
import os
from vessel_deployment import from_env

os.environ.update({
    'MONAI_SERVICE_ENABLE_ARG': 'true',
    'MONAI_SERVICE_ENABLE_MODEL_O': 'false',
    'MONAI_SERVICE_MLFLOW_TRACKING_URI': 'sqlite:///registry.db',
    'MONAI_SERVICE_ARG_MODEL_NAME': 'arg',
    'MONAI_SERVICE_ARG_MODEL_ALIAS': 'champion',
    'MONAI_SERVICE_ARG_WORKER': '{\"mode\": \"direct\"}',
})
settings = from_env()
assert set(settings.registrations) == {'arg_vessel'}
for name in ('arg_inference', 'model_o_inference.runtime', 'dawn_ort', 'monai.deploy'):
    assert name not in sys.modules, name
""",
        (
            "arg_inference",
            "model_o_inference.runtime",
            "onnxruntime",
            "scipy",
            "dawn_ort",
            "monai.deploy",
            "holoscan",
        ),
        """
import monailabel.config
import monailabel.interfaces.tasks.infer_v2
import monailabel.interfaces.utils.app
import monailabel.app
""",
    )
