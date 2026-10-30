"""The installed service entry point owns paths and process-worker configuration."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import uvicorn

from monai_service.serve import logging_config, main, service_environment


def test_service_uses_deployment_root_and_its_own_python(tmp_path: Path) -> None:
    environment = service_environment(tmp_path)
    assert environment["MONAI_SERVICE_ARG_MODEL_DIR"] == str(tmp_path / "models/arg")
    assert environment["MONAI_SERVICE_MODEL_O_MODEL_DIR"] == str(tmp_path / "models/model_o")
    assert environment["MONAI_SERVICE_WORK_DIR"] == str(tmp_path / "runtime")
    assert environment["MONAI_SERVICE_MODEL_O_BACKEND"] == "cuda"
    assert json.loads(environment["MONAI_SERVICE_MODEL_O_WORKER__COMMAND"]) == [
        sys.executable,
        "-m",
        "monai_service.models.model_o",
    ]


def test_entry_point_configures_service_before_starting_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(sys, "argv", ["slicerray-serve", "--root", str(tmp_path), "--port", "9000"])
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    main()
    run.assert_called_once_with(
        "asgi:app", host="127.0.0.1", port=9000, log_config=logging_config()
    )
    assert os.environ["MONAI_SERVICE_ARG_MODEL_DIR"] == str(tmp_path / "models/arg")
