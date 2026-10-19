"""Start a local CUDA inference service from its deployment directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def service_environment(root: Path) -> dict[str, str]:
    environment = {
        "MONAI_SERVICE_ENABLE_ARG": "true",
        "MONAI_SERVICE_ENABLE_MODEL_O": "true",
        "MONAI_SERVICE_ARG_MODEL_DIR": str(root / "models/arg"),
        "MONAI_SERVICE_MODEL_O_MODEL_DIR": str(root / "models/model_o"),
        "MONAI_SERVICE_ARG_DEVICE": "gpu",
        "MONAI_SERVICE_MODEL_O_BACKEND": "cuda",
        "MONAI_SERVICE_WORK_DIR": str(root / "runtime"),
        "ORT_DISABLE_TELEMETRY": "1",
        "MONAI_LABEL_AUTH_ENABLE": "false",
        "MONAI_LABEL_DATASTORE_AUTO_RELOAD": "false",
        "MONAI_LABEL_SESSION_EXPIRY": "3600",
    }
    for task, module in (("ARG", "arg"), ("MODEL_O", "model_o")):
        environment[f"MONAI_SERVICE_{task}_WORKER__MODE"] = "process"
        environment[f"MONAI_SERVICE_{task}_WORKER__COMMAND"] = json.dumps(
            [sys.executable, "-m", f"monai_service.models.{module}"]
        )
        environment[f"MONAI_SERVICE_{task}_WORKER__TIMEOUT_SECONDS"] = "1200"
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Deployment directory")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.update(service_environment(args.root))

    import uvicorn

    uvicorn.run("asgi:app", host="127.0.0.1", port=args.port)
