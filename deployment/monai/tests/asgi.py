"""Explicit test composition for real HTTP and desktop Slicer integration."""

import os
import sys
from pathlib import Path

from monai_service.adapters.execution import DockerExecutor, LocalProcessExecutor
from monai_service.config import ServiceSettings
from monai_service.schemas.execution import ContainerConfig, ProcessConfig
from tests.fixtures import SMOKE_WORKER_PATH, create_test_app

work_dir = Path(os.environ["MONAI_SERVICE_WORK_DIR"])
timeout = float(os.environ.get("MONAI_TEST_WORKER_TIMEOUT_SECONDS", "30"))
image = os.environ.get("MONAI_TEST_WORKER_IMAGE")
executor = (
    DockerExecutor(
        ContainerConfig(
            mode="container",
            image=image,
            timeout_seconds=timeout,
            runtime=os.environ.get("MONAI_SERVICE_CONTAINER_RUNTIME", "docker"),
        )
    )
    if image
    else LocalProcessExecutor(
        ProcessConfig(
            mode="process",
            command=(sys.executable, str(SMOKE_WORKER_PATH)),
            timeout_seconds=timeout,
        )
    )
)
app = create_test_app(ServiceSettings(work_dir), executor)
