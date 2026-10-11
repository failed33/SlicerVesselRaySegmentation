"""The explicit task catalogue hosted by this Label process."""

from pathlib import Path

from monai_service.config import ServiceSettings
from monai_service.models.arg import TASK as ARG
from monai_service.models.model_o import TASK as MODEL_O
from monai_service.registry import from_env as configure_registry

TASKS = (ARG, MODEL_O)


def from_env() -> ServiceSettings:
    return configure_registry(TASKS, default_work_dir=Path(__file__).resolve().parent / ".runtime")
