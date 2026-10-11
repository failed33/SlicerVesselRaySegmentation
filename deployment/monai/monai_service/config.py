"""Model-independent settings for the native Label adapter."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from monai_service.contracts import ModelDeployment


@dataclass(frozen=True)
class ServiceSettings:
    work_dir: Path
    registrations: Mapping[str, ModelDeployment] = field(default_factory=dict)
