"""Native MONAI Label app discovered through the standard main.py loader."""

import logging
from typing import Any, cast

from label_inference.sessions import SynchronizedSessions
from monailabel.config import settings
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask

from monai_service.application import InferenceService
from monai_service.config import ServiceSettings
from monai_service.tasks import LabelInferTask

logger = logging.getLogger(__name__)


class RegisteredInferenceApp(MONAILabelApp):
    def __init__(self, app_dir: str, studies: str, conf: dict[str, Any]) -> None:
        if "service_settings" not in conf:
            from vessel_deployment import from_env

            conf = {**conf, "service_settings": from_env()}
        from monailabel.app import app

        from monai_service.native_api import configure_routes

        configure_routes(app)
        service_settings = cast(ServiceSettings, conf["service_settings"])
        if not settings.MONAI_LABEL_SESSION_PATH:
            settings.MONAI_LABEL_SESSION_PATH = str(service_settings.work_dir / "sessions")
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="Registered inference",
            description="MONAI Label inference through configured workflow deployments.",
            version="0.1.0",
        )

    def _load_sessions(self, load: bool = False) -> SynchronizedSessions | None:
        if not load:
            return None
        return SynchronizedSessions(
            settings.MONAI_LABEL_SESSION_PATH, settings.MONAI_LABEL_SESSION_EXPIRY
        )

    def cleanup_sessions(self) -> None:
        sessions = self.sessions()
        if sessions is not None:
            count = sessions.remove_expired()
            logger.debug("Total sessions cleaned up: %s", count)

    def init_infers(self) -> dict[str, InferTask]:
        service_settings = cast(ServiceSettings, self.conf["service_settings"])
        return {
            identifier: LabelInferTask(
                InferenceService(deployment, service_settings.work_dir / "requests")
            )
            for identifier, deployment in service_settings.registrations.items()
        }

    def init_strategies(self) -> dict[str, Any]:
        return {}

    # Label annotates its factory as Callable, while this app disables it with None.
    def init_batch_infer(self) -> Any:
        return None
