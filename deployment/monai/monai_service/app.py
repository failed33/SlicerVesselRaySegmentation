"""Compose the released MONAI Label server with registered inference services."""

from pathlib import Path

from fastapi import FastAPI
from monailabel.config import settings
from monailabel.interfaces.utils.app import clear_cache

from monai_service.config import ServiceSettings
from monai_service.native_api import configure_routes


def create_app(service_settings: ServiceSettings) -> FastAPI:
    """Configure one released Label server per process, as upstream expects."""
    settings.MONAI_LABEL_APP_DIR = str(Path(__file__).resolve().parents[1] / "label_app")
    if not settings.MONAI_LABEL_STUDIES:
        settings.MONAI_LABEL_STUDIES = str(service_settings.work_dir / "studies")
    settings.MONAI_LABEL_APP_CONF = {"service_settings": service_settings}
    settings.MONAI_LABEL_AUTO_UPDATE_SCORING = False
    settings.MONAI_LABEL_TASKS_TRAIN = False
    settings.MONAI_LABEL_TASKS_STRATEGY = False
    settings.MONAI_LABEL_TASKS_SCORING = False
    settings.MONAI_LABEL_TASKS_BATCH_INFER = False
    clear_cache()

    from monailabel.app import app

    configure_routes(app)
    return app
