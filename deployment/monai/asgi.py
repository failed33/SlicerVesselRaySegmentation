"""Run with uvicorn asgi:app from deployment/monai."""

from monai_service.app import create_app
from vessel_deployment import from_env

app = create_app(from_env())
