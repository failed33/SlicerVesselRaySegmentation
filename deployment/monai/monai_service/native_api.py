"""Apply shared native Label route adaptations to this service."""

from fastapi import FastAPI
from label_inference.routes import configure_routes as configure_native_routes

from monai_service.streaming import register_streaming_route


def configure_routes(app: FastAPI) -> None:
    configure_native_routes(app)
    register_streaming_route(app)


__all__ = ["configure_routes"]
