"""Narrow adaptations of released Label routes, retaining native schemas and RBAC."""

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.routing import APIRoute, iter_route_contexts
from monailabel.endpoints.infer import api_run_inference, run_inference
from monailabel.endpoints.infer import router as inference_router
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.utils.app import app_instance
from pydantic import Field, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

_PARAMETERS = TypeAdapter(dict[str, object])


class _RouteSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONAI_LABEL_", frozen=True, extra="forbid")

    infer_max_part_size: int = Field(default=16 * 1024 * 1024, gt=0)


class _CleanupResponse(Response):
    """Run native file cleanup after every response transmission outcome."""

    def __init__(self, response: Response) -> None:
        super().__init__(status_code=response.status_code, background=response.background)
        self.raw_headers = response.raw_headers
        self.native_response = response
        response.background = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self.native_response(scope, receive, send)
        finally:
            cleanup, self.background = self.background, None
            if cleanup is not None:
                with anyio.CancelScope(shield=True):
                    await cleanup()


class InferenceFormRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        native_handler = super().get_route_handler()

        async def parse_form(request: Request) -> Response:
            # Native FastAPI parsing reuses this form. Files retain Starlette's
            # normal streaming/spooling and close on every handler outcome.
            async with request.form(max_part_size=request.app.state.label_infer_max_part_size):
                response = await native_handler(request)
            # The preview response already owns its producer and file cleanup.
            return _CleanupResponse(response) if self.endpoint is api_run_inference else response

        return parse_form


def _configure_inference_form(app: FastAPI) -> None:
    app.state.label_infer_max_part_size = _RouteSettings().infer_max_part_size
    for route in tuple(inference_router.routes):
        if (
            not isinstance(route, APIRoute)
            or route.endpoint is not api_run_inference
            or isinstance(route, InferenceFormRoute)
        ):
            continue
        # Re-register the native endpoint through the same router and defaults.
        # add_api_route also refreshes FastAPI's lazy inclusion contexts.
        inference_router.routes.remove(route)
        inference_router.add_api_route(
            route.path.removeprefix(inference_router.prefix),
            route.endpoint,
            methods=list(route.methods or ()),
            name=route.name,
            summary=route.summary,
            route_class_override=InferenceFormRoute,
        )


async def _not_implemented(**values: Any) -> None:
    raise HTTPException(status_code=501, detail="Not yet implemented")


def validate_parameters(raw_params: str) -> None:
    """Reject malformed params and identity overrides before native merging."""
    try:
        params = _PARAMETERS.validate_json(raw_params or "{}")
    except ValidationError as error:
        if error.errors(include_input=False)[0]["type"] == "json_invalid":
            raise HTTPException(status_code=400, detail="params must be valid JSON") from error
        raise HTTPException(status_code=422, detail="params must be a JSON object") from error
    if params.keys() & {"model", "image", "session", "session_id"}:
        raise HTTPException(status_code=422, detail="params cannot override request identity")


def session_image(instance: MONAILabelApp, session_id: str) -> Path:
    """Resolve a native session without taking ownership of its image file."""
    sessions = instance.sessions()
    session = sessions.get_session(session_id) if sessions is not None else None
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return Path(session.image)


def _validate_datastore_image(instance: MONAILabelApp, image_id: str) -> None:
    # Use the datastore's indexed lookup, retaining opaque and hierarchical
    # IDs. Native infer otherwise accepts any existing server filesystem path.
    if not instance.datastore().get_image_uri(image_id):
        raise HTTPException(status_code=404, detail="Datastore image not found")
    if Path(image_id).exists():
        raise HTTPException(
            status_code=422, detail="Image identifier collides with a server filesystem path"
        )


def _validate_input(instance: MONAILabelApp, values: dict[str, Any]) -> None:
    """Follow native source precedence: session, upload, then datastore image."""
    if session_id := values.get("session_id"):
        session_image(instance, session_id)
        return
    if values.get("file"):
        return
    if image_id := values.get("image"):
        _validate_datastore_image(instance, image_id)
        return
    raise HTTPException(status_code=422, detail="Provide an image, file or session_id")


def _run_inference(values: dict[str, Any]) -> Any:
    """Validate the request, then retain upstream input and response handling."""
    values.pop("user", None)
    validate_parameters(values.get("params") or "{}")
    instance = app_instance()
    if values["model"] not in instance.info()["models"]:
        raise HTTPException(status_code=404, detail=f"Unknown inference model: {values['model']}")
    _validate_input(instance, values)
    if values.get("output") == "dicom_seg":
        raise HTTPException(status_code=501, detail="Not yet implemented")
    return run_inference(**values)


async def _offloaded_inference(**values: Any) -> Any:
    try:
        return await run_in_threadpool(_run_inference, values)
    except Exception:
        # Upstream registers uploaded files for response cleanup. An error has no
        # response background tasks, so perform that same cleanup before raising.
        with anyio.CancelScope(shield=True):
            await values["background_tasks"]()
        raise


def configure_routes(app: FastAPI) -> None:
    """Adapt route calls while retaining upstream schemas and dependency checks."""
    _configure_inference_form(app)
    deferred_modules = {
        "monailabel.endpoints.activelearning",
        "monailabel.endpoints.batch_infer",
        "monailabel.endpoints.model",
        "monailabel.endpoints.scoring",
        "monailabel.endpoints.train",
        "monailabel.endpoints.wsi_infer",
    }
    # FastAPI includes routers lazily. Inspect its public effective contexts so
    # the dependency call belongs to the handler that serves the request.
    for route in iter_route_contexts(app.routes):
        if not isinstance(route.original_route, APIRoute):
            continue
        if route.endpoint.__module__ in deferred_modules:
            # FastAPI resolves the original RBAC dependencies before calling this.
            route.dependant.call = _not_implemented
        elif route.endpoint.__module__ == "monailabel.endpoints.infer":
            route.dependant.call = _offloaded_inference
