"""Native Label request fields and public Pydantic errors."""

from collections.abc import Mapping
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core import ErrorType

_PUBLIC_ERROR_MESSAGES = frozenset(get_args(ErrorType)) - {
    "value_error",
    "assertion_error",
    "union_tag_invalid",
}
_LABEL_REQUEST_FIELDS = frozenset(
    {
        "model",
        "image",
        "session",
        "description",
        "save_label",
        "label_tag",
        "label",
        "logging",
        "timeout",
    }
)


def parameter_errors(error: ValidationError) -> list[dict[str, object]]:
    """Report schema locations without submitted values or validator context."""
    errors = error.errors(include_input=False, include_context=False, include_url=False)
    for detail in errors:
        detail["loc"] = ("body", "params", *detail["loc"])
        # Custom validators and tagged unions may embed the submitted value in
        # their message even when Pydantic's input/context fields are omitted.
        if detail["type"] not in _PUBLIC_ERROR_MESSAGES:
            detail["msg"] = "Invalid parameter value"
    return [dict(detail) for detail in errors]


def request_parameters(request: Mapping[str, Any]) -> dict[str, Any]:
    """Remove native Label fields before validating application parameters."""
    return {key: value for key, value in request.items() if key not in _LABEL_REQUEST_FIELDS}


class NoParameters(BaseModel):
    """An inference request with no caller-selectable parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")
