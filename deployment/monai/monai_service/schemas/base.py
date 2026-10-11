"""Immutable request schemas shared by serving and worker entry points."""

from label_inference.requests import NoParameters
from pydantic import BaseModel, ConfigDict

__all__ = ["FrozenParameters", "NoParameters"]


class FrozenParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
