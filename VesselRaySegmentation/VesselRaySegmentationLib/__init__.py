"""Slicer workflow and independently importable MONAI transport."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logic import VesselRaySegmentationLogic

__all__ = ["VesselRaySegmentationLogic"]


def __getattr__(name):
    if name == "VesselRaySegmentationLogic":
        from .logic import VesselRaySegmentationLogic

        return VesselRaySegmentationLogic
    raise AttributeError(name)
