"""Slicer workflow and independently importable MONAI transport."""

__all__ = ["VesselRaySegmentationLogic"]


def __getattr__(name):
    if name == "VesselRaySegmentationLogic":
        from .logic import VesselRaySegmentationLogic

        return VesselRaySegmentationLogic
    raise AttributeError(name)
