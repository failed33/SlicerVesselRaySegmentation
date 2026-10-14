"""Native centerline preparation and Markups geometry shared by inference clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PositionZYX = tuple[float, float, float]
PolylineZYX = tuple[PositionZYX, ...]
MARKUP_TYPES = ("vtkMRMLMarkupsCurveNode", "vtkMRMLMarkupsLineNode")


def _has_populated_markup_descendants(item_id: int, hierarchy: Any) -> bool:
    import vtk

    descendants = vtk.vtkIdList()
    hierarchy.GetItemChildren(item_id, descendants, True)
    for index in range(descendants.GetNumberOfIds()):
        node = hierarchy.GetItemDataNode(descendants.GetId(index))
        if (
            node
            and any(node.IsA(kind) for kind in MARKUP_TYPES)
            and node.GetNumberOfControlPoints() > 0
        ):
            return True
    return False


def _markup_nodes(item_id: int) -> list[Any]:
    """Include nested curves, omitting empty Markups used only as parents."""
    import slicer
    import vtk

    hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    children = vtk.vtkIdList()
    hierarchy.GetItemChildren(item_id, children, True)
    items = [item_id, *(children.GetId(i) for i in range(children.GetNumberOfIds()))]
    nodes = []
    for item in items:
        node = hierarchy.GetItemDataNode(item)
        if not node or not any(node.IsA(kind) for kind in MARKUP_TYPES):
            continue
        if node.GetNumberOfControlPoints() == 0 and _has_populated_markup_descendants(
            item, hierarchy
        ):
            continue
        nodes.append(node)
    return nodes


def _world_points(node: Any) -> np.ndarray:
    import slicer

    if node.GetNumberOfDefinedControlPoints() < 2:
        raise ValueError(f"'{node.GetName()}' needs at least two defined points")
    if node.GetNumberOfDefinedControlPoints() != node.GetNumberOfControlPoints():
        raise ValueError(f"Finish placing the points in '{node.GetName()}' before inference")
    if node.IsA("vtkMRMLMarkupsLineNode"):
        return slicer.util.arrayFromMarkupsControlPoints(node, world=True)
    return slicer.util.arrayFromMarkupsCurvePoints(node, world=True)


def collect_polylines(item_id: int, ijk_to_ras: np.ndarray) -> tuple[PolylineZYX, ...]:
    """Snapshot selected Markups into the captured CT grid without rounding.

    Select a line, a curve, or a Subject Hierarchy folder. A VMTK root curve
    includes its descendant branches. Empty Markups with populated descendants
    are organizational parents; empty leaves and unfinished curves are errors.
    Call on Slicer's UI thread; the returned tuples retain no references to MRML
    nodes or mutable VTK point arrays.
    """
    if not item_id:
        raise ValueError("Select a centerline curve, line, or folder")
    nodes = _markup_nodes(item_id)
    if not nodes:
        raise ValueError("The selection contains no Markups curves or lines")
    ras_to_ijk = np.linalg.inv(ijk_to_ras)
    polylines = []
    for node in nodes:
        points = _world_points(node)
        ijk = points @ ras_to_ijk[:3, :3].T + ras_to_ijk[:3, 3]
        polylines.append(tuple((float(k), float(j), float(i)) for i, j, k in ijk))
    return tuple(polylines)


def add_network_curves(network: Any, parent_curve: Any) -> None:
    """Add ordered network geometry below an organizational parent curve.

    Each native VMTK cell becomes one Markups curve with the same world points.
    Radius measurements remain in the network; inference consumes only geometry.
    Call on Slicer's UI thread. The caller owns the parent and its existing children.
    """
    import slicer

    hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    parent_item = hierarchy.GetItemByDataNode(parent_curve)
    markups_logic = slicer.modules.markups.logic()
    slicer.app.pauseRender()
    try:
        for cell_id in range(network.GetNumberOfCells()):
            curve = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", f"{parent_curve.GetName()} ({cell_id})"
            )
            curve.CreateDefaultDisplayNodes()
            curve.GetDisplayNode().SetSelectedColor(0.0, 0.8, 1.0)
            with slicer.util.NodeModify(curve):
                curve.SetNumberOfPointsPerInterpolatingSegment(1)
                curve.SetControlPointPositionsWorld(network.GetCell(cell_id).GetPoints())
                markups_logic.SetAllControlPointsVisibility(curve, False)
            hierarchy.SetItemParent(hierarchy.GetItemByDataNode(curve), parent_item)
    finally:
        slicer.app.resumeRender()


@dataclass(frozen=True)
class NetworkBranch:
    """An unbranched VMTK network cell, measured in the surface's physical frame."""

    cell_id: int
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]]
    length_mm: float
    estimated_volume_mm3: float


def measure_network_branches(network: Any) -> tuple[NetworkBranch, ...]:
    """Measure native lines without tracing paths or joining junctions."""
    radius = network.GetPointData().GetArray("Radius")
    branches = []
    for cell_id in range(network.GetNumberOfCells()):
        cell = network.GetCell(cell_id)
        if cell.GetNumberOfPoints() < 2:
            continue
        ids = [cell.GetPointId(index) for index in range(cell.GetNumberOfPoints())]
        points = np.asarray([network.GetPoint(index) for index in ids])
        radii = np.asarray([radius.GetTuple1(index) for index in ids])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        volume = np.sum(distances * np.pi * (radii[:-1] ** 2 + radii[1:] ** 2) / 2)
        branches.append(
            NetworkBranch(
                cell_id=cell_id,
                endpoints=(tuple(points[0]), tuple(points[-1])),
                length_mm=float(distances.sum()),
                estimated_volume_mm3=float(volume),
            )
        )
    return tuple(branches)


@dataclass(frozen=True)
class NetworkExtraction:
    """Complete native network geometry and per-component extraction accounting."""

    network: Any
    summary: dict[str, Any]


def _surface_components(surface: Any) -> Any:
    import vtk

    regions = vtk.vtkPolyDataConnectivityFilter()
    regions.SetInputData(surface)
    regions.SetExtractionModeToAllRegions()
    regions.Update()
    for index in range(regions.GetNumberOfExtractedRegions()):
        component = vtk.vtkPolyDataConnectivityFilter()
        component.SetInputData(surface)
        component.SetExtractionModeToSpecifiedRegions()
        component.AddSpecifiedRegion(index)
        component.Update()
        yield index, component.GetOutput()


def _extract_component(logic: Any, surface: Any, index: int) -> tuple[Any, dict[str, Any]]:
    import vtk

    prepared = logic.preprocess(surface, surface.GetNumberOfPoints(), 4.0, False)
    volume = vtk.vtkMassProperties()
    volume.SetInputData(prepared)
    volume.Update()
    details = {"component": index, "surface_volume_mm3": volume.GetVolume()}
    try:
        network = logic.extractNetwork(prepared, None)
    except ValueError as error:
        details["error"] = str(error)
        network = vtk.vtkPolyData()
    branches = measure_network_branches(network)
    details.update(branches=len(branches), length_mm=sum(b.length_mm for b in branches))
    return network, details


def extract_networks(segmentation: Any, segment_id: str) -> NetworkExtraction:
    """Extract every branch in every selected surface component with native VMTK.

    These are approximate networks, not Voronoi-refined centerlines. Empty and
    failed surface components are reported rather than replaced with traced paths.
    Input is a prepared segmentation with its parent transform hardened.
    """
    import vtk
    from ExtractCenterline import ExtractCenterlineLogic

    logic = ExtractCenterlineLogic()
    surface = logic.polyDataFromNode(segmentation, segment_id)
    append = vtk.vtkAppendPolyData()
    components = []
    for index, component in _surface_components(surface):
        network, details = _extract_component(logic, component, index)
        components.append(details)
        if network.GetNumberOfCells():
            append.AddInputData(network)
    append.Update()
    network = vtk.vtkPolyData()
    network.DeepCopy(append.GetOutput())
    branches = measure_network_branches(network)
    represented = [component for component in components if component["branches"]]
    total_volume = sum(component["surface_volume_mm3"] for component in components)
    represented_volume = sum(component["surface_volume_mm3"] for component in represented)
    summary = {
        "method": "native approximate VMTK network, every selected surface component",
        "surface_components": len(components),
        "represented_components": len(represented),
        "empty_components": sum(not c["branches"] and "error" not in c for c in components),
        "failed_components": sum("error" in c for c in components),
        "curve_count": len(branches),
        "total_length_mm": sum(branch.length_mm for branch in branches),
        "bounds_ras": list(network.GetBounds()),
        "represented_surface_volume_fraction": represented_volume / total_volume
        if total_volume
        else 0.0,
        "components": components,
    }
    return NetworkExtraction(network, summary)
