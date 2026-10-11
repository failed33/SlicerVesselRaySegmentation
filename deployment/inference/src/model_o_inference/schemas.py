"""ModelO request inputs, independent of serving and scene frameworks."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictFloat

FiniteNumber = Annotated[StrictFloat, Field(allow_inf_nan=False)]
PositionZYX = tuple[FiniteNumber, FiniteNumber, FiniteNumber]
PolylineZYX = Annotated[tuple[PositionZYX, ...], Field(min_length=2)]


class ModelOParameters(BaseModel):
    """Ordered lines of finite ZYX voxel positions in the submitted CT grid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    polylines_zyx: Annotated[tuple[PolylineZYX, ...], Field(min_length=1)]
    overlap: Annotated[FiniteNumber, Field(ge=0.0, lt=1.0)] | None = Field(
        default=0.5,
        description=(
            "Nominal linear patch overlap along each polyline. Ray sampling remains 1 mm; "
            "null selects one patch per ray origin for the dense reference schedule."
        ),
    )
