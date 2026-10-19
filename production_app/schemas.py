from pydantic import BaseModel, ConfigDict, Field


class PaitentData(BaseModel):
    # The original feature names with spaces are kept as aliases.
    model_config = ConfigDict(populate_by_name=True)

    texture_mean: float
    perimeter_mean: float
    area_mean: float
    concavity_mean: float
    concave_points_mean: float = Field(alias="concave points_mean")
    radius_se: float
    area_se: float
    compactness_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    concavity_worst: float
    concave_points_worst: float = Field(alias="concave points_worst")
    symmetry_worst: float
