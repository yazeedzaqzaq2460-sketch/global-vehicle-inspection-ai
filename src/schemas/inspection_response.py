from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DamageItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class VehicleData(BaseModel):
    license_plate: Optional[str] = None
    vin: Optional[str] = None
    engine_number: Optional[str] = None


class InspectionSummary(BaseModel):
    damage_count: int = Field(ge=0)
    highest_confidence: float = Field(ge=0.0, le=1.0)


class InspectionResponse(BaseModel):
    inspection_id: str
    status: str
    image_name: str
    decision: str
    vehicle: VehicleData
    damages: list[DamageItem]
    summary: InspectionSummary