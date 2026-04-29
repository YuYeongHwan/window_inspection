from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from app.models.inspection import InspectionStatus


class InspectionCreate(BaseModel):
    building_id: int


class InspectionResponse(BaseModel):
    id: int
    building_id: int
    video_filename: str
    total_frames: Optional[int]
    processed_frames: Optional[int]
    total_windows: int
    status: InspectionStatus
    inspected_at: datetime

    model_config = {"from_attributes": True}
