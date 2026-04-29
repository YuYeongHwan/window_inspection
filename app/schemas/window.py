from typing import Optional
from pydantic import BaseModel
from app.models.window import ContaminationGrade


class WindowResultResponse(BaseModel):
    id: int
    inspection_id: int
    frame_number: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    contamination_score: float
    grade: ContaminationGrade
    confidence: float
    crop_image_path: Optional[str]

    model_config = {"from_attributes": True}


class GradeSummary(BaseModel):
    grade: ContaminationGrade
    count: int
    percentage: float
