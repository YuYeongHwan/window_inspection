from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class BuildingCreate(BaseModel):
    name: str
    address: Optional[str] = None
    floor_count: Optional[int] = None
    description: Optional[str] = None


class BuildingResponse(BaseModel):
    id: int
    name: str
    address: Optional[str]
    floor_count: Optional[int]
    description: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}
