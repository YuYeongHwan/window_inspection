from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
import enum


class InspectionStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    building_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("buildings.id"), nullable=False
    )
    video_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer)
    processed_frames: Mapped[Optional[int]] = mapped_column(Integer)
    total_windows: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[InspectionStatus] = mapped_column(
        SAEnum(InspectionStatus), default=InspectionStatus.PENDING
    )
    inspected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    building: Mapped["Building"] = relationship("Building", back_populates="inspections")  # noqa: F821
    windows: Mapped[List["WindowResult"]] = relationship(  # noqa: F821
        "WindowResult", back_populates="inspection", cascade="all, delete-orphan"
    )
