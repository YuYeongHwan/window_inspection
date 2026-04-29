from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Building(Base):
    __tablename__ = "buildings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    address: Mapped[Optional[str]] = mapped_column(String(500))
    floor_count: Mapped[Optional[int]] = mapped_column(Integer)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    inspections: Mapped[List["Inspection"]] = relationship(  # noqa: F821
        "Inspection", back_populates="building", cascade="all, delete-orphan"
    )
    windows: Mapped[List["Window"]] = relationship(  # noqa: F821
        "Window", back_populates="building", cascade="all, delete-orphan"
    )
