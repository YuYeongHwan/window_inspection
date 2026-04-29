from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.window import WindowResult, ContaminationGrade
from app.schemas.window import WindowResultResponse, GradeSummary

router = APIRouter(prefix="/api/windows", tags=["windows"])


@router.get("/", response_model=list[WindowResultResponse])
def list_windows(
    inspection_id: Optional[int] = None,
    grade: Optional[ContaminationGrade] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    q = db.query(WindowResult)
    if inspection_id:
        q = q.filter(WindowResult.inspection_id == inspection_id)
    if grade:
        q = q.filter(WindowResult.grade == grade)
    return q.limit(limit).all()


@router.get("/summary/{inspection_id}", response_model=list[GradeSummary])
def grade_summary(inspection_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(WindowResult.grade, func.count(WindowResult.id).label("count"))
        .filter(WindowResult.inspection_id == inspection_id)
        .group_by(WindowResult.grade)
        .all()
    )
    total = sum(r.count for r in rows)
    return [
        GradeSummary(
            grade=r.grade,
            count=r.count,
            percentage=round(r.count / total * 100, 1) if total else 0.0,
        )
        for r in rows
    ]


@router.get("/{window_id}/image")
def get_window_image(window_id: int, db: Session = Depends(get_db)):
    win = db.query(WindowResult).filter(WindowResult.id == window_id).first()
    if not win or not win.crop_image_path:
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return FileResponse(win.crop_image_path, media_type="image/jpeg")
