from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.building import Building
from app.models.inspection import Inspection
from app.models.window import WindowResult

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/")
def dashboard_summary(db: Session = Depends(get_db)):
    """메인 대시보드: 건물 목록 + 각 건물의 최근 점검 요약."""
    buildings = db.query(Building).order_by(Building.created_at.desc()).all()
    result = []

    for b in buildings:
        latest = (
            db.query(Inspection)
            .filter(Inspection.building_id == b.id)
            .order_by(Inspection.inspected_at.desc())
            .first()
        )

        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        total_windows = 0

        if latest:
            rows = (
                db.query(WindowResult.grade, func.count(WindowResult.id).label("cnt"))
                .filter(WindowResult.inspection_id == latest.id)
                .group_by(WindowResult.grade)
                .all()
            )
            for row in rows:
                grade_counts[row.grade.value] = row.cnt
                total_windows += row.cnt

        result.append({
            "building": {
                "id": b.id,
                "name": b.name,
                "address": b.address,
                "floor_count": b.floor_count,
            },
            "latest_inspection": {
                "id": latest.id,
                "status": latest.status.value,
                "inspected_at": latest.inspected_at.isoformat(),
                "total_frames": latest.total_frames,
            } if latest else None,
            "grade_counts": grade_counts,
            "total_windows": total_windows,
        })

    return result


@router.get("/inspection/{inspection_id}")
def inspection_detail(inspection_id: int, db: Session = Depends(get_db)):
    """점검 상세: 창문 결과 전체 + 등급 통계."""
    inspection = db.get(Inspection, inspection_id)
    if not inspection:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="점검을 찾을 수 없습니다.")

    building = db.get(Building, inspection.building_id)

    rows = (
        db.query(WindowResult.grade, func.count(WindowResult.id).label("cnt"))
        .filter(WindowResult.inspection_id == inspection_id)
        .group_by(WindowResult.grade)
        .all()
    )
    total = sum(r.cnt for r in rows)
    grade_summary = {r.grade.value: r.cnt for r in rows}
    for g in "ABCD":
        grade_summary.setdefault(g, 0)

    windows = (
        db.query(WindowResult)
        .filter(WindowResult.inspection_id == inspection_id)
        .order_by(WindowResult.frame_number, WindowResult.id)
        .all()
    )

    return {
        "inspection": {
            "id": inspection.id,
            "status": inspection.status.value,
            "inspected_at": inspection.inspected_at.isoformat(),
            "total_frames": inspection.total_frames,
            "processed_frames": inspection.processed_frames,
            "video_filename": inspection.video_filename,
        },
        "building": {
            "id": building.id,
            "name": building.name,
            "address": building.address,
        } if building else None,
        "grade_summary": grade_summary,
        "total_windows": total,
        "windows": [
            {
                "id": w.id,
                "frame_number": w.frame_number,
                "bbox_x": w.bbox_x, "bbox_y": w.bbox_y,
                "bbox_w": w.bbox_w, "bbox_h": w.bbox_h,
                "contamination_score": w.contamination_score,
                "grade": w.grade.value,
                "confidence": w.confidence,
            }
            for w in windows
        ],
    }
