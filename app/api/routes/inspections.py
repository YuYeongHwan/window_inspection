import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.building import Building
from app.models.inspection import Inspection, InspectionStatus
from app.schemas.inspection import InspectionResponse

router = APIRouter(prefix="/api/inspections", tags=["inspections"])


def _run_pipeline(inspection_id: int):
    from app.core.database import SessionLocal
    from ml.pipeline import InspectionPipeline

    db = SessionLocal()
    try:
        inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
        if not inspection:
            return
        video_path = os.path.join("uploads", inspection.video_filename)
        pipeline = InspectionPipeline()
        pipeline.process(inspection, video_path, db)
    finally:
        db.close()


@router.get("/", response_model=list[InspectionResponse])
def list_inspections(building_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Inspection)
    if building_id:
        q = q.filter(Inspection.building_id == building_id)
    return q.order_by(Inspection.inspected_at.desc()).all()


@router.post("/", response_model=InspectionResponse, status_code=201)
async def create_inspection(
    background_tasks: BackgroundTasks,
    building_id: int = Form(...),
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    building = db.query(Building).filter(Building.id == building_id).first()
    if not building:
        raise HTTPException(status_code=404, detail="건물을 찾을 수 없습니다.")

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    save_path = upload_dir / video.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    inspection = Inspection(
        building_id=building_id,
        video_filename=video.filename,
        status=InspectionStatus.PENDING,
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    background_tasks.add_task(_run_pipeline, inspection.id)
    return inspection


@router.get("/{inspection_id}", response_model=InspectionResponse)
def get_inspection(inspection_id: int, db: Session = Depends(get_db)):
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="검사를 찾을 수 없습니다.")
    return inspection


@router.delete("/{inspection_id}", status_code=204)
def delete_inspection(inspection_id: int, db: Session = Depends(get_db)):
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="검사를 찾을 수 없습니다.")
    db.delete(inspection)
    db.commit()
