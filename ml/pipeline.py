"""
영상 처리 파이프라인: 비디오 → 프레임 샘플링 → 창문 탐지 → 오염도 분석 → DB 저장

실행 예:
    python ml/pipeline.py --video ./test_video.MOV --building_id 1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 로거 설정 ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 탐지 파라미터 ─────────────────────────────────────────────────
MIN_AREA    = 3000
ASPECT_MIN  = 0.5
ASPECT_MAX  = 2.5
NMS_THRESH  = 0.05
BORDER_PAD  = 20
DUP_XY      = 30
MIN_BRIGHT  = 80

# ── 트래킹 파라미터 ───────────────────────────────────────────────
TRACK_IOU_THRESH = 0.30   # 같은 창문으로 판단할 IoU
FRAME_SAMPLE_RATE = 30    # 매 N프레임마다 처리


# ════════════════════════════════════════════════════════════════
# 탐지 함수 (detector.py 로직 인라인)
# ════════════════════════════════════════════════════════════════

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def _is_in_bounds(x, y, bw, bh, img_w, img_h):
    return (x >= BORDER_PAD and y >= BORDER_PAD
            and x + bw <= img_w - BORDER_PAD
            and y + bh <= img_h - BORDER_PAD)


def _brightness_ok(region_hsv):
    return region_hsv.size > 0 and float(np.mean(region_hsv[:, :, 2])) >= MIN_BRIGHT


def _has_grid_pattern(region_bgr):
    if region_bgr.size == 0:
        return False
    rh, rw = region_bgr.shape[:2]
    if rh < 10 or rw < 10:
        return False
    gray   = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    my, mx = max(1, rh // 10), max(1, rw // 10)
    border = np.concatenate([
        gray[:my, :].flatten(), gray[-my:, :].flatten(),
        gray[my:-my, :mx].flatten(), gray[my:-my, -mx:].flatten(),
    ])
    hsv_roi  = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    interior = hsv_roi[my:-my, mx:-mx]
    if interior.size == 0:
        return False
    blue_mask  = cv2.inRange(interior, (90, 20, 60), (130, 255, 255))
    blue_ratio = float(np.count_nonzero(blue_mask)) / interior[:, :, 0].size
    return float(np.mean(border)) >= 160 and blue_ratio >= 0.05


def _area_score(bw, bh, median_area):
    if median_area == 0:
        return 0.0
    ratio = (bw * bh) / median_area
    if ratio < 0.25 or ratio > 4.0:
        return 0.0
    return 1.0 - abs(1.0 - ratio) / max(1.0, ratio)


def _nms(boxes):
    if not boxes:
        return []
    areas  = [bw * bh for _, _, bw, bh in boxes]
    median = float(np.median(areas))
    boxes  = sorted(boxes, key=lambda b: _area_score(b[2], b[3], median), reverse=True)
    kept   = []
    for box in boxes:
        x1, y1, w1, h1 = box
        if not any(_iou(box, k) > NMS_THRESH for k in kept):
            kept.append(box)
    return kept


def _dedup(boxes):
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dup = False
        for i, k in enumerate(kept):
            x2, y2, w2, h2 = k
            if abs(x1-x2) <= DUP_XY and abs(y1-y2) <= DUP_XY:
                if w1*h1 > w2*h2:
                    kept[i] = box
                dup = True
                break
        if not dup:
            kept.append(box)
    return kept


def detect_windows(frame: np.ndarray) -> list[tuple]:
    """창문 박스 [(x,y,w,h), ...] 반환."""
    img_h, img_w = frame.shape[:2]
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []

    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bw * bh < MIN_AREA:
            continue
        aspect = bw / bh if bh > 0 else 0
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue
        if not _is_in_bounds(x, y, bw, bh, img_w, img_h):
            continue
        roi_hsv = hsv[y:y+bh, x:x+bw]
        roi_bgr = frame[y:y+bh, x:x+bw]
        if not _brightness_ok(roi_hsv) and not _has_grid_pattern(roi_bgr):
            continue
        candidates.append((x, y, bw, bh))

    boxes = _nms(candidates)
    boxes = _dedup(boxes)
    boxes.sort(key=lambda b: b[1])
    return boxes


# ════════════════════════════════════════════════════════════════
# 오염도 분석 함수 (grader.py 로직 인라인)
# ════════════════════════════════════════════════════════════════

GRADE_THRESHOLDS = [
    ("A",  0,  10),
    ("B", 10,  30),
    ("C", 30,  60),
    ("D", 60, 100),
]


def _contamination_mask(crop_bgr: np.ndarray) -> np.ndarray:
    hsv         = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    dirty_brown = cv2.inRange(hsv, ( 15, 40,   0), ( 30, 255, 255))
    dirty_dust  = cv2.inRange(hsv, (  0,  0,   0), (180,  30, 149))
    dirty_moss  = cv2.inRange(hsv, ( 35, 40,   0), ( 85, 255, 255))
    contamination = cv2.bitwise_or(dirty_brown, cv2.bitwise_or(dirty_dust, dirty_moss))

    exclude_glass = cv2.inRange(hsv, (100, 50,   0), (130, 255, 255))
    exclude_frame = cv2.inRange(hsv, (  0,  0, 200), (180,  30, 255))
    exclude = cv2.bitwise_or(exclude_glass, exclude_frame)
    contamination = cv2.bitwise_and(contamination, cv2.bitwise_not(exclude))

    kernel = np.ones((5, 5), np.uint8)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_CLOSE, kernel)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_OPEN,  kernel)
    return contamination


def analyze_window(crop_bgr: np.ndarray) -> tuple[str, float]:
    """(grade, contamination_score 0.0~1.0) 반환."""
    mask  = _contamination_mask(crop_bgr)
    pct   = float(np.count_nonzero(mask)) / mask.size * 100
    grade = next((g for g, lo, hi in GRADE_THRESHOLDS if lo <= pct < hi), "D")
    return grade, round(pct / 100.0, 4)


# ════════════════════════════════════════════════════════════════
# 창문 트래커 — 프레임 간 동일 창문 식별
# ════════════════════════════════════════════════════════════════

class WindowTracker:
    def __init__(self, iou_thresh: float = TRACK_IOU_THRESH):
        self.tracks: list[dict] = []
        self.next_id = 1
        self.iou_thresh = iou_thresh

    def update(self, boxes: list[tuple], frame_idx: int) -> list[tuple[int, tuple]]:
        """
        boxes: [(x,y,w,h), ...]
        반환: [(track_id, bbox), ...]  — 이번 프레임에서 매칭된/새로운 트랙 목록
        """
        matched_track_ids = set()
        result = []

        for box in boxes:
            best_iou, best_idx = 0.0, -1
            for i, track in enumerate(self.tracks):
                iou = _iou(box, track["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= self.iou_thresh and best_idx not in matched_track_ids:
                matched_track_ids.add(best_idx)
                self.tracks[best_idx]["bbox"]        = box
                self.tracks[best_idx]["last_frame"]  = frame_idx
                self.tracks[best_idx]["frames_seen"] += 1
                result.append((self.tracks[best_idx]["id"], box))
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks.append({
                    "id": track_id, "bbox": box,
                    "last_frame": frame_idx, "frames_seen": 1,
                })
                result.append((track_id, box))

        return result

    @property
    def unique_count(self) -> int:
        return len(self.tracks)


# ════════════════════════════════════════════════════════════════
# DB 저장
# ════════════════════════════════════════════════════════════════

def _save_results(db, inspection_id: int, building_id: int,
                  frame_idx: int, track_id: int, box: tuple,
                  grade: str, score: float, crop: np.ndarray,
                  results_dir: Path, img_h: int) -> None:
    from app.models.window import WindowResult, ContaminationGrade

    x, y, bw, bh = box
    crop_path = results_dir / f"f{frame_idx:05d}_t{track_id:03d}.jpg"
    cv2.imwrite(str(crop_path), crop)

    wr = WindowResult(
        inspection_id=inspection_id,
        frame_number=frame_idx,
        bbox_x=x, bbox_y=y, bbox_w=bw, bbox_h=bh,
        contamination_score=score,
        grade=ContaminationGrade(grade),
        confidence=1.0,
        crop_image_path=str(crop_path),
    )
    db.add(wr)


def _create_window_records(db, building_id: int, tracker: WindowTracker,
                           img_h: int) -> None:
    from app.models.window import Window

    for track in tracker.tracks:
        x, y, bw, bh = track["bbox"]
        cy_ratio = (y + bh / 2) / img_h if img_h > 0 else 0.5
        # 상/중/하 3구역으로 층수 추정 (1=상, 2=중, 3=하)
        floor_est = 1 if cy_ratio < 0.33 else (2 if cy_ratio < 0.66 else 3)

        win = Window(
            building_id=building_id,
            floor=floor_est,
            position_x=float(x), position_y=float(y),
            width=float(bw), height=float(bh),
        )
        db.add(win)


# ════════════════════════════════════════════════════════════════
# 메인 파이프라인
# ════════════════════════════════════════════════════════════════

def run_pipeline(video_path: str, building_id: int) -> None:
    from app.core.database import SessionLocal, init_db
    from app.models.building import Building
    from app.models.inspection import Inspection, InspectionStatus

    # ── 영상 파일 확인 ────────────────────────────────────────────
    if not os.path.exists(video_path):
        log.error("영상 파일을 찾을 수 없습니다: %s", video_path)
        sys.exit(1)

    # ── DB 초기화 및 연결 ─────────────────────────────────────────
    try:
        init_db()
        db = SessionLocal()
    except Exception as e:
        log.error("DB 연결 실패: %s", e)
        sys.exit(1)

    try:
        # 건물 존재 확인
        building = db.get(Building, building_id)
        if building is None:
            log.error("building_id=%d 가 DB에 없습니다.", building_id)
            sys.exit(1)

        log.info("건물: %s (id=%d)", building.name, building_id)

        # Inspection 레코드 생성
        inspection = Inspection(
            building_id=building_id,
            video_filename=os.path.basename(video_path),
            status=InspectionStatus.PROCESSING,
        )
        db.add(inspection)
        db.commit()
        db.refresh(inspection)
        log.info("Inspection 생성 (id=%d)", inspection.id)

        results_dir = Path("results") / str(inspection.id)
        results_dir.mkdir(parents=True, exist_ok=True)

        # ── 영상 열기 ─────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        inspection.total_frames = total_frames
        db.commit()

        log.info("영상 정보: 총 %d 프레임 / %.1f fps", total_frames, fps)

        tracker        = WindowTracker()
        frame_idx      = 0
        sampled_count  = 0
        total_detected = 0
        img_h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        grade_counts   = {"A": 0, "B": 0, "C": 0, "D": 0}

        # ── 프레임 루프 ───────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SAMPLE_RATE == 0:
                boxes   = detect_windows(frame)
                tracked = tracker.update(boxes, frame_idx)
                total_detected += len(tracked)

                for track_id, box in tracked:
                    x, y, bw, bh = box
                    crop          = frame[y:y+bh, x:x+bw]
                    grade, score  = analyze_window(crop)
                    grade_counts[grade] += 1

                    _save_results(db, inspection.id, building_id,
                                  frame_idx, track_id, box,
                                  grade, score, crop, results_dir, img_h)

                inspection.processed_frames = sampled_count + 1
                inspection.total_windows    = total_detected
                db.commit()

                elapsed = frame_idx / fps
                log.info("[%6.1fs | 프레임 %5d] 창문 %d개 감지",
                         elapsed, frame_idx, len(tracked))
                sampled_count += 1

            frame_idx += 1

        cap.release()

        # ── 고유 창문 Window 레코드 저장 ─────────────────────────
        _create_window_records(db, building_id, tracker, img_h)
        inspection.status = InspectionStatus.COMPLETED
        db.commit()

        # ── 요약 리포트 ───────────────────────────────────────────
        _print_summary(building.name, inspection.id, sampled_count,
                       tracker, grade_counts)

    except Exception as e:
        log.error("파이프라인 오류: %s", e, exc_info=True)
        try:
            from app.models.inspection import InspectionStatus
            inspection.status = InspectionStatus.FAILED
            db.commit()
        except Exception:
            pass
        sys.exit(1)
    finally:
        db.close()


def _print_summary(building_name: str, inspection_id: int,
                   sampled_frames: int, tracker: WindowTracker,
                   grade_counts: dict) -> None:
    total = sum(grade_counts.values())
    log.info("")
    log.info("═" * 52)
    log.info("  분석 완료 — %s  (inspection #%d)", building_name, inspection_id)
    log.info("  처리 프레임: %d  /  고유 창문: %d개  /  탐지 누적: %d건",
             sampled_frames, tracker.unique_count, total)
    log.info("  ─────────────────────────────────────────────")
    labels = {"A": "청결(0~10%)", "B": "보통(10~30%)",
              "C": "오염(30~60%)", "D": "심각(60%+)"}
    for g in "ABCD":
        cnt  = grade_counts[g]
        pct  = cnt / total * 100 if total > 0 else 0
        bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        log.info("  %s등급 %-12s [%s] %3d건 (%4.1f%%)",
                 g, labels[g], bar, cnt, pct)
    log.info("═" * 52)


# ════════════════════════════════════════════════════════════════
# CLI 진입점
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="창문 오염도 검사 파이프라인")
    parser.add_argument("--video",       required=True, help="분석할 영상 파일 경로")
    parser.add_argument("--building_id", required=True, type=int, help="DB buildings.id")
    args = parser.parse_args()

    run_pipeline(args.video, args.building_id)
