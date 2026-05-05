import cv2
import numpy as np
import os

IMAGE_PATH = "sample_frame.jpg"
OUTPUT_DIR = "results/grader_test"

# ── 등급 임계값 (오염 픽셀 비율 %) ───────────────────────────────
GRADE_THRESHOLDS = [
    ("A",  0,  10),   # 청결
    ("B", 10,  30),   # 보통
    ("C", 30,  60),   # 오염
    ("D", 60, 100),   # 심각
]

GRADE_COLORS = {
    "A": (0, 255,   0),   # 초록
    "B": (0, 255, 255),   # 노랑
    "C": (0, 165, 255),   # 주황
    "D": (0,   0, 255),   # 빨강
}

# ── 탐지 파라미터 (detector.py와 동일) ───────────────────────────
MIN_AREA   = 3000
ASPECT_MIN = 0.5
ASPECT_MAX = 2.5
NMS_THRESH = 0.05
BORDER_PAD = 20
DUP_XY     = 30
MIN_BRIGHT = 80
# ─────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════
# 탐지 헬퍼 (detector.py와 동일 로직)
# ════════════════════════════════════════════════════════════════

def _is_in_bounds(x, y, bw, bh, img_w, img_h):
    return (x >= BORDER_PAD and y >= BORDER_PAD
            and x + bw <= img_w - BORDER_PAD
            and y + bh <= img_h - BORDER_PAD)


def _brightness_ok(region_hsv):
    if region_hsv.size == 0:
        return False
    return float(np.mean(region_hsv[:, :, 2])) >= MIN_BRIGHT


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


def _nms(boxes, iou_thresh):
    if not boxes:
        return []
    areas  = [bw * bh for _, _, bw, bh in boxes]
    median = float(np.median(areas))
    boxes  = sorted(boxes, key=lambda b: _area_score(b[2], b[3], median), reverse=True)
    kept   = []
    for box in boxes:
        x1, y1, w1, h1 = box
        skip = False
        for k in kept:
            x2, y2, w2, h2 = k
            ix    = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy    = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            inter = ix * iy
            union = w1*h1 + w2*h2 - inter
            if union > 0 and inter / union > iou_thresh:
                skip = True
                break
        if not skip:
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
    """Canny + contour로 창문 박스 [(x,y,w,h), ...] 반환."""
    img_h, img_w = frame.shape[:2]
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)

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

    boxes = _nms(candidates, NMS_THRESH)
    boxes = _dedup(boxes)
    boxes.sort(key=lambda b: b[1])
    return boxes


# ════════════════════════════════════════════════════════════════
# 오염도 분석
# ════════════════════════════════════════════════════════════════

def build_contamination_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """
    창문 크롭에서 오염 픽셀 마스크(0/255) 생성.

    오염 픽셀 (양성):
      - 갈색/황색 얼룩 : H 15-30,  S > 40
      - 회색/검은 먼지 : S < 30,   V < 150
      - 녹색 이끼      : H 35-85,  S > 40

    제외 (음성 — 정상 창문):
      - 파란 유리      : H 100-130, S > 50  (하늘·반사)
      - 흰색 창틀      : S < 30,   V > 200
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # ── 오염 마스크 ──────────────────────────────────────────────
    dirty_brown = cv2.inRange(hsv, ( 15, 40,   0), ( 30, 255, 255))
    dirty_dust  = cv2.inRange(hsv, (  0,  0,   0), (180,  30, 149))
    dirty_moss  = cv2.inRange(hsv, ( 35, 40,   0), ( 85, 255, 255))

    contamination = cv2.bitwise_or(dirty_brown, cv2.bitwise_or(dirty_dust, dirty_moss))

    # ── 정상 창문 제외 마스크 ────────────────────────────────────
    exclude_glass = cv2.inRange(hsv, (100, 50,   0), (130, 255, 255))
    exclude_frame = cv2.inRange(hsv, (  0,  0, 200), (180,  30, 255))

    exclude = cv2.bitwise_or(exclude_glass, exclude_frame)
    contamination = cv2.bitwise_and(contamination, cv2.bitwise_not(exclude))

    kernel        = np.ones((5, 5), np.uint8)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_CLOSE, kernel)
    contamination = cv2.morphologyEx(contamination, cv2.MORPH_OPEN,  kernel)

    return contamination


def make_heatmap(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """오염 마스크를 JET 컬러맵으로 변환 후 원본에 반투명 합성."""
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return cv2.addWeighted(crop_bgr, 0.5, colored, 0.5, 0)


def assign_grade(pct: float) -> str:
    for grade, lo, hi in GRADE_THRESHOLDS:
        if lo <= pct < hi:
            return grade
    return "D"


def analyze_window(crop_bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
    """
    Returns:
        grade   : A/B/C/D
        pct     : 오염 픽셀 비율 (0~100)
        heatmap : 히트맵 이미지
    """
    mask    = build_contamination_mask(crop_bgr)
    pct     = float(np.count_nonzero(mask)) / mask.size * 100
    grade   = assign_grade(pct)
    heatmap = make_heatmap(crop_bgr, mask)
    return grade, pct, heatmap


# ════════════════════════════════════════════════════════════════
# 시각화
# ════════════════════════════════════════════════════════════════

def draw_grade_box(canvas: np.ndarray, x: int, y: int,
                   bw: int, bh: int, idx: int, grade: str, pct: float) -> None:
    color = GRADE_COLORS[grade]
    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)

    label = f"#{idx} {grade} {pct:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # 위 공간이 없으면 박스 안쪽에 표시
    ly = y - 4 if y - th - 8 >= 0 else y + th + 6
    cv2.rectangle(canvas, (x, ly - th - 4), (x + tw + 6, ly + 2), color, -1)
    cv2.putText(canvas, label, (x + 3, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")
    exit(1)

img_h, img_w = frame.shape[:2]
print(f"이미지 크기: {img_w}x{img_h}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

boxes = detect_windows(frame)
print(f"탐지된 창문: {len(boxes)}개\n")

overlay  = frame.copy()
summary  = []

for idx, (x, y, bw, bh) in enumerate(boxes, start=1):
    crop = frame[y:y + bh, x:x + bw]
    grade, pct, heatmap = analyze_window(crop)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"window_{idx:02d}_crop.jpg"),    crop)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"window_{idx:02d}_heatmap.jpg"), heatmap)

    draw_grade_box(overlay, x, y, bw, bh, idx, grade, pct)
    summary.append((idx, grade, pct))

cv2.imwrite(os.path.join(OUTPUT_DIR, "final_result.jpg"), overlay)

# ── 분석 요약 출력 ────────────────────────────────────────────────
grade_label = {"A": "청결", "B": "보통", "C": "오염", "D": "심각"}
print("\n─── 분석 요약 ───────────────────────────────────")
for idx, grade, pct in summary:
    bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    desc = grade_label.get(grade, "")
    print(f"  창문 #{idx:02d}  [{bar}]  {pct:5.1f}%  {grade}등급 ({desc})")

grade_counts = {g: sum(1 for _, gr, _ in summary if gr == g) for g in "ABCD"}
print(f"\n  합계: A={grade_counts['A']}  B={grade_counts['B']}"
      f"  C={grade_counts['C']}  D={grade_counts['D']}  "
      f"(전체 {len(summary)}개)")
print(f"\n저장 완료: {OUTPUT_DIR}/")
print("  final_result.jpg       — 등급 오버레이 원본")
print("  window_NN_crop.jpg     — 창문 크롭")
print("  window_NN_heatmap.jpg  — 오염 히트맵")
