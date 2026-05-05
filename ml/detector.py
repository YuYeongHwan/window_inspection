import cv2
import numpy as np

IMAGE_PATH = "sample_frame.jpg"

# ── 파라미터 ──────────────────────────────────────────────────────
MIN_AREA    = 3000    # 최소 넓이 (px²)
ASPECT_MIN  = 0.5     # 가로/세로 비율 하한
ASPECT_MAX  = 2.5     # 가로/세로 비율 상한
NMS_THRESH  = 0.05    # IoU overlap threshold
BORDER_PAD  = 20      # 이미지 경계에서 벗어나면 안 되는 여백 (px)
DUP_XY      = 30      # 중복 박스 판단 좌표 허용 오차 (px)
MIN_BRIGHT  = 80      # HSV V채널 최솟값 (이보다 어두우면 제외)
# ─────────────────────────────────────────────────────────────────


def is_in_bounds(x: int, y: int, bw: int, bh: int, img_w: int, img_h: int) -> bool:
    """경계에서 BORDER_PAD px 이상 안쪽에 있어야 True."""
    return (x >= BORDER_PAD and y >= BORDER_PAD
            and x + bw <= img_w - BORDER_PAD
            and y + bh <= img_h - BORDER_PAD)


def floor_label(idx: int, y: int, bh: int, img_h: int) -> str:
    """박스 중심 y 위치로 TOP/MID/BOT 구분, 번호 포함."""
    cy = y + bh / 2
    ratio = cy / img_h
    if ratio < 0.33:
        zone = "TOP"
    elif ratio < 0.66:
        zone = "MID"
    else:
        zone = "BOT"
    return f"F{idx} {zone}"


def brightness_ok(region_hsv: np.ndarray) -> bool:
    """영역 평균 밝기가 MIN_BRIGHT 이상이면 True."""
    if region_hsv.size == 0:
        return False
    return float(np.mean(region_hsv[:, :, 2])) >= MIN_BRIGHT


def has_grid_pattern(region_bgr: np.ndarray) -> bool:
    """창문 격자 패턴 확인: 밝은 테두리(흰색 프레임) + 푸른 유리 영역."""
    if region_bgr.size == 0:
        return False

    rh, rw = region_bgr.shape[:2]
    if rh < 10 or rw < 10:
        return False

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)

    # 외곽 10% 테두리 평균 밝기
    margin_y = max(1, rh // 10)
    margin_x = max(1, rw // 10)
    border = np.concatenate([
        gray[:margin_y, :].flatten(),
        gray[-margin_y:, :].flatten(),
        gray[margin_y:-margin_y, :margin_x].flatten(),
        gray[margin_y:-margin_y, -margin_x:].flatten(),
    ])
    border_mean = float(np.mean(border))

    # 내부 HSV에서 파란 계열 픽셀 비율 확인 (H: 90~130)
    hsv_roi = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    interior = hsv_roi[margin_y:-margin_y, margin_x:-margin_x]
    if interior.size == 0:
        return False
    blue_mask = cv2.inRange(interior, (90, 20, 60), (130, 255, 255))
    blue_ratio = float(np.count_nonzero(blue_mask)) / interior[:, :, 0].size

    # 테두리가 밝고(흰 프레임) 내부에 파란 유리가 있으면 창문
    return border_mean >= 160 and blue_ratio >= 0.05


def _area_score(bw: int, bh: int, median_area: float) -> float:
    """중앙값 면적 대비 얼마나 적절한 크기인지 0~1로 반환. 너무 크거나 작으면 낮아짐."""
    area = bw * bh
    if median_area == 0:
        return 0.0
    ratio = area / median_area
    # 0.25~4.0 배 범위를 적정 크기로 보고, 중앙값에 가까울수록 1.0
    if ratio < 0.25 or ratio > 4.0:
        return 0.0
    return 1.0 - abs(1.0 - ratio) / max(1.0, ratio)


def dedup_by_position(boxes: list[tuple]) -> list[tuple]:
    """비슷한 (x, y) 위치의 박스를 면적 기준으로 하나만 남김."""
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        duplicate = False
        for i, k in enumerate(kept):
            x2, y2, w2, h2 = k
            if abs(x1 - x2) <= DUP_XY and abs(y1 - y2) <= DUP_XY:
                # 둘 중 면적이 더 큰 쪽 유지
                if w1 * h1 > w2 * h2:
                    kept[i] = box
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def nms(boxes: list[tuple], iou_thresh: float) -> list[tuple]:
    """면적 적절도 점수 내림차순으로 NMS 적용."""
    if not boxes:
        return []

    areas   = [bw * bh for _, _, bw, bh in boxes]
    median  = float(np.median(areas))
    boxes   = sorted(boxes, key=lambda b: _area_score(b[2], b[3], median), reverse=True)
    kept    = []

    for box in boxes:
        x1, y1, w1, h1 = box
        dominated = False
        for k in kept:
            x2, y2, w2, h2 = k
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            if union > 0 and inter / union > iou_thresh:
                dominated = True
                break
        if not dominated:
            kept.append(box)

    return kept


# ── 이미지 로드 ───────────────────────────────────────────────────
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")
    exit(1)

img_h, img_w = frame.shape[:2]
print(f"이미지 크기: {img_w}x{img_h}")

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# ── Canny edge detection ──────────────────────────────────────────
gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges   = cv2.Canny(blurred, 30, 100)

cv2.imwrite("debug_edge.jpg", edges)
print("debug_edge.jpg 저장 완료")

# ── contour 필터링 ────────────────────────────────────────────────
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

candidates = []

for cnt in contours:
    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) != 4:
        continue

    x, y, bw, bh = cv2.boundingRect(approx)
    area   = bw * bh
    aspect = bw / bh if bh > 0 else 0

    if area < MIN_AREA:
        continue
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        continue
    if not is_in_bounds(x, y, bw, bh, img_w, img_h):
        continue

    roi_hsv = hsv[y:y + bh, x:x + bw]
    roi_bgr = frame[y:y + bh, x:x + bw]

    # 밝기 필터 통과 또는 격자 패턴으로 창문 확인
    if not brightness_ok(roi_hsv) and not has_grid_pattern(roi_bgr):
        continue

    candidates.append((x, y, bw, bh))

# ── NMS → 좌표 중복 제거 ─────────────────────────────────────────
final_boxes = nms(candidates, NMS_THRESH)
final_boxes = dedup_by_position(final_boxes)

# 위→아래 정렬 (층수 표시를 직관적으로)
final_boxes.sort(key=lambda b: b[1])

# ── 결과 시각화 ───────────────────────────────────────────────────
result = frame.copy()

for i, (x, y, bw, bh) in enumerate(final_boxes, start=1):
    label = floor_label(i, y, bh, img_h)
    cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    # 라벨 배경
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(result, (x, y - th - 6), (x + tw + 4, y), (0, 255, 0), -1)
    cv2.putText(result, label, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite("debug_result.jpg", result)
print("debug_result.jpg 저장 완료")
print(f"탐지된 창문: {len(final_boxes)}개")
