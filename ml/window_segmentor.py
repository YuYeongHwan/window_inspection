"""
건물 외벽 창문 자동 감지 및 세그멘테이션 모듈.

탐지 방법 (method 파라미터):
  "sam"    : SAM(Segment Anything Model, mobile_sam.pt) + 창문 휴리스틱 필터
             마스크를 직접 학습 없이 생성 → 형태/크기/색상 기준으로 창문 후보 선별
  "opencv" : 엣지 기반 사각형 탐지 (모델 불필요, 가장 빠른 fallback)

SAM 모델 가중치는 첫 실행 시 ultralytics가 자동 다운로드합니다.
  mobile_sam.pt ≈ 40 MB  (기본, 빠름)
  sam_b.pt      ≈ 375 MB (정확도 높음)

사용 예시:
  from ml.window_segmentor import WindowSegmentor

  seg = WindowSegmentor(method="sam")
  result = seg.process_image("building.jpg")
  print(result["windows"])

  result = seg.process_video("building.mp4", sample_rate=30)
  print(result["total_windows_detected"])
"""

from __future__ import annotations

import cv2
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

try:
    from ultralytics import SAM
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class WindowCandidate:
    id: int
    frame_number: int
    bbox: Dict[str, int]          # {"x", "y", "w", "h"}
    polygon: List[List[int]]      # [[x,y], ...]
    confidence: float             # 0.0 ~ 1.0
    area: int                     # 픽셀 단위 면적

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────

class WindowSegmentor:
    """
    건물 외벽 영상에서 창문 영역을 자동 감지·세그멘테이션.

    Parameters
    ----------
    method : "sam" | "opencv"
    model_path : SAM 가중치 경로 (없으면 자동 다운로드)
    min_area_ratio : 창문으로 인정할 최소 면적 비율 (이미지 전체 대비)
    max_area_ratio : 창문으로 인정할 최대 면적 비율
    min_rectangularity : 바운딩박스 채움 비율 하한 (사각형 유사도)
    min_solidity : 볼록 hull 대비 채움 비율 하한
    output_dir : 시각화 이미지·JSON 저장 경로
    """

    def __init__(
        self,
        method: str = "sam",
        model_path: str = "mobile_sam.pt",
        min_area_ratio: float = 0.003,
        max_area_ratio: float = 0.18,
        min_rectangularity: float = 0.60,
        min_solidity: float = 0.65,
        output_dir: str = "results/segmentation",
    ) -> None:
        self.method = method if SAM_AVAILABLE or method == "opencv" else "opencv"
        self.model_path = model_path
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_rectangularity = min_rectangularity
        self.min_solidity = min_solidity
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._sam: Optional[SAM] = None
        if self.method == "sam" and SAM_AVAILABLE:
            self._sam = SAM(model_path)  # 가중치 없으면 자동 다운로드
            print(f"[WindowSegmentor] SAM 모델 로드 완료: {model_path}")
        else:
            print("[WindowSegmentor] OpenCV fallback 모드로 실행합니다.")

    # ──────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        단일 이미지를 처리합니다.

        Returns
        -------
        dict with keys: source, windows, output_image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

        candidates = self._detect(image, frame_number=0)

        stem = Path(image_path).stem
        vis_path = self.output_dir / f"{stem}_overlay.jpg"
        self._save_visualization(image, candidates, str(vis_path))

        result = {
            "source": image_path,
            "total_windows_detected": len(candidates),
            "output_image": str(vis_path),
            "windows": [c.to_dict() for c in candidates],
        }

        json_path = self.output_dir / f"{stem}_result.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[완료] 창문 {len(candidates)}개 탐지 → {vis_path}")
        return result

    def process_video(
        self,
        video_path: str,
        sample_rate: int = 30,
    ) -> Dict[str, Any]:
        """
        동영상을 처리합니다. sample_rate 프레임마다 1장씩 분석합니다.

        Returns
        -------
        dict with keys: source, total_frames, processed_frames,
                        total_windows_detected, output_images, windows
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"동영상을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        stem = Path(video_path).stem
        frame_out_dir = self.output_dir / stem
        frame_out_dir.mkdir(parents=True, exist_ok=True)

        all_candidates: List[WindowCandidate] = []
        vis_paths: List[str] = []
        global_id = 1
        frame_idx = 0

        print(f"[WindowSegmentor] 총 {total_frames}프레임, {sample_rate}프레임마다 처리")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                candidates = self._detect(frame, frame_number=frame_idx)

                for c in candidates:
                    c.id = global_id
                    global_id += 1

                all_candidates.extend(candidates)

                vis_path = frame_out_dir / f"frame_{frame_idx:06d}_overlay.jpg"
                self._save_visualization(frame, candidates, str(vis_path))
                vis_paths.append(str(vis_path))

                elapsed_sec = frame_idx / fps
                print(
                    f"  [{elapsed_sec:6.1f}s | 프레임 {frame_idx:5d}] "
                    f"창문 {len(candidates)}개 탐지"
                )

            frame_idx += 1

        cap.release()

        result: Dict[str, Any] = {
            "source": video_path,
            "total_frames": total_frames,
            "processed_frames": len(vis_paths),
            "total_windows_detected": len(all_candidates),
            "output_images": vis_paths,
            "windows": [c.to_dict() for c in all_candidates],
        }

        json_path = self.output_dir / f"{stem}_result.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"\n[완료] 총 {len(all_candidates)}개 창문 탐지 "
            f"({len(vis_paths)}프레임 처리) → {json_path}"
        )
        return result

    # ──────────────────────────────────────────
    # 탐지 내부 로직
    # ──────────────────────────────────────────

    def _detect(
        self, image: np.ndarray, frame_number: int
    ) -> List[WindowCandidate]:
        if self._sam is not None:
            return self._detect_sam(image, frame_number)
        return self._detect_opencv(image, frame_number)

    def _detect_sam(
        self, image: np.ndarray, frame_number: int
    ) -> List[WindowCandidate]:
        """
        SAM 자동 마스크 생성 후 창문 휴리스틱 필터 적용.

        SAM은 이미지 내 모든 "물체" 마스크를 생성하고,
        _score_mask()로 창문일 가능성을 0~1로 평가해 임계값 이상만 유지.
        """
        h, w = image.shape[:2]
        image_area = h * w

        try:
            results = self._sam(image, verbose=False)
        except Exception as e:
            print(f"[SAM 오류] {e} → OpenCV fallback 사용")
            return self._detect_opencv(image, frame_number)

        candidates: List[WindowCandidate] = []

        for result in results:
            if result.masks is None:
                continue

            masks_data = result.masks.data      # (N, H, W) tensor
            masks_xy = result.masks.xy           # list of (K, 2) arrays (polygon)

            for mask_tensor, poly_xy in zip(masks_data, masks_xy):
                mask = mask_tensor.cpu().numpy().astype(np.uint8)

                # 마스크 크기가 이미지와 다를 경우 리사이즈
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                score = self._score_mask(mask, image, image_area)
                if score is None:
                    continue

                # 바운딩박스
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(cnt)

                # 폴리곤 단순화 (SAM xy 또는 윤곽선 사용)
                if len(poly_xy) >= 4:
                    polygon = _simplify_polygon(poly_xy)
                else:
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    polygon = approx.reshape(-1, 2).tolist()

                candidates.append(
                    WindowCandidate(
                        id=len(candidates) + 1,
                        frame_number=frame_number,
                        bbox={"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                        polygon=polygon,
                        confidence=round(score, 4),
                        area=int(cv2.contourArea(cnt)),
                    )
                )

        # 겹치는 후보 제거 (IoU 기반)
        candidates = _nms_candidates(candidates, iou_threshold=0.5)
        return candidates

    def _detect_opencv(
        self, image: np.ndarray, frame_number: int
    ) -> List[WindowCandidate]:
        """
        OpenCV 엣지 + 윤곽선 기반 사각형 탐지 (SAM 없을 때 fallback).

        파이프라인:
          그레이스케일 → 블러 → Canny → 팽창 → 윤곽선 추출
          → 사각형 근사 → 면적/종횡비 필터
        """
        h, w = image.shape[:2]
        image_area = h * w
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: List[WindowCandidate] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * self.min_area_ratio:
                continue
            if area > image_area * self.max_area_ratio:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if not (4 <= len(approx) <= 8):
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            if bh == 0:
                continue
            aspect = bw / bh
            if not (0.25 < aspect < 4.0):
                continue

            # 볼록 hull 솔리디티
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0 or (area / hull_area) < self.min_solidity:
                continue

            candidates.append(
                WindowCandidate(
                    id=len(candidates) + 1,
                    frame_number=frame_number,
                    bbox={"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                    polygon=approx.reshape(-1, 2).tolist(),
                    confidence=0.50,
                    area=int(area),
                )
            )

        candidates = _nms_candidates(candidates, iou_threshold=0.4)
        return candidates

    # ──────────────────────────────────────────
    # 창문 휴리스틱 스코어링
    # ──────────────────────────────────────────

    def _score_mask(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        image_area: int,
    ) -> Optional[float]:
        """
        마스크가 창문일 가능성을 0~1로 반환. 기준 미달이면 None.

        평가 항목 (가중 합산):
          rectangularity  0.40  — 바운딩박스 채움 비율 (창문은 사각형에 가까움)
          solidity        0.30  — 볼록성 (창문은 오목한 부분이 적음)
          color_score     0.30  — 색상 균일도 (유리면은 반사가 균일)
        """
        area = int(mask.sum())
        if area < image_area * self.min_area_ratio:
            return None
        if area > image_area * self.max_area_ratio:
            return None

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)

        # ── 솔리디티 ──────────────────────────
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            return None
        solidity = cv2.contourArea(cnt) / hull_area
        if solidity < self.min_solidity:
            return None

        # ── 사각형 유사도 ─────────────────────
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            return None
        rectangularity = area / (bw * bh)
        if rectangularity < self.min_rectangularity:
            return None

        # ── 종횡비 ───────────────────────────
        aspect = bw / bh
        if not (0.25 < aspect < 4.0):
            return None

        # ── 색상 균일도 ───────────────────────
        pixels = image[mask > 0].astype(np.float32)
        if len(pixels) == 0:
            return None
        # 채널별 표준편차 평균 → 낮을수록 균일
        channel_stds = np.std(pixels, axis=0)          # (3,)
        mean_std = float(np.mean(channel_stds))
        # 0~80 범위를 0~1 균일도로 변환
        color_score = 1.0 - min(mean_std / 80.0, 1.0)

        score = rectangularity * 0.40 + solidity * 0.30 + color_score * 0.30
        return float(np.clip(score, 0.0, 1.0))

    # ──────────────────────────────────────────
    # 시각화
    # ──────────────────────────────────────────

    def _save_visualization(
        self,
        image: np.ndarray,
        candidates: List[WindowCandidate],
        output_path: str,
    ) -> None:
        """
        시각화 이미지 저장.

        - 창문 외 영역: 그레이스케일 (외벽=회색)
        - 창문 영역:    원본 색상 + 초록 반투명 오버레이
        - 창문 테두리:  초록 실선
        - 창문 ID:      흰색 레이블
        - 좌상단:       탐지 통계 정보
        """
        h, w = image.shape[:2]

        # ── 배경: 전체를 그레이스케일(외벽=회색)로 ──
        gray_bgr = cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
        result = gray_bgr.copy()

        # ── 창문마다 원본 픽셀 복원 + 초록 오버레이 ──
        for c in candidates:
            pts = np.array(c.polygon, dtype=np.int32)

            # 마스크 생성
            win_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(win_mask, [pts], 255)

            # 원본 색상 복원
            result[win_mask > 0] = image[win_mask > 0]

            # 초록 반투명 레이어
            green_layer = result.copy()
            cv2.fillPoly(green_layer, [pts], (0, 200, 50))
            cv2.addWeighted(green_layer, 0.30, result, 0.70, 0, result)

            # 초록 테두리
            cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 60), thickness=2)

            # ID 레이블
            cx = c.bbox["x"] + c.bbox["w"] // 2
            cy = c.bbox["y"] + c.bbox["h"] // 2
            _draw_label(result, f"#{c.id}", cx, cy)

        # ── 통계 정보 ──
        method_str = "SAM" if self._sam else "OpenCV"
        info = f"Windows: {len(candidates)}  |  Method: {method_str}"
        cv2.rectangle(result, (0, 0), (len(info) * 11 + 10, 36), (0, 0, 0), -1)
        cv2.putText(
            result, info, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
        )

        cv2.imwrite(output_path, result)


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────

def _simplify_polygon(xy: np.ndarray, n_points: int = 20) -> List[List[int]]:
    """SAM이 돌려주는 세밀한 폴리곤을 n_points 이하로 단순화."""
    pts = xy.astype(np.int32).reshape(-1, 1, 2)
    epsilon = 0.01 * cv2.arcLength(pts, True)
    approx = cv2.approxPolyDP(pts, epsilon, True)
    # 여전히 너무 많으면 한 번 더
    if len(approx) > n_points:
        epsilon = 0.03 * cv2.arcLength(pts, True)
        approx = cv2.approxPolyDP(pts, epsilon, True)
    return approx.reshape(-1, 2).tolist()


def _box_iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    """두 bbox의 IoU(Intersection over Union)를 계산."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _nms_candidates(
    candidates: List[WindowCandidate], iou_threshold: float = 0.5
) -> List[WindowCandidate]:
    """
    Non-Maximum Suppression: 높은 confidence 후보를 우선 유지하고
    IoU가 threshold 이상인 낮은 confidence 후보를 제거.
    """
    sorted_cands = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    kept: List[WindowCandidate] = []

    for cand in sorted_cands:
        if all(_box_iou(cand.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(cand)

    # ID 재부여
    for i, c in enumerate(kept, start=1):
        c.id = i

    return kept


def _draw_label(image: np.ndarray, text: str, cx: int, cy: int) -> None:
    """이미지에 검정 배경 + 흰 텍스트 레이블을 그립니다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0 = cx - tw // 2 - 4
    y0 = cy - th - 4
    cv2.rectangle(image, (x0, y0), (x0 + tw + 8, cy + baseline), (0, 0, 0), -1)
    cv2.putText(
        image, text, (x0 + 4, cy - 2),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
