"""
YOLOv8 기반 창문 탐지 모듈.
pretrained 모델로 먼저 테스트하고, 이후 창문 데이터셋으로 fine-tuning 가능.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class DetectedWindow:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    crop: np.ndarray


class WindowDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None

        if YOLO_AVAILABLE:
            model_file = Path(model_path)
            if model_file.exists():
                self.model = YOLO(str(model_file))
            else:
                # weights 없으면 yolov8n 자동 다운로드
                Path("ml/weights").mkdir(parents=True, exist_ok=True)
                self.model = YOLO("yolov8n.pt")
                self.model.save(str(model_file))

    def detect(self, frame: np.ndarray) -> list[DetectedWindow]:
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_fallback(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[DetectedWindow]:
        results = self.model(frame, verbose=False)[0]
        windows = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            crop = frame[y:y+h, x:x+w]
            if crop.size > 0:
                windows.append(DetectedWindow(bbox=(x, y, w, h), confidence=conf, crop=crop))

        return windows

    def _detect_fallback(self, frame: np.ndarray) -> list[DetectedWindow]:
        """YOLO 없을 때 OpenCV 기반 단순 사각형 탐지 (테스트용)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        windows = []
        h_frame, w_frame = frame.shape[:2]
        min_area = (w_frame * h_frame) * 0.005  # 전체 화면의 0.5% 이상

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / h if h > 0 else 0
                if 0.4 < aspect < 2.5:
                    crop = frame[y:y+h, x:x+w]
                    if crop.size > 0:
                        windows.append(
                            DetectedWindow(bbox=(x, y, w, h), confidence=0.5, crop=crop)
                        )

        return windows


if __name__ == "__main__":
    import sys
    import os

    VIDEO_PATH  = "test_video.MOV"
    SAMPLE_RATE = 30        # 매 N 프레임마다 1장 처리
    OUTPUT_DIR  = "results/detector_test"
    CONF_THRESH = 0.3

    if not os.path.exists(VIDEO_PATH):
        print(f"[오류] 파일을 찾을 수 없습니다: {VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detector = WindowDetector(
        model_path="ml/weights/yolov8n.pt",
        confidence_threshold=CONF_THRESH,
    )
    mode = "YOLO" if detector.model is not None else "OpenCV fallback"
    print(f"[탐지 모드] {mode}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[영상 정보] 총 {total_frames}프레임 / {fps:.1f}fps")

    frame_idx      = 0
    sampled_count  = 0
    total_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_RATE == 0:
            windows = detector.detect(frame)
            total_detected += len(windows)

            # 바운딩박스 + 신뢰도 오버레이
            vis = frame.copy()
            for i, win in enumerate(windows):
                x, y, w, h = win.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"#{i+1} {win.confidence:.2f}"
                cv2.putText(vis, label, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # 창문 크롭 저장
                crop_path = os.path.join(
                    OUTPUT_DIR, f"f{frame_idx:05d}_w{i+1}.jpg"
                )
                cv2.imwrite(crop_path, win.crop)

            # 오버레이 프레임 저장
            info = f"frame={frame_idx}  windows={len(windows)}  mode={mode}"
            cv2.putText(vis, info, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, info, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

            out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, vis)

            elapsed = frame_idx / fps
            print(f"  [{elapsed:6.1f}s | 프레임 {frame_idx:5d}] 창문 {len(windows)}개")
            sampled_count += 1

        frame_idx += 1

    cap.release()

    print(f"\n[결과] 처리 프레임: {sampled_count} / 탐지된 창문(누적): {total_detected}")
    print(f"[저장] {OUTPUT_DIR}/")
