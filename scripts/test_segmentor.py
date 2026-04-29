"""
창문 세그멘테이션 테스트 스크립트.

사용법:
  # 이미지
  python scripts/test_segmentor.py --input building.jpg

  # 동영상 (스마트폰 촬영)
  python scripts/test_segmentor.py --input building.mp4 --sample-rate 30

  # OpenCV fallback (SAM 없이)
  python scripts/test_segmentor.py --input building.jpg --method opencv

  # SAM 정밀 모델
  python scripts/test_segmentor.py --input building.mp4 --model sam_b.pt
"""
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.window_segmentor import WindowSegmentor


def main() -> None:
    parser = argparse.ArgumentParser(description="창문 세그멘테이션 테스트")
    parser.add_argument("--input", required=True, help="이미지 또는 동영상 경로")
    parser.add_argument(
        "--method", choices=["sam", "opencv"], default="sam",
        help="탐지 방법 (기본: sam)"
    )
    parser.add_argument(
        "--model", default="mobile_sam.pt",
        help="SAM 모델 가중치 (기본: mobile_sam.pt)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=30,
        help="동영상: N 프레임마다 1장 처리 (기본: 30)"
    )
    parser.add_argument(
        "--output-dir", default="results/segmentation",
        help="결과 저장 폴더"
    )
    parser.add_argument(
        "--min-area", type=float, default=0.003,
        help="창문 최소 면적 비율 (기본: 0.003)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[오류] 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    seg = WindowSegmentor(
        method=args.method,
        model_path=args.model,
        min_area_ratio=args.min_area,
        output_dir=args.output_dir,
    )

    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        result = seg.process_image(args.input)
    elif ext in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
        result = seg.process_video(args.input, sample_rate=args.sample_rate)
    else:
        print(f"[오류] 지원하지 않는 형식입니다: {ext}")
        sys.exit(1)

    print("\n──────── 결과 요약 ────────")
    print(f"탐지된 창문 수: {result['total_windows_detected']}")
    if "processed_frames" in result:
        print(f"처리된 프레임:  {result['processed_frames']}")
    if result["windows"]:
        grades_info = [
            f"  #{w['id']:3d}  bbox={w['bbox']}  conf={w['confidence']:.2f}"
            for w in result["windows"][:10]
        ]
        print("창문 목록 (최대 10개):")
        print("\n".join(grades_info))
        if len(result["windows"]) > 10:
            print(f"  ... 외 {len(result['windows']) - 10}개")
    print("──────────────────────────")


if __name__ == "__main__":
    main()
