"""
main.py
SmartVision-Track - Entry Point

Usage:
    python main.py                      # webcam index 0
    python main.py --source 1           # webcam index 1
    python main.py --source rtsp://...  # RTSP stream
    python main.py --source video.mp4   # file video
    python main.py --config custom.yaml # config tùy chỉnh
"""

import argparse
import cv2
import yaml
import logging
import sys
import time
from pathlib import Path

# Setup logging trước khi import modules khác
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-10s] %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/runtime.log", mode="a", encoding="utf-8"),
    ],
)
Path("logs").mkdir(exist_ok=True)

from core.pipeline import SmartVisionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="SmartVision-Track")
    parser.add_argument("--source", default=None,
                        help="Video source: 0 (webcam) | rtsp://... | video.mp4")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Hiển thị cửa sổ OpenCV")
    parser.add_argument("--headless", action="store_true",
                        help="Chạy không cần GUI (chỉ log)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Override source từ CLI args
    if args.source is not None:
        src = args.source
        if src.isdigit():
            src = int(src)
        cfg["source"]["webcam_index"] = src
    else:
        src = cfg["source"]["webcam_index"]

    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("  SmartVision-Track - Khởi động hệ thống")
    logger.info("=" * 60)

    pipeline = SmartVisionPipeline(cfg)

    try:
        w, h = pipeline.initialize(src)
        logger.info(f"Source: {src} | Resolution: {w}×{h}")
        logger.info(f"Device: {pipeline.detector.get_device_info()}")
        logger.info("Press 'q' để thoát, 'r' để reset counts, 's' để screenshot")

        pipeline.start()

        headless = args.headless
        window_name = cfg["display"]["window_name"]

        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))

        screenshot_count = 0

        while True:
            frame = pipeline.get_frame(timeout=0.05)
            if frame is None:
                if not pipeline.running:
                    logger.info("Stream đã kết thúc")
                    break
                continue

            if not headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("User requested quit")
                    break
                elif key == ord("r"):
                    pipeline.counter.reset_counts()
                    logger.info("Counts reset")
                elif key == ord("s"):
                    fname = f"logs/screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(fname, frame)
                    logger.info(f"Screenshot saved: {fname}")
                    screenshot_count += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Lỗi nghiêm trọng: {e}")
    finally:
        pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()

        stats = pipeline.get_stats()
        logger.info("=" * 40)
        logger.info("  KẾT QUẢ PHIÊN:")
        logger.info(f"  Tổng vào : {stats['count_in']}")
        logger.info(f"  Tổng ra  : {stats['count_out']}")
        logger.info(f"  Còn trong: {stats['occupancy']}")
        logger.info("=" * 40)


if __name__ == "__main__":
    main()
