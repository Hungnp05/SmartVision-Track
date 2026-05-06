import argparse
import cv2
import yaml
import logging
import sys
import time
import math
import numpy as np
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12s] %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/runtime.log", mode="a", encoding="utf-8"),
    ],
)

from core.pipeline import SmartVisionPipeline


def parse_args():
    p = argparse.ArgumentParser(description="SmartVision-Track")
    p.add_argument("--source",   default=None)
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--headless", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_loading_screen(frame: np.ndarray, message: str, dots: int, elapsed: float):
    """Vẽ loading overlay lên camera frame — non-blocking."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    bw, bh = 480, 120
    bx = (w - bw) // 2
    by = (h - bh) // 2
    cv2.rectangle(frame, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), (0, 200, 100), 2)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (15, 15, 15), -1)

    # Spinner
    cx_s = bx + 40
    cy_s = by + bh // 2
    angle = (elapsed * 300) % 360
    for i in range(8):
        a  = math.radians(angle + i * 45)
        px = int(cx_s + 16 * math.cos(a))
        py = int(cy_s + 16 * math.sin(a))
        c  = int(255 * (i + 1) / 8)
        cv2.circle(frame, (px, py), 3, (0, c, 60), -1)

    dot_str = "." * (dots % 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Dang khoi dong{dot_str}",
                (bx + 70, by + 35), font, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, message,
                (bx + 70, by + 65), font, 0.50, (0, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{elapsed:.1f}s",
                (bx + 70, by + 90), font, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    # Pulse bar
    bar_w = bw - 20
    bar_x = bx + 10
    bar_y = by + bh - 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (40, 40, 40), -1)
    pulse = int((math.sin(elapsed * 2) + 1) / 2 * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + pulse, bar_y + 6), (0, 200, 80), -1)
    return frame


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    logger = logging.getLogger("main")

    source = args.source
    if source is None:
        source = cfg["source"]["webcam_index"]
    elif str(source).isdigit():
        source = int(source)

    logger.info("=" * 55)
    logger.info("  SmartVision-Track")
    logger.info("=" * 55)

    pipeline = SmartVisionPipeline(cfg)

    #  Bước 1: Mở camera nhanh 
    try:
        w, h = pipeline.open_source(source)
        logger.info(f"Camera opened: {w}x{h}")
    except RuntimeError as e:
        logger.error(f"Loi mo camera: {e}")
        sys.exit(1)

    window_name = cfg["display"]["window_name"]
    headless    = args.headless

    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))

    #  Bước 2: Bắt đầu background threads 
    pipeline.start_init_async()   # Load models trong background
    pipeline.start_pipeline()     # Capture thread bắt đầu đọc camera

    #  Bước 3: Loading screen (main thread responsive) 
    init_start      = time.time()
    dots            = 0
    dot_timer       = time.time()
    last_raw_frame  = None

    loading_messages = [
        "Loading YOLOv8 model...",
        "Loading DeepSORT Re-ID (MobileNet)...",
        "Warming up GPU...",
        "Almost ready...",
    ]

    logger.info("Loading screen active — window responsive trong khi load")

    while not pipeline.is_ready():
        elapsed = time.time() - init_start

        # Lấy frame camera mới nhất để hiển thị live
        try:
            if pipeline.raw_queue.queue:
                last_raw_frame = list(pipeline.raw_queue.queue)[-1].copy()
        except Exception:
            pass

        display = (last_raw_frame.copy()
                   if last_raw_frame is not None
                   else np.zeros((h, w, 3), dtype=np.uint8))

        msg_idx = min(int(elapsed / 4), len(loading_messages) - 1)
        draw_loading_screen(display, loading_messages[msg_idx], dots, elapsed)

        if not headless:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF  # 30ms loop → UI không bị đóng băng
            if key == ord("q"):
                logger.info("User quit during loading")
                pipeline.stop()
                cv2.destroyAllWindows()
                return

        if time.time() - dot_timer > 0.5:
            dots += 1
            dot_timer = time.time()

        # Kiểm tra lỗi nghiêm trọng
        if pipeline._init_error:
            logger.error(f"Model load failed: {pipeline._init_error}")
            if not headless:
                err = display.copy()
                cv2.putText(err, f"ERROR: Check console for details",
                            (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, err)
                cv2.waitKey(4000)
                cv2.destroyAllWindows()
            pipeline.stop()
            sys.exit(1)

    logger.info(f"Models loaded in {time.time() - init_start:.1f}s — starting realtime")

    #  Bước 4: Realtime inference loop 
    logger.info("Keys: [q] quit  [r] reset counts  [s] screenshot")
    screenshot_count = 0

    while True:
        frame = pipeline.get_frame()

        if frame is None:
            if not pipeline.running:
                logger.info("Stream ended")
                break
            if not headless:
                cv2.waitKey(5)
            continue

        if not headless:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit")
                break
            elif key == ord("r"):
                pipeline.counter.reset_counts()
                logger.info("Counts reset")
            elif key == ord("s"):
                fname = f"logs/screenshot_{screenshot_count:04d}.jpg"
                cv2.imwrite(fname, frame)
                logger.info(f"Screenshot saved: {fname}")
                screenshot_count += 1

    #  Cleanup 
    pipeline.stop()
    if not headless:
        cv2.destroyAllWindows()

    stats = pipeline.get_stats()
    logger.info("=" * 40)
    logger.info(f"  Tong vao : {stats['count_in']}")
    logger.info(f"  Tong ra  : {stats['count_out']}")
    logger.info(f"  Con trong: {stats['occupancy']}")
    logger.info(f"  Frames   : {stats['frame_count']}")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
