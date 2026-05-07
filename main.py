import argparse
import cv2
import yaml
import logging
import sys
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime

# Logging setup
Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-10s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/runtime.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

from core.pipeline import SmartVisionPipeline


#  CLI

def parse_args():
    p = argparse.ArgumentParser(description="SmartVision-Track")
    p.add_argument("--source",  default=None,
                   help="0=webcam, 1=webcam2, path/to/video.mp4, rtsp://...")
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--headless", action="store_true",
                   help="Chạy không GUI (chỉ log + CSV)")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_cfg(cfg: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


#  LOADING SCREEN

def draw_loading(frame: np.ndarray, msg: str, elapsed: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Dim
    ov = out.copy()
    cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)

    bw, bh = min(500, w-40), 120
    bx = (w - bw) // 2
    by = (h - bh) // 2

    # Box
    cv2.rectangle(out, (bx,by), (bx+bw, by+bh), (12,12,12), -1)
    cv2.rectangle(out, (bx-1,by-1), (bx+bw+1,by+bh+1), (0,200,90), 2)

    # Spinner
    cx_s, cy_s = bx+42, by+bh//2
    angle = (elapsed * 320) % 360
    for i in range(10):
        a  = math.radians(angle + i*36)
        px = int(cx_s + 17*math.cos(a))
        py = int(cy_s + 17*math.sin(a))
        c  = int(255 * (i+1) / 10)
        cv2.circle(out, (px,py), 3, (0, c, 60), -1)

    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "SmartVision-Track  —  Dang tai model...",
                (bx+72, by+30), f, 0.58, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(out, msg,
                (bx+72, by+60), f, 0.50, (0,210,100),  1, cv2.LINE_AA)
    cv2.putText(out, f"Elapsed: {elapsed:.1f}s   (co the mat 10-20s lan dau)",
                (bx+72, by+88), f, 0.40, (110,110,110), 1, cv2.LINE_AA)

    # Pulse bar
    bpw = bw - 20
    bpx = bx + 10
    bpy = by + bh - 14
    cv2.rectangle(out, (bpx,bpy), (bpx+bpw, bpy+7), (35,35,35), -1)
    pulse = int((math.sin(elapsed*2.5)+1)/2 * bpw)
    cv2.rectangle(out, (bpx,bpy), (bpx+pulse, bpy+7), (0,200,80), -1)

    return out


#  MAIN

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    # Resolve source
    src = args.source
    if src is None:
        src = cfg["source"]["webcam_index"]
    elif str(src).isdigit():
        src = int(src)

    logger.info("=" * 60)
    logger.info("  SmartVision-Track — khoi dong")
    logger.info(f"  Source : {src}")
    logger.info(f"  Model  : {cfg['detection']['model']}")
    logger.info(f"  Device : {cfg['detection']['device']}")
    logger.info("=" * 60)

    pipeline = SmartVisionPipeline(cfg)

    # 1. Mở camera (< 1 giây)
    try:
        w, h = pipeline.open_source(src)
        logger.info(f"Camera: {w}x{h}")
    except RuntimeError as e:
        logger.error(f"Khong mo duoc camera: {e}")
        sys.exit(1)

    win = cfg["display"]["window_name"]
    headless = args.headless

    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(w, 1280), min(h, 720))
        logger.info("Phim tat: Q=thoat  R=reset  S=screenshot  +/-=di line  H=help  SPACE=pause")

    # 2. Khởi động background threads
    pipeline.start_init_async()    # load YOLO + DeepSORT
    pipeline.start_pipeline()      # capture thread bắt đầu ngay

    # 3. Loading screen loop
    t0         = time.time()
    last_raw   = None
    load_msgs  = [
        "Loading YOLOv8...",
        "Loading DeepSORT Re-ID (MobileNet)...",
        "Warming up GPU (CUDA)...",
        "Almost ready...",
    ]

    while not pipeline.is_ready():
        elapsed = time.time() - t0

        # Snapshot raw frame mới nhất từ camera
        try:
            q = pipeline.raw_queue.queue
            if q:
                last_raw = list(q)[-1].copy()
        except Exception:
            pass

        base = (last_raw.copy()
                if last_raw is not None
                else np.zeros((h, w, 3), dtype=np.uint8))

        msg = load_msgs[min(int(elapsed / 5), len(load_msgs)-1)]
        display = draw_loading(base, msg, elapsed)

        if not headless:
            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):     # Q hoặc ESC
                pipeline.stop()
                cv2.destroyAllWindows()
                return

        if pipeline._init_error:
            logger.error(f"Model load FAILED: {pipeline._init_error}")
            if not headless:
                cv2.putText(display, "ERROR! Xem terminal de biet them.",
                            (20, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,0,230), 2, cv2.LINE_AA)
                cv2.imshow(win, display)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
            pipeline.stop()
            sys.exit(1)

    logger.info(f"✅ Models loaded in {time.time()-t0:.1f}s")
    logger.info("=== BAT DAU INFERENCE REALTIME ===")

    # 4. Realtime loop
    screenshot_n = 0
    paused       = False
    show_help    = True
    last_frame   = None      # frame cache khi không có frame mới

    # Tham số dịch line (bước 5%)
    LINE_STEP = 0.05

    while True:
        # Lấy frame mới nhất từ AI thread
        frame = pipeline.get_frame()

        if frame is not None:
            last_frame = frame
        else:
            # Không có frame mới → dùng frame cũ (không để trắng màn)
            frame = last_frame
            if not pipeline.running:
                logger.info("Stream ket thuc")
                break

        if frame is None:
            # Chưa có frame nào cả — chờ
            if not headless:
                cv2.waitKey(10)
            continue

        if not headless:
            if not paused:
                cv2.imshow(win, frame)
            else:
                # Pause: vẽ chữ PAUSED lên frame
                paused_frame = frame.copy()
                txt = "  PAUSED — nhan SPACE de tiep tuc  "
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                tx = (w - tw) // 2
                ty = h // 2
                cv2.rectangle(paused_frame, (tx-10, ty-th-10),
                              (tx+tw+10, ty+10), (20,20,20), -1)
                cv2.putText(paused_frame, txt, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,200,200), 2, cv2.LINE_AA)
                cv2.imshow(win, paused_frame)

            key = cv2.waitKey(1) & 0xFF

            # Phím tắt
            if key in (ord("q"), 27):          # Q / ESC = thoát
                break

            elif key == ord("r"):              # R = reset counts
                pipeline.counter.reset_counts()
                logger.info(">> Counts reset ve 0")

            elif key == ord("s"):              # S = screenshot
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"logs/screenshot_{ts}.jpg"
                cv2.imwrite(fname, frame)
                logger.info(f">> Screenshot: {fname}")
                screenshot_n += 1

            elif key == ord(" "):              # SPACE = pause/resume
                paused = not paused
                logger.info(f">> {'PAUSED' if paused else 'RESUMED'}")

            elif key == ord("h"):              # H = toggle help
                show_help = not show_help

            # Dịch crossing line
            elif key in (ord("+"), ord("=")):  # + = line xuống
                _shift_line(pipeline, cfg, args.config, dy=+LINE_STEP)

            elif key == ord("-"):              # - = line lên
                _shift_line(pipeline, cfg, args.config, dy=-LINE_STEP)

            elif key == ord("]"):              # ] = line phải
                _shift_line(pipeline, cfg, args.config, dx=+LINE_STEP)

            elif key == ord("["):              # [ = line trái
                _shift_line(pipeline, cfg, args.config, dx=-LINE_STEP)

    # Cleanup
    pipeline.stop()
    if not headless:
        cv2.destroyAllWindows()

    stats = pipeline.get_stats()
    logger.info("=" * 45)
    logger.info("  KET QUA PHIEN:")
    logger.info(f"    Tong vao  : {stats['count_in']}")
    logger.info(f"    Tong ra   : {stats['count_out']}")
    logger.info(f"    Con trong : {stats['occupancy']}")
    logger.info(f"    Total FPS : {stats['frame_count']} frames")
    logger.info("=" * 45)


#  HELPER: Dịch chuyển crossing line bằng phím tắt

def _shift_line(pipeline, cfg: dict, cfg_path: str,
                dx: float = 0.0, dy: float = 0.0):
    """Dịch line theo dx/dy relative, cập nhật pipeline & config."""
    counter_cfg = cfg["counter"]
    ls = list(counter_cfg["line_start"])
    le = list(counter_cfg["line_end"])

    ls[0] = max(0.0, min(1.0, ls[0] + dx))
    ls[1] = max(0.0, min(1.0, ls[1] + dy))
    le[0] = max(0.0, min(1.0, le[0] + dx))
    le[1] = max(0.0, min(1.0, le[1] + dy))

    counter_cfg["line_start"] = ls
    counter_cfg["line_end"]   = le

    # Cập nhật ngay vào pipeline đang chạy
    if pipeline.counter:
        pipeline.counter.update_line(tuple(ls), tuple(le))

    # Lưu config để giữ lại sau khi restart
    save_cfg(cfg, cfg_path)
    logger.info(f">> Line moved → start={ls}  end={le}")


if __name__ == "__main__":
    main()
