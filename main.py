import argparse
import cv2
import yaml
import logging
import sys
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Logging
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

from core.pipeline        import SmartVisionPipeline
from core.video_processor import VideoProcessor


#  CLI

def parse_args():
    p = argparse.ArgumentParser(description="SmartVision-Track")
    p.add_argument("--mode",    choices=["live", "video"], default="live",
                   help="live=webcam/RTSP realtime | video=xử lý file video")
    p.add_argument("--source",  default=None,
                   help="[live] 0=webcam, rtsp://..., v.v.")
    p.add_argument("--video",   default=None,
                   help="[video] đường dẫn tới file video cần xử lý")
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--headless", action="store_true",
                   help="Không hiển thị cửa sổ OpenCV")
    p.add_argument("--no-annotated", action="store_true",
                   help="[video] Không lưu video annotated")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_cfg(cfg: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


#  LOADING SCREEN (dùng cho chế độ live)

def draw_loading(frame: np.ndarray, msg: str, elapsed: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    ov = out.copy()
    cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)

    bw, bh = min(500, w-40), 120
    bx, by = (w-bw)//2, (h-bh)//2
    cv2.rectangle(out, (bx,by), (bx+bw,by+bh), (12,12,12), -1)
    cv2.rectangle(out, (bx-1,by-1), (bx+bw+1,by+bh+1), (0,200,90), 2)

    cxs, cys = bx+42, by+bh//2
    angle = (elapsed * 320) % 360
    for i in range(10):
        a  = math.radians(angle + i*36)
        px = int(cxs + 17*math.cos(a))
        py = int(cys + 17*math.sin(a))
        cv2.circle(out, (px,py), 3, (0, int(255*(i+1)/10), 60), -1)

    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "SmartVision-Track  —  Dang tai model...",
                (bx+72, by+30), f, 0.58, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(out, msg,
                (bx+72, by+60), f, 0.50, (0,210,100),  1, cv2.LINE_AA)
    cv2.putText(out, f"Elapsed: {elapsed:.1f}s   (co the mat 10-20s lan dau)",
                (bx+72, by+88), f, 0.40, (110,110,110), 1, cv2.LINE_AA)

    bpw = bw-20; bpx = bx+10; bpy = by+bh-14
    cv2.rectangle(out, (bpx,bpy), (bpx+bpw,bpy+7), (35,35,35), -1)
    pulse = int((math.sin(elapsed*2.5)+1)/2 * bpw)
    cv2.rectangle(out, (bpx,bpy), (bpx+pulse,bpy+7), (0,200,80), -1)
    return out


#  MODE 1: LIVE (webcam / RTSP)

def run_live(args, cfg):
    src = args.source
    if src is None:
        src = cfg["source"]["webcam_index"]
    elif str(src).isdigit():
        src = int(src)

    logger.info("=" * 60)
    logger.info("  CHE DO: LIVE")
    logger.info(f"  Source : {src}")
    logger.info(f"  Output : data/live_track_<ngay>.csv")
    logger.info("=" * 60)

    pipeline = SmartVisionPipeline(cfg)
    try:
        w, h = pipeline.open_source(src)
    except RuntimeError as e:
        logger.error(f"Khong mo duoc camera: {e}")
        sys.exit(1)

    win      = cfg["display"]["window_name"]
    headless = args.headless

    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(w, 1280), min(h, 720))
        logger.info("Phim tat: Q=thoat  R=reset  S=screenshot  +/-=di line  SPACE=pause")

    pipeline.start_init_async()
    pipeline.start_pipeline()

    # Loading screen
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
        try:
            q = pipeline.raw_queue.queue
            if q:
                last_raw = list(q)[-1].copy()
        except Exception:
            pass

        base    = last_raw.copy() if last_raw is not None else np.zeros((h,w,3), dtype=np.uint8)
        msg     = load_msgs[min(int(elapsed/5), len(load_msgs)-1)]
        display = draw_loading(base, msg, elapsed)

        if not headless:
            cv2.imshow(win, display)
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                pipeline.stop()
                cv2.destroyAllWindows()
                return

        if pipeline._init_error:
            logger.error(f"Model load FAILED: {pipeline._init_error}")
            pipeline.stop()
            sys.exit(1)

    logger.info(f" Ready in {time.time()-t0:.1f}s — BAT DAU LIVE")

    # Realtime loop
    screenshot_n = 0
    paused       = False
    last_frame   = None
    LINE_STEP    = 0.05

    while True:
        frame = pipeline.get_frame()
        if frame is not None:
            last_frame = frame
        else:
            frame = last_frame
            if not pipeline.running:
                logger.info("Stream ket thuc")
                break
            ai_alive = any(t.name == "ai" and t.is_alive() for t in pipeline._threads)
            if not ai_alive:
                logger.error("AI thread da chet! Xem loi o terminal.")
                break

        if frame is None:
            if not headless:
                cv2.waitKey(10)
            continue

        if not headless:
            cv2.imshow(win, frame if not paused else _draw_paused(frame.copy(), w, h))
            key = cv2.waitKey(1) & 0xFF

            if   key in (ord("q"), 27):  break
            elif key == ord("r"):
                pipeline.counter.reset_counts()
                logger.info(">> Counts reset")
            elif key == ord("s"):
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"logs/screenshot_{ts}.jpg"
                cv2.imwrite(fname, frame)
                logger.info(f">> Screenshot: {fname}")
            elif key == ord(" "):
                paused = not paused
                logger.info(f">> {'PAUSED' if paused else 'RESUMED'}")
            elif key in (ord("+"), ord("=")):
                _shift_line(pipeline, cfg, args.config, dy=+LINE_STEP)
            elif key == ord("-"):
                _shift_line(pipeline, cfg, args.config, dy=-LINE_STEP)
            elif key == ord("]"):
                _shift_line(pipeline, cfg, args.config, dx=+LINE_STEP)
            elif key == ord("["):
                _shift_line(pipeline, cfg, args.config, dx=-LINE_STEP)

    pipeline.stop()
    if not headless:
        cv2.destroyAllWindows()

    stats = pipeline.get_stats()
    logger.info("=" * 45)
    logger.info("  KET QUA LIVE SESSION:")
    logger.info(f"    Tong vao  : {stats['count_in']}")
    logger.info(f"    Tong ra   : {stats['count_out']}")
    logger.info(f"    Con trong : {stats['occupancy']}")
    logger.info(f"    CSV       : data/live_track_{datetime.now().strftime('%Y-%m-%d')}.csv")
    logger.info("=" * 45)


def _draw_paused(frame, w, h):
    txt = "  PAUSED — nhan SPACE de tiep tuc  "
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    tx, ty = (w-tw)//2, h//2
    cv2.rectangle(frame, (tx-10,ty-th-10), (tx+tw+10,ty+10), (20,20,20), -1)
    cv2.putText(frame, txt, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,200,200), 2, cv2.LINE_AA)
    return frame


def _shift_line(pipeline, cfg, cfg_path, dx=0.0, dy=0.0):
    c  = cfg["counter"]
    ls = [max(0.0,min(1.0,c["line_start"][0]+dx)), max(0.0,min(1.0,c["line_start"][1]+dy))]
    le = [max(0.0,min(1.0,c["line_end"][0]+dx)),   max(0.0,min(1.0,c["line_end"][1]+dy))]
    c["line_start"], c["line_end"] = ls, le
    if pipeline.counter:
        pipeline.counter.update_line(tuple(ls), tuple(le))
    save_cfg(cfg, cfg_path)
    logger.info(f">> Line → start={ls} end={le}")


#  MODE 2: VIDEO FILE

def run_video(args, cfg):
    video_path = args.video
    if video_path is None:
        # Không có --video → hỏi đường dẫn
        logger.info("Khong co --video, nhap duong dan file video:")
        video_path = input("  Video path: ").strip().strip('"').strip("'")

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"File khong ton tai: {video_path}")
        sys.exit(1)

    save_annotated = not args.no_annotated

    logger.info("=" * 60)
    logger.info("  CHE DO: VIDEO")
    logger.info(f"  File   : {video_path}")
    logger.info(f"  Output : data/video_track_{video_path.stem}_<ts>.csv")
    logger.info(f"  Video annotated: {'Co' if save_annotated else 'Khong'}")
    logger.info("=" * 60)

    # Hiển thị preview window (nếu không headless)
    headless = args.headless
    win      = "SmartVision-Track — Video Processing"

    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)

    processor = VideoProcessor(cfg)

    last_progress_frame = [None]   # closure để hiển thị preview

    def progress_cb(frame_no, total_frames, fps_proc):
        if not headless and frame_no % 5 == 0:
            # Lấy frame hiện tại từ video để preview
            pct = frame_no / max(total_frames, 1)
            _show_video_progress_screen(
                win, frame_no, total_frames, fps_proc, pct,
                video_path.name
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt("User cancelled video processing")

    try:
        result = processor.process(
            video_path,
            save_annotated=save_annotated,
            progress_cb=progress_cb if not headless else None,
        )
    except KeyboardInterrupt:
        logger.info("Nguoi dung huy xu ly video")
        if not headless:
            cv2.destroyAllWindows()
        return

    if not headless:
        cv2.destroyAllWindows()

    # Kết quả
    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║           KET QUA XU LY VIDEO               ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info(f"║  File video  : {video_path.name[:42]:<42}║")
    logger.info(f"║  Thoi luong  : {str(timedelta(seconds=int(result['duration_sec']))):<42}║")
    logger.info(f"║  Tong frames : {result['total_frames']:<42}║")
    logger.info(f"║  Thoi gian XL: {result['process_time']:.1f}s{'':<38}║")
    logger.info(f"║  Tong IN     : {result['total_in']:<42}║")
    logger.info(f"║  Tong OUT    : {result['total_out']:<42}║")
    logger.info(f"║  Events      : {result['event_count']:<42}║")
    logger.info(f"║  CSV         : {str(result['csv_path'])[-42:]:<42}║")
    if result.get("annotated_path"):
        logger.info(f"║  Video XL    : {str(result['annotated_path'])[-42:]:<42}║")
    logger.info("╚══════════════════════════════════════════════╝")


def _show_video_progress_screen(win, frame_no, total_frames, fps_proc, pct, filename):
    """Hiển thị màn hình tiến độ xử lý video trong OpenCV window."""
    h, w = 540, 960
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Title
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "SmartVision-Track — Dang xu ly video...",
                (30, 50), f, 0.9, (0, 210, 100), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"File: {filename}",
                (30, 90), f, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h2 = 30, 150, w-60, 40
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h2), (40,40,40), -1)
    filled = int(pct * bar_w)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h2), (0,200,80), -1)
    cv2.rectangle(canvas, (bar_x-1, bar_y-1), (bar_x+bar_w+1, bar_y+bar_h2+1), (80,80,80), 1)

    pct_txt = f"{pct*100:.1f}%"
    (tw,_),_ = cv2.getTextSize(pct_txt, f, 0.7, 2)
    cv2.putText(canvas, pct_txt, (bar_x + bar_w//2 - tw//2, bar_y+28),
                f, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # Stats
    stats = [
        ("Frame",    f"{frame_no} / {total_frames}"),
        ("Toc do",   f"{fps_proc:.1f} FPS"),
        ("Tien do",  f"{pct*100:.1f}%"),
    ]
    if fps_proc > 0 and total_frames > 0:
        eta = (total_frames - frame_no) / fps_proc
        stats.append(("ETA", str(timedelta(seconds=int(eta)))))

    for i, (label, val) in enumerate(stats):
        x = 30 + i * 230
        y = 240
        cv2.putText(canvas, label, (x, y),       f, 0.5,  (120,120,120), 1, cv2.LINE_AA)
        cv2.putText(canvas, val,   (x, y+32),     f, 0.75, (220,220,220), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Nhan Q de huy...",
                (30, h-30), f, 0.45, (100,100,100), 1, cv2.LINE_AA)

    cv2.imshow(win, canvas)


#  ENTRY POINT

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    if args.mode == "video":
        run_video(args, cfg)
    else:
        run_live(args, cfg)


if __name__ == "__main__":
    main()