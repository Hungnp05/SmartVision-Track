import cv2
import yaml
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

from core.detector import PersonDetector
from core.tracker  import PersonTracker
from core.counter  import CrossingLineCounter
from utils.logger  import VideoTrackLogger
from utils.drawing import Drawer

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Xử lý 1 file video, xuất CSV + video annotated (tuỳ chọn)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def process(
        self,
        video_path: str,
        save_annotated: bool = True,
        progress_cb=None,          # callback(frame_no, total_frames, fps_proc)
    ) -> dict:
        """
        Xử lý video file đồng bộ.

        Args:
            video_path:     đường dẫn file video (.mp4, .avi, .mov, ...)
            save_annotated: lưu video đã annotate hay không
            progress_cb:    hàm callback tiến độ (optional)

        Returns:
            dict với: csv_path, annotated_path, total_in, total_out,
                      total_frames, duration_sec, events
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video không tồn tại: {video_path}")

        logger.info(f"=== VideoProcessor: {video_path.name} ===")

        # Mở video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / video_fps

        logger.info(
            f"  Resolution : {frame_w}x{frame_h}"
            f"  FPS        : {video_fps:.1f}"
            f"  Frames     : {total_frames}"
            f"  Duration   : {timedelta(seconds=int(duration_sec))}"
        )

        # Khởi tạo components
        logger.info("  Loading YOLOv8...")
        detector = PersonDetector(self.cfg["detection"])

        logger.info("  Loading DeepSORT...")
        tracker  = PersonTracker(self.cfg["tracking"])

        logger.info("  Setting up counter & logger...")
        counter  = CrossingLineCounter(self.cfg["counter"], frame_w, frame_h)
        vlogger  = VideoTrackLogger(video_path)
        drawer   = Drawer(self.cfg["display"], frame_w, frame_h)

        # VideoWriter (annotated output)
        out_writer   = None
        annotated_path = None
        if save_annotated:
            annotated_path = (
                Path("data") /
                f"video_annotated_{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            Path("data").mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(
                str(annotated_path), fourcc, video_fps, (frame_w, frame_h)
            )
            logger.info(f"  Annotated output: {annotated_path}")

        # Processing loop
        logger.info("  Bắt đầu xử lý...")
        frame_no   = 0
        t_start    = time.time()
        fps_window = []     # rolling window để tính FPS xử lý

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            t_frame  = time.time()

            # Detect + Track + Count
            detections = detector.detect(frame)
            tracks     = tracker.update(detections, frame)
            events     = counter.process_tracks(tracks)

            # Ghi events vào CSV
            for ev in events:
                vlogger.log_event(ev, frame_no, video_fps)

            # Lấy stats để vẽ
            stats = counter.get_stats()
            stats["fps"] = fps_window[-1] if fps_window else 0

            # Draw & write
            if out_writer:
                annotated = drawer.draw_all(frame, tracks, events, counter, stats)
                # Thêm progress bar vào video
                _draw_video_progress(annotated, frame_no, total_frames, video_fps)
                out_writer.write(annotated)

            # Tính FPS xử lý (rolling 10 frames)
            elapsed_frame = time.time() - t_frame
            fps_proc = 1.0 / elapsed_frame if elapsed_frame > 0 else 0
            fps_window.append(fps_proc)
            if len(fps_window) > 10:
                fps_window.pop(0)
            avg_fps = sum(fps_window) / len(fps_window)

            # Flush CSV mỗi 100 frames
            if frame_no % 100 == 0:
                vlogger.flush()

            # Log tiến độ mỗi 10%
            if total_frames > 0 and frame_no % max(1, total_frames // 10) == 0:
                pct     = frame_no / total_frames * 100
                elapsed = time.time() - t_start
                eta     = (elapsed / frame_no) * (total_frames - frame_no)
                logger.info(
                    f"  [{pct:5.1f}%] frame {frame_no}/{total_frames} | "
                    f"{avg_fps:.1f} FPS | ETA {timedelta(seconds=int(eta))}"
                )

            # Progress callback (cho UI)
            if progress_cb:
                progress_cb(frame_no, total_frames, avg_fps)

        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()

        csv_path = vlogger.close()

        total_time = time.time() - t_start
        logger.info("=" * 50)
        logger.info(f"  XONG: {frame_no} frames trong {total_time:.1f}s")
        logger.info(f"  Tổng IN  : {vlogger.total_in}")
        logger.info(f"  Tổng OUT : {vlogger.total_out}")
        logger.info(f"  Events   : {vlogger.event_count}")
        logger.info(f"  CSV      : {csv_path}")
        if annotated_path:
            logger.info(f"  Video    : {annotated_path}")
        logger.info("=" * 50)

        return {
            "csv_path":       csv_path,
            "annotated_path": annotated_path,
            "total_in":       vlogger.total_in,
            "total_out":      vlogger.total_out,
            "event_count":    vlogger.event_count,
            "total_frames":   frame_no,
            "duration_sec":   duration_sec,
            "process_time":   total_time,
        }


def _draw_video_progress(frame, frame_no: int, total_frames: int, fps: float):
    """Vẽ thanh tiến độ ở dưới cùng video annotated."""
    h, w = frame.shape[:2]
    bar_h = 18
    y     = h - bar_h

    # Background
    cv2.rectangle(frame, (0, y), (w, h), (15, 15, 15), -1)

    # Progress bar
    if total_frames > 0:
        prog = int(frame_no / total_frames * w)
        cv2.rectangle(frame, (0, y + 2), (prog, h - 2), (0, 200, 80), -1)

    # Text: frame / total và timecode
    video_time = frame_no / max(fps, 1)
    tc  = str(timedelta(seconds=int(video_time)))
    dur = str(timedelta(seconds=int(total_frames / max(fps, 1))))
    txt = f"  {tc} / {dur}   frame {frame_no}/{total_frames}"
    cv2.putText(frame, txt, (4, h - 4),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (200, 200, 200), 1, cv2.LINE_AA)