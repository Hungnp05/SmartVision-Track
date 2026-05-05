"""
core/pipeline.py
Async Pipeline Orchestrator.

Kiến trúc threading:
- Thread 1 (Capture):  Đọc frame từ camera/RTSP liên tục
- Thread 2 (AI):       YOLOv8 detection + DeepSORT tracking
- Thread 3 (Main/UI):  Nhận processed frame, vẽ, hiển thị OpenCV
- Thread 4 (Logger):   Ghi events ra CSV/SQLite theo batch

Queue giữa các thread:
  raw_queue:       Capture → AI
  processed_queue: AI → UI
  event_queue:     AI → Logger
"""

import threading
import queue
import time
import cv2
import numpy as np
import logging
from datetime import datetime

from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.counter import CrossingLineCounter
from utils.logger import EventLogger
from utils.drawing import Drawer

logger = logging.getLogger(__name__)


class SmartVisionPipeline:
    """Pipeline chính của SmartVision-Track."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.running = False

        # Queues
        maxsize = cfg["performance"]["queue_maxsize"]
        self.raw_queue       = queue.Queue(maxsize=maxsize)
        self.processed_queue = queue.Queue(maxsize=maxsize)
        self.event_queue     = queue.Queue(maxsize=500)

        # Shared state (protected bởi lock)
        self._lock = threading.Lock()
        self._stats = {
            "fps": 0.0,
            "count_in": 0,
            "count_out": 0,
            "occupancy": 0,
            "active_tracks": 0,
            "frame_count": 0,
        }

        # Components (khởi tạo sau khi biết frame size)
        self.detector   = None
        self.tracker    = None
        self.counter    = None
        self.event_log  = None
        self.drawer     = None

        self._threads = []
        self._cap = None

    def initialize(self, source) -> tuple[int, int]:
        """Mở video source và khởi tạo tất cả components."""
        self._cap = self._open_source(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không thể mở source: {source}")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Không đọc được frame đầu tiên")
        h, w = frame.shape[:2]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        logger.info(f"Frame size: {w}×{h}")

        # Khởi tạo components
        self.detector  = PersonDetector(self.cfg["detection"])
        self.tracker   = PersonTracker(self.cfg["tracking"])
        self.counter   = CrossingLineCounter(self.cfg["counter"], w, h)
        self.event_log = EventLogger(self.cfg["logging"])
        self.drawer    = Drawer(self.cfg["display"], w, h)

        return w, h

    def _open_source(self, source):
        if isinstance(source, int) or str(source).isdigit():
            cap = cv2.VideoCapture(int(source))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["source"]["resolution"][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["source"]["resolution"][1])
            cap.set(cv2.CAP_PROP_FPS, self.cfg["source"]["fps_cap"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        else:
            cap = cv2.VideoCapture(str(source))
        return cap

    # ------------------------------------------------------------------ #
    #  THREAD 1: Capture
    # ------------------------------------------------------------------ #
    def _capture_thread(self):
        logger.info("[Capture] Thread started")
        while self.running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("[Capture] End of stream or error")
                self.running = False
                break
            try:
                self.raw_queue.put_nowait(frame)
            except queue.Full:
                pass  # Drop frame nếu AI thread chưa xử lý kịp
        logger.info("[Capture] Thread stopped")

    # ------------------------------------------------------------------ #
    #  THREAD 2: AI Processing (detect + track + count)
    # ------------------------------------------------------------------ #
    def _ai_thread(self):
        logger.info("[AI] Thread started")
        fps_timer = time.time()
        fps_frame_count = 0

        while self.running:
            try:
                frame = self.raw_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            # 1. Detect persons
            detections = self.detector.detect(frame)

            # 2. Track
            tracks = self.tracker.update(detections, frame)

            # 3. Count crossings
            events = self.counter.process_tracks(tracks)

            # 4. Send events to logger
            for ev in events:
                try:
                    self.event_queue.put_nowait(ev)
                except queue.Full:
                    pass

            # 5. Update stats
            stats = self.counter.get_stats()
            elapsed = time.perf_counter() - t0

            fps_frame_count += 1
            if time.time() - fps_timer >= 1.0:
                with self._lock:
                    self._stats.update({
                        "fps": fps_frame_count,
                        "count_in":  stats["count_in"],
                        "count_out": stats["count_out"],
                        "occupancy": stats["current_occupancy"],
                        "active_tracks": stats["active_tracks"],
                        "frame_count": self._stats["frame_count"] + fps_frame_count,
                    })
                fps_frame_count = 0
                fps_timer = time.time()

            # 6. Draw và gửi frame sang UI
            annotated = self.drawer.draw_all(
                frame, tracks, events, self.counter, self._stats
            )
            try:
                self.processed_queue.put_nowait(annotated)
            except queue.Full:
                pass

        logger.info("[AI] Thread stopped")

    # ------------------------------------------------------------------ #
    #  THREAD 3: Logger
    # ------------------------------------------------------------------ #
    def _logger_thread(self):
        logger.info("[Logger] Thread started")
        buffer = []
        last_flush = time.time()
        flush_interval = self.cfg["logging"]["flush_interval_sec"]

        while self.running or not self.event_queue.empty():
            try:
                event = self.event_queue.get(timeout=0.5)
                buffer.append(event)
                self.event_log.log_event(event)
            except queue.Empty:
                pass

            if time.time() - last_flush >= flush_interval:
                self.event_log.flush()
                last_flush = time.time()

        self.event_log.flush()
        logger.info("[Logger] Thread stopped")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def start(self):
        self.running = True
        threads = [
            threading.Thread(target=self._capture_thread, name="capture", daemon=True),
            threading.Thread(target=self._ai_thread,      name="ai",      daemon=True),
            threading.Thread(target=self._logger_thread,  name="logger",  daemon=True),
        ]
        for t in threads:
            t.start()
        self._threads = threads
        logger.info("Pipeline started with 3 background threads")

    def get_frame(self, timeout: float = 0.05) -> np.ndarray | None:
        """Lấy processed frame mới nhất (non-blocking)."""
        try:
            return self.processed_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def stop(self):
        logger.info("Stopping pipeline...")
        self.running = False
        for t in self._threads:
            t.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info("Pipeline stopped")
