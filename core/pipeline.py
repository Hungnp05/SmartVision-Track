import threading
import queue
import time
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SmartVisionPipeline:
    """Pipeline chính của SmartVision-Track."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.running = False

        # Queues — dùng size nhỏ để luôn xử lý frame MỚI NHẤT
        # raw_queue size=1: capture luôn ghi đè frame cũ
        self.raw_queue       = queue.Queue(maxsize=1)
        # processed_queue size=2: đủ để UI lấy, không block AI
        self.processed_queue = queue.Queue(maxsize=2)
        self.event_queue     = queue.Queue(maxsize=500)

        # Shared state
        self._lock = threading.Lock()
        self._stats = {
            "fps": 0.0, "count_in": 0, "count_out": 0,
            "occupancy": 0, "active_tracks": 0, "frame_count": 0,
        }

        self.detector   = None
        self.tracker    = None
        self.counter    = None
        self.event_log  = None
        self.drawer     = None

        self._threads = []
        self._cap = None
        self.frame_w = 640
        self.frame_h = 480

        # Event báo hiệu init xong
        self._init_done  = threading.Event()
        self._init_error = None

    #  Open camera (nhanh — chỉ mở cap, không load model)
    def open_source(self, source) -> tuple[int, int]:
        """Mở camera ngay, trả về (w, h). Chỉ mất < 1 giây."""
        self._source = source
        self._cap = self._make_cap(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không mở được camera/source: {source}")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Không đọc được frame đầu tiên từ camera")

        self.frame_h, self.frame_w = frame.shape[:2]
        logger.info(f"Camera opened: {self.frame_w}×{self.frame_h}")
        return self.frame_w, self.frame_h

    def _make_cap(self, source):
        src_cfg = self.cfg["source"]
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)  # DSHOW nhanh hơn trên Windows
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  src_cfg["resolution"][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_cfg["resolution"][1])
            cap.set(cv2.CAP_PROP_FPS,          src_cfg["fps_cap"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # Buffer nhỏ nhất → frame luôn mới
        else:
            cap = cv2.VideoCapture(str(source))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    #  Background init thread — load YOLO + DeepSORT (chậm, ~5-15s)
    def _init_models_thread(self):
        """Chạy trong background. Main thread vẫn responsive."""
        try:
            from core.detector import PersonDetector
            from core.tracker  import PersonTracker
            from core.counter  import CrossingLineCounter
            from utils.logger  import EventLogger
            from utils.drawing import Drawer

            logger.info("[Init] Loading YOLOv8...")
            self.detector = PersonDetector(self.cfg["detection"])

            logger.info("[Init] Loading DeepSORT Re-ID model...")
            self.tracker = PersonTracker(self.cfg["tracking"])

            logger.info("[Init] Setting up counter, logger, drawer...")
            self.counter   = CrossingLineCounter(
                self.cfg["counter"], self.frame_w, self.frame_h
            )
            self.event_log = EventLogger(self.cfg["logging"])
            self.drawer    = Drawer(self.cfg["display"], self.frame_w, self.frame_h)

            logger.info("[Init] ✅ All models loaded successfully")
        except Exception as e:
            logger.exception(f"[Init] ❌ Model loading failed: {e}")
            self._init_error = e
        finally:
            self._init_done.set()

    #  THREAD: Capture — luôn đọc frame mới nhất
    def _capture_thread(self):
        logger.info("[Capture] Thread started")
        while self.running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("[Capture] Cannot read frame — stream ended or camera error")
                self.running = False
                break

            # Nếu queue đầy: bỏ frame cũ, đưa frame mới vào
            # → AI thread luôn nhận frame MỚI NHẤT (không bị delay tích lũy)
            try:
                self.raw_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.raw_queue.get_nowait()   # bỏ frame cũ
                except queue.Empty:
                    pass
                try:
                    self.raw_queue.put_nowait(frame)
                except queue.Full:
                    pass
        logger.info("[Capture] Thread stopped")

    #  THREAD: AI — detect + track + count + draw
    def _ai_thread(self):
        logger.info("[AI] Thread started — waiting for models...")
        self._init_done.wait()  # Đợi init xong

        if self._init_error:
            logger.error(f"[AI] Init failed, stopping: {self._init_error}")
            self.running = False
            return

        logger.info("[AI] Models ready — starting inference loop")
        fps_timer   = time.time()
        fps_count   = 0
        total_count = 0

        while self.running:
            try:
                frame = self.raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── Inference 
            detections = self.detector.detect(frame)
            tracks     = self.tracker.update(detections, frame)
            events     = self.counter.process_tracks(tracks)

            # ── Log events 
            for ev in events:
                try:
                    self.event_queue.put_nowait(ev)
                except queue.Full:
                    pass

            # ── FPS counter
            fps_count   += 1
            total_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                stats = self.counter.get_stats()
                with self._lock:
                    self._stats.update({
                        "fps":           fps_count,
                        "count_in":      stats["count_in"],
                        "count_out":     stats["count_out"],
                        "occupancy":     stats["current_occupancy"],
                        "active_tracks": stats["active_tracks"],
                        "frame_count":   total_count,
                    })
                fps_count = 0
                fps_timer = now

            # ── Draw ─────
            with self._lock:
                current_stats = dict(self._stats)
            annotated = self.drawer.draw_all(
                frame, tracks, events, self.counter, current_stats
            )

            # Gửi frame sang UI — nếu đầy thì bỏ frame cũ (không block!)
            try:
                self.processed_queue.put_nowait(annotated)
            except queue.Full:
                try:
                    self.processed_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.processed_queue.put_nowait(annotated)
                except queue.Full:
                    pass

        logger.info("[AI] Thread stopped")

    #  THREAD: Logger
    def _logger_thread(self):
        logger.info("[Logger] Thread started")
        last_flush    = time.time()
        flush_interval = self.cfg["logging"]["flush_interval_sec"]

        while self.running or not self.event_queue.empty():
            try:
                event = self.event_queue.get(timeout=0.5)
                self.event_log.log_event(event)
            except queue.Empty:
                pass

            if time.time() - last_flush >= flush_interval:
                self.event_log.flush()
                last_flush = time.time()

        self.event_log.flush()
        logger.info("[Logger] Thread stopped")

    #  Public API
    def start_init_async(self):
        """Bắt đầu load models trong background (non-blocking)."""
        t = threading.Thread(target=self._init_models_thread,
                             name="init-models", daemon=True)
        t.start()
        self._threads.append(t)

    def is_ready(self) -> bool:
        """True khi models đã load xong."""
        return self._init_done.is_set() and self._init_error is None

    def start_pipeline(self):
        """Khởi động capture + AI + logger threads."""
        self.running = True
        workers = [
            threading.Thread(target=self._capture_thread, name="capture", daemon=True),
            threading.Thread(target=self._ai_thread,      name="ai",      daemon=True),
            threading.Thread(target=self._logger_thread,  name="logger",  daemon=True),
        ]
        for t in workers:
            t.start()
        self._threads.extend(workers)
        logger.info("Pipeline started (3 worker threads)")

    def get_frame(self) -> np.ndarray | None:
        """Lấy frame mới nhất — non-blocking."""
        try:
            return self.processed_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_raw_frame(self) -> np.ndarray | None:
        """Lấy raw frame để hiển thị loading screen."""
        try:
            return self.raw_queue.queue[-1].copy() if self.raw_queue.queue else None
        except Exception:
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
        if self.event_log:
            self.event_log.close()
        logger.info("Pipeline stopped cleanly")
