import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrossingEvent:
    track_id:  int
    direction: str        # "IN" | "OUT"
    timestamp: datetime
    cx: int
    cy: int


@dataclass
class TrackHistory:
    positions:   list  = field(default_factory=list)   # [(cx, cy), ...]
    last_side:   Optional[float] = None                # -1.0 | 1.0
    cooldown:    int   = 0                             # frames còn lại không đếm
    miss_frames: int   = 0                             # số frame liên tiếp mất track
    crossed:     bool  = False                         # đã từng cross chưa


class CrossingLineCounter:
    """
    Đếm người vào/ra dựa trên một vạch ảo (crossing line).

    Thuật toán đơn giản & đáng tin cậy:
    - Mỗi track, lưu "bên nào" (sign of cross product) so với line vector.
    - Khi bên thay đổi → crossing event.
    - Cooldown sau mỗi cross để tránh flicker.
    - History giữ lại 90 frame sau khi track mất để handle re-entry.
    """

    COOLDOWN_FRAMES = 15   # frame không đếm sau mỗi lần cross
    HISTORY_TTL     = 90   # frame giữ history sau khi track biến mất

    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        ls = cfg["line_start"]
        le = cfg["line_end"]
        self.line_p1  = np.array([ls[0] * frame_w, ls[1] * frame_h], dtype=float)
        self.line_p2  = np.array([le[0] * frame_w, le[1] * frame_h], dtype=float)
        self.line_vec = self.line_p2 - self.line_p1

        self.count_in  = 0
        self.count_out = 0

        self._histories: dict[int, TrackHistory] = {}

        logger.info(
            f"CrossingLine init: p1={self.line_p1.tolist()} "
            f"p2={self.line_p2.tolist()} frame={frame_w}x{frame_h}"
        )

    # Line update (từ dashboard)
    def update_line(self, p1_rel: tuple, p2_rel: tuple):
        self.line_p1  = np.array([p1_rel[0] * self.frame_w,
                                   p1_rel[1] * self.frame_h], dtype=float)
        self.line_p2  = np.array([p2_rel[0] * self.frame_w,
                                   p2_rel[1] * self.frame_h], dtype=float)
        self.line_vec = self.line_p2 - self.line_p1
        logger.info(f"Line updated → p1={self.line_p1} p2={self.line_p2}")

    # Cross product: dương = bên phải vector, âm = bên trái
    def _side(self, point: np.ndarray) -> float:
        pv    = point - self.line_p1
        cross = self.line_vec[0] * pv[1] - self.line_vec[1] * pv[0]
        return float(np.sign(cross))   # -1.0, 0.0, hoặc 1.0

    # Main update
    def process_tracks(self, tracks: list[dict]) -> list[CrossingEvent]:
        active_ids = {t["id"] for t in tracks}
        new_events = []

        # Tăng miss_frames cho track không còn active
        for tid, h in self._histories.items():
            if tid not in active_ids:
                h.miss_frames += 1

        # Xóa history quá cũ (TTL hết)
        stale = [tid for tid, h in self._histories.items()
                 if h.miss_frames > self.HISTORY_TTL]
        for tid in stale:
            del self._histories[tid]

        # Xử lý từng track đang active
        for track in tracks:
            tid      = track["id"]
            x1,y1,x2,y2 = track["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            pt = np.array([cx, cy], dtype=float)

            # Khởi tạo history nếu track mới hoàn toàn
            if tid not in self._histories:
                self._histories[tid] = TrackHistory()

            h = self._histories[tid]
            h.miss_frames = 0   # reset vì track đang active
            h.positions.append((cx, cy))
            if len(h.positions) > 60:
                h.positions.pop(0)

            # Giảm cooldown
            if h.cooldown > 0:
                h.cooldown -= 1
                continue   # bỏ qua kiểm tra crossing khi đang cooldown

            side = self._side(pt)

            # Cần ít nhất 1 frame trước để biết "bên cũ"
            if h.last_side is None:
                if side != 0.0:
                    h.last_side = side
                continue

            # Kiểm tra crossing: bên thay đổi
            if side != 0.0 and side != h.last_side:
                direction = "IN" if (h.last_side < 0 and side > 0) else "OUT"

                event = CrossingEvent(
                    track_id  = tid,
                    direction = direction,
                    timestamp = datetime.now(),
                    cx = cx,
                    cy = cy,
                )
                new_events.append(event)

                if direction == "IN":
                    self.count_in  += 1
                else:
                    self.count_out += 1

                h.cooldown  = self.COOLDOWN_FRAMES
                h.last_side = side
                h.crossed   = True

                logger.info(
                    f"[Counter] Track#{tid} → {direction} "
                    f"(IN={self.count_in} OUT={self.count_out})"
                )
            elif side != 0.0:
                h.last_side = side

        return new_events

    # Helpers
    def get_stats(self) -> dict:
        active = sum(1 for h in self._histories.values()
                     if h.miss_frames == 0)
        return {
            "count_in":         self.count_in,
            "count_out":        self.count_out,
            "current_occupancy": max(0, self.count_in - self.count_out),
            "active_tracks":    active,
        }

    def get_line_pixels(self) -> tuple[tuple, tuple]:
        return (
            (int(self.line_p1[0]), int(self.line_p1[1])),
            (int(self.line_p2[0]), int(self.line_p2[1])),
        )

    def reset_counts(self):
        self.count_in  = 0
        self.count_out = 0
        logger.info("Counts reset to 0")
