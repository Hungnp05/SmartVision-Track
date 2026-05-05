"""
core/counter.py
Business Logic Layer: Crossing Line Counter.

Thuật toán:
1. Lưu lịch sử tọa độ tâm bbox (cx, cy) của mỗi track_id.
2. Khi track update, kiểm tra xem tâm vừa "băng qua" vạch ảo không.
3. Xác định hướng di chuyển (In/Out) dựa trên phía trước và sau khi qua.
4. Chống flicker bằng crossing_threshold: phải cách vạch ít nhất N pixel.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrossingEvent:
    track_id: int
    direction: str
    timestamp: datetime
    cx: int
    cy: int


@dataclass
class TrackHistory:
    positions: list = field(default_factory=list)  # [(cx, cy), ...]
    last_side: Optional[int] = None                # -1 | 0 | 1
    counted: bool = False


class CrossingLineCounter:
    """
    Đếm người vào/ra dựa trên một vạch ảo.
    Line được định nghĩa bởi 2 điểm, hướng đi được xác định
    bằng cross product của vector chuyển động với vector vuông góc line.
    """

    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.threshold = cfg["crossing_threshold"]

        # Chuyển tọa độ relative sang pixel
        lx1, ly1 = cfg["line_start"]
        lx2, ly2 = cfg["line_end"]
        self.line_p1 = np.array([lx1 * frame_w, ly1 * frame_h])
        self.line_p2 = np.array([lx2 * frame_w, ly2 * frame_h])
        self.line_vec = self.line_p2 - self.line_p1

        self.count_in  = 0
        self.count_out = 0

        self._track_histories: dict[int, TrackHistory] = {}
        self._recent_events: list[CrossingEvent] = []

        logger.info(
            f"Crossing line: ({self.line_p1}) → ({self.line_p2}), "
            f"threshold={self.threshold}px"
        )

    def update_line(self, p1_rel: tuple, p2_rel: tuple):
        """Cập nhật vị trí vạch đếm (gọi từ dashboard khi user kéo)."""
        self.line_p1 = np.array([p1_rel[0] * self.frame_w, p1_rel[1] * self.frame_h])
        self.line_p2 = np.array([p2_rel[0] * self.frame_w, p2_rel[1] * self.frame_h])
        self.line_vec = self.line_p2 - self.line_p1
        logger.info(f"Line updated: {self.line_p1} → {self.line_p2}")

    def _side_of_line(self, point: np.ndarray) -> float:
        """
        Tính cross product để biết điểm ở phía nào của line.
        > 0: bên phải line vector
        < 0: bên trái line vector
        = 0: trên đường line
        """
        pv = point - self.line_p1
        cross = self.line_vec[0] * pv[1] - self.line_vec[1] * pv[0]
        return cross

    def _dist_to_line(self, point: np.ndarray) -> float:
        """Khoảng cách vuông góc từ điểm đến đường thẳng."""
        line_len = np.linalg.norm(self.line_vec)
        if line_len == 0:
            return 0.0
        return abs(
            self.line_vec[0] * (self.line_p1[1] - point[1])
            - self.line_vec[1] * (self.line_p1[0] - point[0])
        ) / line_len

    def process_tracks(self, tracks: list[dict]) -> list[CrossingEvent]:
        """
        Nhận danh sách active tracks, kiểm tra crossing, trả về events mới.
        """
        active_ids = {t["id"] for t in tracks}

        # Xóa tracks không còn active (tránh memory leak)
        stale = [tid for tid in self._track_histories if tid not in active_ids]
        for tid in stale:
            del self._track_histories[tid]

        new_events = []

        for track in tracks:
            tid = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            point = np.array([cx, cy], dtype=float)

            if tid not in self._track_histories:
                self._track_histories[tid] = TrackHistory()

            history = self._track_histories[tid]
            history.positions.append((cx, cy))

            # Giới hạn lịch sử
            if len(history.positions) > 60:
                history.positions.pop(0)

            current_side = np.sign(self._side_of_line(point))
            dist = self._dist_to_line(point)

            # Chỉ xét crossing nếu đã có side trước đó và vượt threshold
            if history.last_side is not None and history.last_side != 0:
                if current_side != 0 and current_side != history.last_side:
                    if dist < self.threshold * 3:  # Phải đủ gần line
                        direction = self._classify_direction(
                            history.last_side, current_side
                        )
                        event = CrossingEvent(
                            track_id=tid,
                            direction=direction,
                            timestamp=datetime.now(),
                            cx=cx,
                            cy=cy,
                        )
                        new_events.append(event)
                        self._recent_events.append(event)

                        if direction == "IN":
                            self.count_in += 1
                        else:
                            self.count_out += 1

                        logger.debug(
                            f"Track {tid} crossed line → {direction} "
                            f"(total IN={self.count_in}, OUT={self.count_out})"
                        )

            if current_side != 0:
                history.last_side = current_side

        # Giới hạn recent events buffer
        if len(self._recent_events) > 1000:
            self._recent_events = self._recent_events[-500:]

        return new_events

    def _classify_direction(self, prev_side: float, curr_side: float) -> str:
        """
        IN = từ phía "trái/trên" của vector line sang phía "phải/dưới"
        OUT = ngược lại
        Quy ước có thể đảo bằng cách đổi chiều line vector.
        """
        if prev_side < 0 and curr_side > 0:
            return "IN"
        return "OUT"

    def get_stats(self) -> dict:
        return {
            "count_in": self.count_in,
            "count_out": self.count_out,
            "current_occupancy": max(0, self.count_in - self.count_out),
            "active_tracks": len(self._track_histories),
        }

    def get_line_pixels(self) -> tuple[tuple, tuple]:
        """Trả về tọa độ pixel của line để vẽ."""
        return (
            (int(self.line_p1[0]), int(self.line_p1[1])),
            (int(self.line_p2[0]), int(self.line_p2[1])),
        )

    def reset_counts(self):
        self.count_in = 0
        self.count_out = 0
        logger.info("Counts reset to 0")
