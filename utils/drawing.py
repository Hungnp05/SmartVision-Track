"""
utils/drawing.py
OpenCV drawing helpers: bounding boxes, track IDs, trails, stats overlay.
"""

import cv2
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Màu sắc theo ID (HSV → BGR để phân biệt các track)
def id_to_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 37) % 180  # Phân tán đều trong vòng màu
    color = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


class Drawer:
    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.line_color   = tuple(cfg["line_color"])
        self.text_color   = tuple(cfg["text_color"])
        self.font         = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale   = cfg["font_scale"]
        self.alpha        = cfg["overlay_alpha"]

        # Lịch sử trail cho mỗi track
        self._trails: dict[int, list[tuple]] = defaultdict(list)
        self._trail_length = 30

    def draw_all(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        new_events: list,
        counter,
        stats: dict,
    ) -> np.ndarray:
        """Vẽ tất cả elements lên frame và return frame đã annotate."""
        out = frame.copy()

        # 1. Trails
        if self.cfg.get("show_trails", True):
            self._draw_trails(out, tracks)

        # 2. Crossing line
        self._draw_line(out, counter)

        # 3. Bounding boxes & IDs
        if self.cfg.get("show_bboxes", True):
            self._draw_tracks(out, tracks, counter)

        # 4. Flash effect khi có crossing event
        if new_events:
            self._flash_event(out, new_events)

        # 5. Stats overlay (góc trên bên trái)
        self._draw_stats_overlay(out, stats)

        return out

    def _draw_trails(self, frame: np.ndarray, tracks: list[dict]):
        active_ids = {t["id"] for t in tracks}
        stale = [tid for tid in self._trails if tid not in active_ids]
        for tid in stale:
            del self._trails[tid]

        for track in tracks:
            tid = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            self._trails[tid].append((cx, cy))
            if len(self._trails[tid]) > self._trail_length:
                self._trails[tid].pop(0)

            color = id_to_color(tid)
            pts = self._trails[tid]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    def _draw_line(self, frame: np.ndarray, counter):
        p1, p2 = counter.get_line_pixels()

        # Line chính (glow effect bằng 3 đường)
        cv2.line(frame, p1, p2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(frame, p1, p2, self.line_color, 2, cv2.LINE_AA)
        cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

        # Label "IN" và "OUT" phía trên/dưới line
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(frame, "IN",  (mid[0] + 10, mid[1] - 15),
                    self.font, 0.55, (0, 220, 60), 2, cv2.LINE_AA)
        cv2.putText(frame, "OUT", (mid[0] + 10, mid[1] + 25),
                    self.font, 0.55, (0, 80, 220), 2, cv2.LINE_AA)

    def _draw_tracks(self, frame: np.ndarray, tracks: list[dict], counter):
        for track in tracks:
            tid = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            conf = track.get("confidence", 0.0)
            color = id_to_color(tid)

            # Bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Nhãn ID + confidence
            label = f"#{tid}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
            pad = 4
            cv2.rectangle(
                frame,
                (x1, y1 - th - 2 * pad),
                (x1 + tw + 2 * pad, y1),
                color, -1
            )
            cv2.putText(
                frame, label,
                (x1 + pad, y1 - pad),
                self.font, self.font_scale,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    def _flash_event(self, frame: np.ndarray, events: list):
        """Vẽ border màu xanh/đỏ khi có crossing."""
        direction = events[-1].direction
        color = (0, 220, 60) if direction == "IN" else (0, 60, 220)
        h, w = frame.shape[:2]
        thick = 8
        cv2.rectangle(frame, (0, 0), (w, h), color, thick)

    def _draw_stats_overlay(self, frame: np.ndarray, stats: dict):
        """Semi-transparent overlay ở góc trên-trái với thống kê realtime."""
        overlay = frame.copy()
        box_w, box_h = 220, 135
        pad = 12
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        lines = [
            (f"FPS:  {stats.get('fps', 0):.0f}",               (0, 220, 180)),
            (f"IN:   {stats.get('count_in', 0)}",               (0, 200, 60)),
            (f"OUT:  {stats.get('count_out', 0)}",              (60, 80, 220)),
            (f"Inside: {stats.get('occupancy', 0)}",            (220, 200, 0)),
            (f"Tracks: {stats.get('active_tracks', 0)}",        (180, 180, 180)),
        ]
        for i, (text, color) in enumerate(lines):
            y = pad + 20 + i * 22
            cv2.putText(frame, text, (pad + 10, y),
                        self.font, 0.55, color, 1, cv2.LINE_AA)
