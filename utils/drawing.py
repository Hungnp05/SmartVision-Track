import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


def id_to_color(track_id) -> tuple[int, int, int]:
    """Màu unique per track ID (HSV → BGR)."""
    track_id = int(track_id)  # DeepSORT có thể trả về str
    hue   = (track_id * 47 + 30) % 180
    color = np.array([[[hue, 210, 230]]], dtype=np.uint8)
    bgr   = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def put_text_with_bg(img, text, pos, font_scale=0.55, color=(255,255,255),
                     bg_color=(20,20,20), thickness=1, pad=5):
    """Vẽ text với background rectangle."""
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


class Drawer:
    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self.cfg      = cfg
        self.frame_w  = frame_w
        self.frame_h  = frame_h
        self.alpha    = cfg.get("overlay_alpha", 0.35)

        # Trail history
        self._trails: dict[int, list] = defaultdict(list)
        self._trail_len = cfg.get("trail_length", 35)

        # FPS tính trong Drawer (không phụ thuộc pipeline)
        self._fps_timer    = time.perf_counter()
        self._fps_count    = 0
        self._fps_display  = 0.0

        # Flash event state
        self._flash_frames = 0
        self._flash_dir    = ""

    # Public
    def draw_all(self, frame, tracks, new_events, counter, stats) -> np.ndarray:
        out = frame.copy()

        # FPS tự tính
        self._fps_count += 1
        now = time.perf_counter()
        dt  = now - self._fps_timer
        if dt >= 0.5:
            self._fps_display = round(self._fps_count / dt, 1)
            self._fps_count   = 0
            self._fps_timer   = now

        # Layers
        if self.cfg.get("show_trails", True):
            self._draw_trails(out, tracks)

        self._draw_line(out, counter)

        if self.cfg.get("show_bboxes", True):
            self._draw_tracks(out, tracks)

        if new_events:
            self._flash_frames = 18
            self._flash_dir    = new_events[-1].direction

        if self._flash_frames > 0:
            self._draw_flash(out)
            self._flash_frames -= 1

        self._draw_stats_panel(out, stats)
        self._draw_help_bar(out)
        self._draw_timestamp(out)

        return out

    # Trails
    def _draw_trails(self, frame, tracks):
        active = {t["id"] for t in tracks}
        for tid in [k for k in self._trails if k not in active]:
            del self._trails[tid]

        for t in tracks:
            tid = t["id"]
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            self._trails[tid].append((cx, cy))
            if len(self._trails[tid]) > self._trail_len:
                self._trails[tid].pop(0)

            pts   = self._trails[tid]
            color = id_to_color(tid)
            for i in range(1, len(pts)):
                a = i / len(pts)
                cv2.line(frame, pts[i-1], pts[i], color,
                         max(1, int(a * 3)), cv2.LINE_AA)

    # Crossing line
    def _draw_line(self, frame, counter):
        p1, p2 = counter.get_line_pixels()
        h, w   = frame.shape[:2]

        # Glow: 3 lớp
        cv2.line(frame, p1, p2, (0,  0,  0),   6, cv2.LINE_AA)
        cv2.line(frame, p1, p2, (0, 230, 110),  2, cv2.LINE_AA)
        cv2.line(frame, p1, p2, (180, 255, 200), 1, cv2.LINE_AA)

        # Nhãn IN / OUT
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        put_text_with_bg(frame, " IN ",  (mid[0]+12, mid[1]-10),
                         0.55, (80, 255, 120), (0, 80, 30), 2, 3)
        put_text_with_bg(frame, " OUT ", (mid[0]+12, mid[1]+28),
                         0.55, (80, 140, 255), (0, 20, 80), 2, 3)

    # Tracks
    def _draw_tracks(self, frame, tracks):
        for t in tracks:
            tid         = t["id"]
            x1,y1,x2,y2 = t["bbox"]
            conf        = t.get("confidence", 0.0)
            color       = id_to_color(tid)

            # Bbox với corner accent thay vì full rectangle
            self._draw_corner_box(frame, x1, y1, x2, y2, color)

            # Label
            label = f" #{tid}  {conf:.2f} "
            put_text_with_bg(frame, label, (x1, y1 - 6),
                             0.52, (255,255,255), color, 1, 2)

            # Dot tâm
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, (255,255,255), 1, cv2.LINE_AA)

    def _draw_corner_box(self, frame, x1, y1, x2, y2, color, L=20, t=2):
        """Vẽ 4 góc thay vì full rectangle — trông gọn hơn."""
        corners = [
            [(x1,y1),(x1+L,y1)], [(x1,y1),(x1,y1+L)],
            [(x2,y1),(x2-L,y1)], [(x2,y1),(x2,y1+L)],
            [(x1,y2),(x1+L,y2)], [(x1,y2),(x1,y2-L)],
            [(x2,y2),(x2-L,y2)], [(x2,y2),(x2,y2-L)],
        ]
        for p1c, p2c in corners:
            cv2.line(frame, p1c, p2c, color, t, cv2.LINE_AA)
        # Thin full box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1, cv2.LINE_AA)

    # Flash khi crossing
    def _draw_flash(self, frame):
        h, w  = frame.shape[:2]
        is_in = self._flash_dir == "IN"
        color = (0, 220, 80) if is_in else (0, 80, 220)
        alpha = self._flash_frames / 18

        # Border flash
        thick = int(12 * alpha)
        if thick > 0:
            cv2.rectangle(frame, (0,0), (w,h), color, thick)

        # Text ở giữa
        if self._flash_frames > 10:
            txt = ">>> IN <<<" if is_in else "<<< OUT >>>"
            (tw, th), _ = cv2.getTextSize(txt, FONT, 1.8, 3)
            tx = (w - tw) // 2
            ty = h // 2 + th // 2
            cv2.putText(frame, txt, (tx+2, ty+2), FONT, 1.8,
                        (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(frame, txt, (tx, ty), FONT, 1.8,
                        color, 3, cv2.LINE_AA)

    # Stats panel (góc trên-trái)
    def _draw_stats_panel(self, frame, stats):
        pad    = 10
        box_w  = 210
        box_h  = 150
        margin = 12

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad),
                      (pad + box_w, pad + box_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (pad, pad),
                      (pad + box_w, pad + box_h), (50, 50, 50), 1)

        # Dùng FPS từ Drawer (chính xác hơn _stats["fps"])
        fps_val = self._fps_display

        rows = [
            (f"FPS   {fps_val:.1f}",                  (0, 230, 200)),
            (f"IN    {stats.get('count_in',  0)}",     (60, 230, 90)),
            (f"OUT   {stats.get('count_out', 0)}",     (80, 100, 230)),
            (f"INSIDE {stats.get('occupancy', 0)}",    (220, 200, 50)),
            (f"TRACKS {stats.get('active_tracks', 0)}",( 160, 160, 160)),
        ]
        x0 = pad + margin
        for i, (txt, color) in enumerate(rows):
            y = pad + margin + 4 + i * 26
            cv2.putText(frame, txt, (x0, y), FONT, 0.58,
                        color, 1, cv2.LINE_AA)

    # Help bar (dưới màn hình)
    def _draw_help_bar(self, frame):
        h, w = frame.shape[:2]
        bar_h = 22
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10,10,10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        help_txt = "  [Q] Thoat   [R] Reset counts   [S] Screenshot   [+/-] Di chuyen line"
        cv2.putText(frame, help_txt, (8, h - 6), FONT, 0.40,
                    (160, 160, 160), 1, cv2.LINE_AA)

    # Timestamp
    def _draw_timestamp(self, frame):
        h, w  = frame.shape[:2]
        ts    = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(ts, FONT, 0.45, 1)
        x = w - tw - 12
        y = 22
        cv2.putText(frame, ts, (x+1, y+1), FONT, 0.45, (0,0,0),   1, cv2.LINE_AA)
        cv2.putText(frame, ts, (x,   y),   FONT, 0.45, (180,180,180), 1, cv2.LINE_AA)