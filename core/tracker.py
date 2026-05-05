"""
core/tracker.py
DeepSORT tracker wrapper.

DeepSORT kết hợp:
- Kalman Filter: dự đoán vị trí khi bị mất dấu
- Re-ID CNN (MobileNet): trích xuất appearance feature
- Hungarian Algorithm: khớp detection với existing tracks
"""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Wrap DeepSORT để output track có unique ID bền vững,
    kể cả khi người bị che khuất tạm thời (occlusion).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.tracker = DeepSort(
            max_age=cfg["max_age"],
            n_init=cfg["n_init"],
            max_iou_distance=cfg["max_iou_distance"],
            max_cosine_distance=cfg["max_cosine_distance"],
            nn_budget=cfg["nn_budget"],
            embedder=cfg["embedder"],
            half=True,
            today=None,
        )

        logger.info(
            f"DeepSORT init: max_age={cfg['max_age']}, "
            f"n_init={cfg['n_init']}, embedder={cfg['embedder']}"
        )

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """
        Cập nhật tracker với detections mới.

        Args:
            detections: list từ PersonDetector.detect()
                        [{bbox: [x1,y1,x2,y2], confidence: float}, ...]
            frame: BGR frame gốc (cần để trích xuất Re-ID features)

        Returns:
            list tracks đang active:
            [{id: int, bbox: [x1,y1,x2,y2], confidence: float}, ...]
        """
        # DeepSORT expect format: [[x1,y1,w,h], confidence, class_id]
        raw_detections = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            raw_detections.append(([x1, y1, w, h], d["confidence"], 0))

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            active_tracks.append({
                "id": tid,
                "bbox": [x1, y1, x2, y2],
                "confidence": track.det_conf or 0.0,
            })

        return active_tracks
