"""
core/detector.py
YOLOv8 inference wrapper với frame skipping & GPU acceleration.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Wrap YOLOv8 để chỉ detect class=0 (person).
    Hỗ trợ frame skipping, FP16 half-precision, CUDA/MPS/CPU.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_path = cfg["model"]
        self.confidence = cfg["confidence"]
        self.iou = cfg["iou_threshold"]
        self.device = self._resolve_device(cfg["device"])
        self.half = cfg["half_precision"] and self.device != "cpu"
        self.input_size = cfg["input_size"]
        self.frame_skip = max(1, cfg["frame_skip"])

        self._frame_counter = 0
        self._last_detections = []

        logger.info(f"Loading YOLOv8 model: {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)

        # Warm-up inference để khởi tạo CUDA context
        if self.device != "cpu":
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, device=self.device, half=self.half, verbose=False)
            logger.info("GPU warm-up complete")

    def _resolve_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA không khả dụng, fallback về CPU")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS không khả dụng, fallback về CPU")
            return "cpu"
        return device

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Chạy inference trên frame.
        Trả về list của dict: {bbox: [x1,y1,x2,y2], confidence: float}

        Nếu frame_skip > 1, chỉ inference mỗi N frame và cache kết quả.
        """
        self._frame_counter += 1

        if self._frame_counter % self.frame_skip != 0:
            return self._last_detections

        results = self.model(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            half=self.half,
            imgsz=self.input_size,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                })

        self._last_detections = detections
        return detections

    def get_device_info(self) -> str:
        if self.device == "cuda":
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"CUDA: {name} ({mem:.1f}GB)"
        return self.device.upper()
