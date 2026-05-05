# SmartVision-Track
**Hệ thống Giám sát & Phân tích Lưu lượng Người Real-time**

## Yêu cầu hệ thống
- Python 3.10+
- CUDA 11.8+ (RTX 4050 recommended)
- Webcam hoặc RTSP stream

## Cài đặt nhanh

```bash
# 1. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Download YOLOv8 weights (tự động khi chạy lần đầu)
# hoặc thủ công:
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

# 4. Chạy ứng dụng chính (OpenCV window)
python main.py --source 0           # webcam
python main.py --source rtsp://...  # RTSP stream
python main.py --source video.mp4   # file video

# 5. Chạy Dashboard Streamlit (terminal riêng)
streamlit run dashboard/app.py
```

## Cấu trúc thư mục
```
SmartVision-Track/
├── main.py                 # Entry point - OpenCV realtime view
├── requirements.txt
├── config.yaml             # Tất cả tham số cấu hình
├── core/
│   ├── detector.py         # YOLOv8 inference wrapper
│   ├── tracker.py          # DeepSORT tracker wrapper
│   ├── counter.py          # Crossing line logic
│   └── pipeline.py         # Async pipeline orchestrator
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── utils/
│   ├── logger.py           # CSV + SQLite logger
│   ├── drawing.py          # OpenCV drawing helpers
│   └── video_source.py     # Webcam/RTSP abstraction
├── data/
│   └── events.db           # SQLite database (auto-created)
└── logs/
    └── events.csv          # CSV log (auto-created)
```

## Tính năng
- ✅ YOLOv8 person detection với TensorRT acceleration
- ✅ DeepSORT tracking với Re-ID (unique ID mỗi người)
- ✅ Crossing line counter (vào/ra) với configurable line position  
- ✅ Async processing (UI thread tách khỏi AI thread)
- ✅ Frame skipping để tối ưu hiệu năng
- ✅ SQLite + CSV logging
- ✅ Streamlit dashboard với biểu đồ realtime
- ✅ Occlusion handling qua Kalman Filter
