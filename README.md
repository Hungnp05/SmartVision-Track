# SmartVision-Track

> **Real-time People Surveillance & Traffic Analysis System**  
> Detect, track and count people entering/exiting in real time using YOLOv8 + DeepSORT.

---

## Project Purpose

SmartVision-Track is an **Edge AI** application built to solve the problem of **people counting and traffic analysis** in real-world environments such as:

- Retail stores & supermarkets — count customers entering/exiting per hour
- Offices & buildings — control the number of people in a zone
- Factories & warehouses — monitor occupational safety compliance
- Security cameras — analyze movement patterns and foot traffic

The system runs entirely **offline** — no internet required — directly on local hardware (Edge AI). It supports both **real-time webcam** streaming and **offline video file** processing.

---

## Performance

| Hardware | Model | Detection FPS | Display FPS |
|---|---|---|---|
| RTX 4050 (CUDA + FP16) | YOLOv8s | ~45 FPS | 30 FPS |
| RTX 4050 (CUDA + FP16) | YOLOv8n | ~80 FPS | 30 FPS |
| CPU i7-13650HX (no GPU) | YOLOv8n | ~8–12 FPS | 8–12 FPS |
| CPU i7-13650HX (no GPU) | YOLOv8s | ~4–6 FPS | 4–6 FPS |

> **Notes:**
> - Actual FPS depends on camera resolution and the number of people in frame.
> - With `frame_skip: 2` in `config.yaml`, AI runs inference every 2 frames → ~1.8× FPS boost with minimal impact on tracking accuracy.
> - The **first launch** takes 10–20 seconds to download and cache the model weights.

---

## Project Structure

```
SmartVision-Track/
│
├── main.py                          # Entry point — live or video mode
├── config.yaml                      # All configurable parameters
├── requirements.txt                 # Python dependencies
│
├── core/                            # AI processing core
│   ├── __init__.py
│   ├── detector.py                  # YOLOv8 wrapper (class=0 Person only)
│   ├── tracker.py                   # DeepSORT wrapper (Re-ID + Kalman Filter)
│   ├── counter.py                   # Crossing line logic (IN/OUT counting)
│   ├── pipeline.py                  # Async pipeline (4 parallel threads)
│   └── video_processor.py          # Synchronous video file processor
│
├── utils/                           # Utilities
│   ├── __init__.py
│   ├── drawing.py                   # Draw bbox, trail, overlay, flash events
│   └── logger.py                   # CSV writer (live_track + video_track) & SQLite
│
├── dashboard/
│   └── app.py                      # Streamlit Analytics Dashboard
│
├── data/                            # Output data (auto-created at runtime)
│   ├── events.db                    # SQLite database — used by dashboard
│   ├── live_track_YYYY-MM-DD.csv   # Real-time webcam log (one file per day)
│   ├── video_track_<name>_<ts>.csv # Video processing log
│   └── video_annotated_<name>.mp4  # Annotated output video (optional)
│
├── logs/                            # System logs (auto-created at runtime)
│   ├── runtime.log                  # Full runtime log
│   └── screenshot_*.jpg             # Screenshots (press S key)
│
├── demo_data_generator.py           # Generate sample data for the dashboard
└── venv/                            # Python virtual environment (not committed)
```

---

## How It Works

### Live Mode (Webcam / RTSP)

```
┌──────────────────────────────────────────────────────────────────┐
│                        4 PARALLEL THREADS                         │
│                                                                    │
│  [Capture]         [AI Thread]           [Logger]   [Main/UI]    │
│     │                   │                   │           │         │
│  Read frame ──→  raw_queue   ──→  YOLO detect          │         │
│     │           (size=1,         ──→  DeepSORT track   │         │
│     │           always newest)   ──→  Crossing count   │         │
│     │                   │            ──→  Draw overlay  │         │
│     │                   │            ──→  processed_q   │         │
│     │                   │                   │    └──→  imshow    │
│     │               event_queue ──→  log_event          │         │
│     │                               ──→  live_track.csv │         │
│     │                               ──→  events.db      │         │
└──────────────────────────────────────────────────────────────────┘
```

**Startup sequence:**
1. `open_source()` — opens the camera immediately (< 1 second), UI window appears
2. `start_init_async()` — loads YOLO + DeepSORT in a **background thread** (UI never blocks)
3. **Loading screen** — displays raw camera feed + spinner animation while loading
4. Once models are ready → automatically switches to real-time inference

### Video Mode (File Processing)

```
python main.py --mode video --video clip.mp4
    │
    ├── Open video file (cv2.VideoCapture)
    ├── Load YOLOv8 + DeepSORT (synchronous, one-time)
    │
    └── Per-frame loop:
            ├── detect()     → YOLOv8 (Person class only)
            ├── update()     → DeepSORT tracking
            ├── process()    → Crossing line count
            ├── log_event()  → video_track.csv (written immediately)
            └── draw_all() + write() → annotated video (optional)
    │
    └── Done: print summary report to terminal
```

### Crossing Line Algorithm

```
Each frame:
  1. Compute centroid (cx, cy) of each bounding box
  2. Calculate cross product: sign( line_vec × point_vec )
       → +1.0 = right side of line vector
       → -1.0 = left side of line vector
  3. If sign changes from previous frame → CROSSING EVENT
  4. Classify direction: -1→+1 = "IN"  |  +1→-1 = "OUT"
  5. Apply 15-frame cooldown per track to prevent bounce/flicker
```

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU fallback) | NVIDIA RTX (CUDA 11.8+) |
| Camera | USB 720p | USB/IP 1080p |
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |

### Step 1 — Clone or extract the project

```bash
# Using Git
git clone https://github.com/Hungnp05/SmartVision-Track
cd SmartVision-Track

# Or extract the zip archive
cd SmartVision-Track
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install all packages
pip install -r requirements.txt
```

> **PyTorch with CUDA:** To use GPU acceleration (NVIDIA RTX), install the CUDA build of PyTorch instead of the default CPU build:
> ```bash
> # PyTorch CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
>
> # PyTorch CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4 — Download YOLOv8 weights (automatic)

Weights are downloaded automatically on first run. To pre-download manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

---

## Running the Application

> **Always activate the virtual environment before running:**
> ```bash
> # Windows
> venv\Scripts\activate
>
> # Linux / macOS
> source venv/bin/activate
> ```

---

### 1. Live — Default webcam (index 0)

```bash
python main.py
```

---

### 2. Live — Specific webcam (index 1, 2, ...)

```bash
python main.py --source 1
```

---

### 3. Live — IP Camera / RTSP stream

```bash
python main.py --source "rtsp://admin:password@192.168.1.100:554/stream"
```

---

### 4. Live — Headless mode (no display window, CSV only)

```bash
python main.py --headless
```

---

### 5. Video — Process a video file (with annotated output)

```bash
# Windows
python main.py --mode video --video "C:\Videos\recording.mp4"

# Linux / macOS
python main.py --mode video --video /home/user/videos/recording.mp4
```

Files automatically created in `data/`:
- `video_track_recording_20260508_100701.csv`
- `video_annotated_recording_20260508_100701.mp4`

---

### 6. Video — CSV output only, skip annotated video (~30% faster)

```bash
python main.py --mode video --video recording.mp4 --no-annotated
```

---

### 7. Video — Enter path interactively (no --video flag)

```bash
python main.py --mode video
# Program will prompt: "Video path: "
```

---

### 8. Custom config file

```bash
python main.py --config my_config.yaml
python main.py --mode video --video clip.mp4 --config my_config.yaml
```

---

### 9. Analytics Dashboard (Streamlit)

Run in a separate terminal alongside any of the modes above:

```bash
# Second terminal (venv activated)
streamlit run dashboard/app.py
# Open in browser: http://localhost:8501
```

---

### 10. Generate demo data (no camera needed)

```bash
python demo_data_generator.py
streamlit run dashboard/app.py
```

---

## Keyboard Shortcuts (Live Mode)

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit the application |
| `R` | Reset IN/OUT counters to zero |
| `S` | Save screenshot → `logs/screenshot_<timestamp>.jpg` |
| `SPACE` | Pause / Resume |
| `+` or `=` | Move crossing line **down** by 5% |
| `-` | Move crossing line **up** by 5% |
| `]` | Move crossing line **right** by 5% |
| `[` | Move crossing line **left** by 5% |

> Line position changes are **automatically saved** to `config.yaml`.

---

## Configuration — `config.yaml`

Key parameters to tune for your environment:

```yaml
detection:
  model: "yolov8n.pt"    # n = fastest, s = balanced, m = most accurate
  confidence: 0.45        # Raise if too many false positives
  device: "cuda"          # Change to "cpu" if no GPU available
  frame_skip: 2           # 1 = no skip, 2 = infer every 2nd frame

tracking:
  max_age: 30             # Frames to keep a track alive when lost (occlusion)
  n_init: 3               # Consecutive frames needed to confirm a new track

counter:
  line_start: [0.0, 0.5]  # Line start point (x, y) — relative coords 0.0–1.0
  line_end:   [1.0, 0.5]  # Line end point
```

---

## CSV Output Format

### `data/live_track_YYYY-MM-DD.csv`

Written in real time during webcam sessions. One row per crossing event.

| Column | Type | Description |
|---|---|---|
| `date` | TEXT | Event date (`2026-05-08`) |
| `time` | TEXT | Event time (`10:07:23.441`) |
| `track_id` | INT | Unique ID assigned to this person |
| `direction` | TEXT | `IN` or `OUT` |
| `cx` | INT | Bounding box centroid X (pixels) |
| `cy` | INT | Bounding box centroid Y (pixels) |
| `session_in` | INT | Cumulative IN count since session start |
| `session_out` | INT | Cumulative OUT count since session start |
| `occupancy` | INT | Estimated number of people currently in zone |

### `data/video_track_<name>_<timestamp>.csv`

Written after processing a video file. One row per crossing event.

| Column | Type | Description |
|---|---|---|
| `video_file` | TEXT | Source video filename (no extension) |
| `frame_no` | INT | Frame number where the crossing occurred |
| `video_time_sec` | FLOAT | Timestamp in the video (seconds) |
| `track_id` | INT | Unique ID assigned to this person |
| `direction` | TEXT | `IN` or `OUT` |
| `cx` | INT | Bounding box centroid X (pixels) |
| `cy` | INT | Bounding box centroid Y (pixels) |
| `total_in` | INT | Cumulative IN count up to this event |
| `total_out` | INT | Cumulative OUT count up to this event |
| `occupancy` | INT | Estimated occupancy at this point in time |

---

## Troubleshooting

**Camera fails to open:**
```bash
# Try a different webcam index
python main.py --source 1

# Windows: disable DirectShow backend
# In config.yaml: use_dshow: false
```

**CUDA not available (falling back to CPU):**
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with the correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Low FPS on CPU:**
```yaml
# config.yaml — adjust these parameters:
detection:
  model: "yolov8n.pt"  # Use Nano instead of Small
  frame_skip: 3         # Skip more frames
  input_size: 320       # Reduce input resolution
```

**Track IDs switching frequently:**
```yaml
tracking:
  max_age: 50                # Keep tracks alive longer when lost
  n_init: 2                  # Confirm new tracks faster
  max_cosine_distance: 0.4   # Relax Re-ID appearance matching
```

---

## Main Dependencies

| Package | Version | Role |
|---|---|---|
| `ultralytics` | ≥ 8.0 | YOLOv8 object detection |
| `deep-sort-realtime` | ≥ 1.3.2 | DeepSORT tracking + Re-ID embedder |
| `opencv-python` | ≥ 4.8 | Video capture & frame drawing |
| `torch` / `torchvision` | ≥ 2.0 | Deep learning backend (CPU or CUDA) |
| `streamlit` | ≥ 1.28 | Analytics dashboard UI |
| `plotly` | ≥ 5.17 | Interactive charts |
| `pandas` | ≥ 2.0 | CSV data processing |
| `PyYAML` | ≥ 6.0 | Config file read/write |


## Use app without download code:
The recipient just needs:
1. download zip file: https://drive.google.com/file/d/1gOdjutdl--EbpLJjyT6OaUo_jFxPQTqO/view?usp=sharing
2. Unzip the zip
3. Double-click run.bat
4. Select mode → done