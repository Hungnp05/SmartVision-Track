import csv
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


class EventLogger:
    """Logger cho chế độ webcam realtime → data/live_track.csv"""

    # Cột đầy đủ hơn version cũ
    CSV_HEADER = [
        "date", "time", "track_id", "direction",
        "cx", "cy", "session_in", "session_out", "occupancy"
    ]

    def __init__(self, cfg: dict):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.db_path = Path(cfg["db_path"])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # live_track.csv: 1 file per ngày
        today = datetime.now().strftime("%Y-%m-%d")
        self.csv_path = DATA_DIR / f"live_track_{today}.csv"

        self._lock       = threading.Lock()
        self._buffer     = []
        self._session_in  = 0
        self._session_out = 0

        self._init_csv()
        self._init_db()

        logger.info(f"EventLogger: CSV={self.csv_path}, DB={self.db_path}")

    def _init_csv(self):
        write_header = not self.csv_path.exists()
        self._csv_file   = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow(self.CSV_HEADER)
            self._csv_file.flush()

    def _init_db(self):
        self._db_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                track_id    INTEGER NOT NULL,
                direction   TEXT NOT NULL,
                cx          INTEGER,
                cy          INTEGER,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        self._db_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)"
        )
        self._db_conn.commit()

    def log_event(self, event):
        """Ghi crossing event từ webcam (thread-safe)."""
        if event.direction == "IN":
            self._session_in  += 1
        else:
            self._session_out += 1

        occupancy = max(0, self._session_in - self._session_out)
        now = event.timestamp
        row_csv = [
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S.%f")[:-3],
            event.track_id,
            event.direction,
            event.cx,
            event.cy,
            self._session_in,
            self._session_out,
            occupancy,
        ]
        row_db = (now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                  event.track_id, event.direction, event.cx, event.cy)

        with self._lock:
            self._csv_writer.writerow(row_csv)
            self._buffer.append(row_db)

    def flush(self):
        with self._lock:
            if self._buffer:
                self._db_conn.executemany(
                    "INSERT INTO events (timestamp, track_id, direction, cx, cy) "
                    "VALUES (?, ?, ?, ?, ?)",
                    self._buffer,
                )
                self._db_conn.commit()
                self._buffer.clear()
            self._csv_file.flush()

    # Queries cho dashboard
    def query_hourly_counts(self, date: str = None) -> list[dict]:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        rows = self._db_conn.execute("""
            SELECT strftime('%H', timestamp) AS hour, direction, COUNT(*) cnt
            FROM events WHERE timestamp LIKE ?
            GROUP BY hour, direction ORDER BY hour
        """, (f"{date}%",)).fetchall()
        result = {}
        for hour, direction, count in rows:
            h = int(hour)
            if h not in result:
                result[h] = {"hour": h, "IN": 0, "OUT": 0}
            result[h][direction] = count
        return list(result.values())

    def query_recent_events(self, limit: int = 50) -> list[dict]:
        rows = self._db_conn.execute("""
            SELECT timestamp, track_id, direction, cx, cy
            FROM events ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [{"timestamp": r[0], "track_id": r[1],
                 "direction": r[2], "cx": r[3], "cy": r[4]} for r in rows]

    def get_total_counts(self) -> dict:
        row = self._db_conn.execute("""
            SELECT SUM(CASE WHEN direction='IN' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN direction='OUT' THEN 1 ELSE 0 END)
            FROM events WHERE timestamp LIKE ?
        """, (f"{datetime.now().strftime('%Y-%m-%d')}%",)).fetchone()
        return {"total_in": row[0] or 0, "total_out": row[1] or 0}

    def close(self):
        self.flush()
        self._csv_file.close()
        self._db_conn.close()
        logger.info(f"EventLogger closed. CSV: {self.csv_path}")


class VideoTrackLogger:
    """
    Logger riêng cho chế độ xử lý video file.
    Ghi ra: data/video_track_<tên_video>_<timestamp>.csv

    Cột: video_file, frame_no, video_time, track_id, direction, cx, cy,
         total_in, total_out, occupancy
    """

    CSV_HEADER = [
        "video_file", "frame_no", "video_time_sec",
        "track_id", "direction",
        "cx", "cy",
        "total_in", "total_out", "occupancy",
    ]

    def __init__(self, video_path: str | Path):
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.video_name = Path(video_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = DATA_DIR / f"video_track_{self.video_name}_{ts}.csv"

        self._total_in  = 0
        self._total_out = 0
        self._rows      = []   # buffer, flush khi xong

        self._csv_file   = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADER)

        logger.info(f"VideoTrackLogger: {self.csv_path}")

    def log_event(self, event, frame_no: int, video_fps: float):
        """Ghi một crossing event từ video processor."""
        if event.direction == "IN":
            self._total_in  += 1
        else:
            self._total_out += 1

        occupancy    = max(0, self._total_in - self._total_out)
        video_time   = round(frame_no / max(video_fps, 1), 3)

        row = [
            self.video_name,
            frame_no,
            video_time,
            event.track_id,
            event.direction,
            event.cx,
            event.cy,
            self._total_in,
            self._total_out,
            occupancy,
        ]
        self._csv_writer.writerow(row)
        self._rows.append(row)

    def flush(self):
        self._csv_file.flush()

    def close(self) -> Path:
        self.flush()
        self._csv_file.close()
        logger.info(
            f"VideoTrackLogger closed: {len(self._rows)} events → {self.csv_path}"
        )
        return self.csv_path

    @property
    def total_in(self):  return self._total_in
    @property
    def total_out(self): return self._total_out
    @property
    def event_count(self): return len(self._rows)