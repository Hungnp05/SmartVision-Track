"""
utils/logger.py
Thread-safe event logger: ghi sự kiện crossing ra CSV và SQLite.
"""

import csv
import sqlite3
import os
import threading
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Ghi mỗi crossing event ra:
    1. CSV: logs/events.csv  (append-only, dễ export)
    2. SQLite: data/events.db (query được, dùng cho dashboard)
    """

    CSV_HEADER = ["timestamp", "track_id", "direction", "cx", "cy"]

    def __init__(self, cfg: dict):
        self.csv_path = Path(cfg["csv_path"])
        self.db_path  = Path(cfg["db_path"])
        self._lock = threading.Lock()

        # Tạo thư mục nếu chưa có
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_csv()
        self._init_db()
        self._buffer = []

        logger.info(f"EventLogger initialized: CSV={self.csv_path}, DB={self.db_path}")

    def _init_csv(self):
        write_header = not self.csv_path.exists()
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
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
        """Ghi một crossing event (thread-safe)."""
        ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = [ts, event.track_id, event.direction, event.cx, event.cy]

        with self._lock:
            self._csv_writer.writerow(row)
            self._buffer.append(row)

    def flush(self):
        """Flush buffer xuống SQLite và CSV."""
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

    def query_hourly_counts(self, date: str = None) -> list[dict]:
        """
        Query số lượng IN/OUT theo từng giờ (cho dashboard chart).
        date: 'YYYY-MM-DD', mặc định = hôm nay
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        rows = self._db_conn.execute("""
            SELECT
                strftime('%H', timestamp) AS hour,
                direction,
                COUNT(*) AS count
            FROM events
            WHERE timestamp LIKE ?
            GROUP BY hour, direction
            ORDER BY hour
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
            FROM events
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [
            {"timestamp": r[0], "track_id": r[1],
             "direction": r[2], "cx": r[3], "cy": r[4]}
            for r in rows
        ]

    def get_total_counts(self) -> dict:
        row = self._db_conn.execute("""
            SELECT
                SUM(CASE WHEN direction='IN'  THEN 1 ELSE 0 END),
                SUM(CASE WHEN direction='OUT' THEN 1 ELSE 0 END)
            FROM events
            WHERE timestamp LIKE ?
        """, (f"{datetime.now().strftime('%Y-%m-%d')}%",)).fetchone()
        return {
            "total_in":  row[0] or 0,
            "total_out": row[1] or 0,
        }

    def close(self):
        self.flush()
        self._csv_file.close()
        self._db_conn.close()
