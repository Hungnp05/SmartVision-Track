"""
demo_data_generator.py
Tạo dữ liệu demo cho dashboard mà không cần camera thật.
Chạy: python demo_data_generator.py
"""

import sqlite3
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH  = Path("data/events.db")
CSV_PATH = Path("logs/events.csv")
Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Tạo DB
conn = sqlite3.connect(str(DB_PATH))
conn.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        direction TEXT NOT NULL,
        cx INTEGER,
        cy INTEGER,
        created_at TEXT DEFAULT (datetime('now'))
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
conn.commit()

# Xóa data cũ (chỉ cho demo)
conn.execute("DELETE FROM events")
conn.commit()

# Tạo dữ liệu giả lập theo pattern thực tế:
# - Buổi sáng (7-9h): đông người vào
# - Buổi trưa (12-13h): đông hơn
# - Chiều (17-19h): nhiều người ra
today = datetime.now().strftime("%Y-%m-%d")
events = []

hourly_pattern = {
    6:  (2,  1),   # (avg_in_per_min, avg_out_per_min)
    7:  (8,  2),
    8:  (15, 3),
    9:  (10, 5),
    10: (5,  4),
    11: (6,  5),
    12: (12, 8),
    13: (8,  12),
    14: (4,  6),
    15: (5,  5),
    16: (6,  7),
    17: (3,  14),
    18: (2,  12),
    19: (1,  8),
    20: (1,  3),
}

track_id = 1
for hour, (avg_in, avg_out) in hourly_pattern.items():
    for minute in range(60):
        n_in  = max(0, int(random.gauss(avg_in / 60 * 5, 0.5)))
        n_out = max(0, int(random.gauss(avg_out / 60 * 5, 0.5)))

        for _ in range(n_in):
            sec = random.randint(0, 299)
            ts  = f"{today} {hour:02d}:{minute:02d}:{sec % 60:02d}.{random.randint(0,999):03d}"
            events.append((ts, track_id, "IN", random.randint(200, 1000), random.randint(300, 600)))
            track_id += 1

        for _ in range(n_out):
            sec = random.randint(0, 299)
            ts  = f"{today} {hour:02d}:{minute:02d}:{sec % 60:02d}.{random.randint(0,999):03d}"
            # OUT dùng ID đã từng vào
            out_id = random.randint(1, max(1, track_id - 1))
            events.append((ts, out_id, "OUT", random.randint(200, 1000), random.randint(300, 600)))

# Sắp xếp theo thời gian
events.sort(key=lambda x: x[0])

# Insert vào DB
conn.executemany(
    "INSERT INTO events (timestamp, track_id, direction, cx, cy) VALUES (?,?,?,?,?)",
    events,
)
conn.commit()
conn.close()

# Ghi CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "track_id", "direction", "cx", "cy"])
    writer.writerows(events)

total_in  = sum(1 for e in events if e[2] == "IN")
total_out = sum(1 for e in events if e[2] == "OUT")
print(f" Demo data generated: {len(events)} events")
print(f"   IN:  {total_in}")
print(f"   OUT: {total_out}")
print(f"   DB:  {DB_PATH}")
print(f"   CSV: {CSV_PATH}")
print(f"\nChạy dashboard: streamlit run dashboard/app.py")
