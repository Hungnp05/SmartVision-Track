"""
dashboard/app.py
Streamlit Dashboard - SmartVision-Track

Chạy: streamlit run dashboard/app.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys

# Thêm parent dir vào path để import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartVision-Track Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #1a1a2e;
    border: 1px solid #16213e;
    border-radius: 12px;
    padding: 16px;
}
.metric-label { color: #8892b0 !important; font-size: 13px !important; }
.metric-value { color: #ccd6f6 !important; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
div[data-testid="stVerticalBlock"] > div:has(div.stMetric) {
    background: #0a0a1a;
    border-radius: 8px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DB Helper
# ──────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "events.db"


def get_conn():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def load_hourly_data(date: str) -> pd.DataFrame:
    conn = get_conn()
    if conn is None:
        return pd.DataFrame(columns=["hour", "direction", "count"])
    df = pd.read_sql("""
        SELECT
            CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
            direction,
            COUNT(*) AS count
        FROM events
        WHERE timestamp LIKE ?
        GROUP BY hour, direction
        ORDER BY hour
    """, conn, params=(f"{date}%",))
    conn.close()
    return df


def load_timeline_data(date: str, interval: str = "10min") -> pd.DataFrame:
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    df = pd.read_sql("""
        SELECT timestamp, direction
        FROM events
        WHERE timestamp LIKE ?
        ORDER BY timestamp
    """, conn, params=(f"{date}%",))
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df["value"] = 1
    df_pivot = df.groupby(["direction"])["value"].resample(interval).sum().unstack(0).fillna(0)
    df_pivot.index.name = "time"
    df_pivot = df_pivot.reset_index()
    return df_pivot


def load_recent_events(limit: int = 100) -> pd.DataFrame:
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    df = pd.read_sql("""
        SELECT timestamp, track_id, direction
        FROM events
        ORDER BY id DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


def get_today_totals(date: str) -> dict:
    conn = get_conn()
    if conn is None:
        return {"total_in": 0, "total_out": 0}
    row = conn.execute("""
        SELECT
            SUM(CASE WHEN direction='IN'  THEN 1 ELSE 0 END),
            SUM(CASE WHEN direction='OUT' THEN 1 ELSE 0 END)
        FROM events WHERE timestamp LIKE ?
    """, (f"{date}%",)).fetchone()
    conn.close()
    return {"total_in": row[0] or 0, "total_out": row[1] or 0}


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Cài đặt")

    selected_date = st.date_input(
        "Ngày xem dữ liệu",
        value=datetime.today(),
        max_value=datetime.today(),
    )
    date_str = selected_date.strftime("%Y-%m-%d")

    st.divider()
    st.markdown("### Auto-refresh")
    auto_refresh = st.toggle("Bật auto-refresh", value=True)
    refresh_interval = st.slider("Khoảng thời gian (giây)", 2, 30, 5)

    st.divider()
    st.markdown("### Biểu đồ")
    chart_type = st.selectbox(
        "Loại biểu đồ theo giờ",
        ["Cột nhóm", "Cột xếp chồng", "Đường"]
    )
    timeline_interval = st.selectbox(
        "Granularity timeline",
        ["5min", "10min", "30min", "1h"],
        index=1,
    )

    st.divider()
    st.markdown("### Crossing Line")
    st.caption("Tọa độ relative (0.0 – 1.0)")
    col_a, col_b = st.columns(2)
    with col_a:
        lx1 = st.number_input("X1", 0.0, 1.0, 0.0, 0.05)
        ly1 = st.number_input("Y1", 0.0, 1.0, 0.5, 0.05)
    with col_b:
        lx2 = st.number_input("X2", 0.0, 1.0, 1.0, 0.05)
        ly2 = st.number_input("Y2", 0.0, 1.0, 0.5, 0.05)

    if st.button("Áp dụng line mới", use_container_width=True):
        # Ghi config mới vào shared file để main.py đọc
        import yaml
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            cfg["counter"]["line_start"] = [lx1, ly1]
            cfg["counter"]["line_end"]   = [lx2, ly2]
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            st.success("Đã lưu line mới vào config.yaml")

    st.divider()
    st.markdown("### Thông tin")
    st.caption(f"DB: `{DB_PATH}`")
    db_exists = DB_PATH.exists()
    st.markdown(
        f"{'🟢 DB online' if db_exists else '🔴 DB chưa tạo (chạy main.py trước)'}"
    )

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px'> SmartVision-Track</h1>"
    "<p style='color:#8892b0;margin-top:0'>Hệ thống Giám sát & Phân tích Lưu lượng Người Real-time</p>",
    unsafe_allow_html=True,
)
st.divider()

# ──────────────────────────────────────────────────────────────
# KPI Cards
# ──────────────────────────────────────────────────────────────
totals = get_today_totals(date_str)
total_in    = totals["total_in"]
total_out   = totals["total_out"]
occupancy   = max(0, total_in - total_out)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚶 Tổng vào hôm nay", f"{total_in:,}", help="Số người đi vào")
c2.metric("🚪 Tổng ra hôm nay",  f"{total_out:,}", help="Số người đi ra")
c3.metric("👥 Đang ở trong",     f"{occupancy:,}", help="Ước tính số người hiện tại")
c4.metric(
    "📅 Ngày xem",
    selected_date.strftime("%d/%m/%Y"),
    help="Dữ liệu của ngày được chọn",
)

st.divider()

# ──────────────────────────────────────────────────────────────
# Row 1: Biểu đồ theo giờ
# ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader(f" Lưu lượng theo giờ — {date_str}")
    hourly_df = load_hourly_data(date_str)

    if hourly_df.empty:
        st.info("Chưa có dữ liệu cho ngày này. Hãy chạy main.py để thu thập dữ liệu.")
    else:
        # Pivot để có cột IN và OUT
        pivot = hourly_df.pivot_table(
            index="hour", columns="direction", values="count", fill_value=0
        ).reset_index()
        for col in ["IN", "OUT"]:
            if col not in pivot.columns:
                pivot[col] = 0

        # Tất cả 24 giờ
        all_hours = pd.DataFrame({"hour": range(24)})
        pivot = all_hours.merge(pivot, on="hour", how="left").fillna(0)

        fig = go.Figure()
        color_in  = "#00c896"
        color_out = "#4a90d9"

        if chart_type == "Cột nhóm":
            fig.add_bar(x=pivot["hour"], y=pivot.get("IN", 0),
                        name="Vào", marker_color=color_in, opacity=0.85)
            fig.add_bar(x=pivot["hour"], y=pivot.get("OUT", 0),
                        name="Ra", marker_color=color_out, opacity=0.85)
            fig.update_layout(barmode="group")

        elif chart_type == "Cột xếp chồng":
            fig.add_bar(x=pivot["hour"], y=pivot.get("IN", 0),
                        name="Vào", marker_color=color_in)
            fig.add_bar(x=pivot["hour"], y=pivot.get("OUT", 0),
                        name="Ra", marker_color=color_out)
            fig.update_layout(barmode="stack")

        else:  # Đường
            fig.add_scatter(x=pivot["hour"], y=pivot.get("IN", 0),
                            name="Vào", mode="lines+markers",
                            line=dict(color=color_in, width=2))
            fig.add_scatter(x=pivot["hour"], y=pivot.get("OUT", 0),
                            name="Ra", mode="lines+markers",
                            line=dict(color=color_out, width=2))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Giờ trong ngày", tickmode="linear", tick0=0, dtick=1,
                       gridcolor="#1f2937"),
            yaxis=dict(title="Số người", gridcolor="#1f2937"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Timeline")
    timeline_df = load_timeline_data(date_str, timeline_interval)
    if timeline_df.empty:
        st.info("Chưa có dữ liệu timeline.")
    else:
        fig2 = go.Figure()
        if "IN" in timeline_df.columns:
            fig2.add_scatter(
                x=timeline_df["time"], y=timeline_df["IN"],
                name="Vào", fill="tozeroy",
                line=dict(color="#00c896", width=1.5),
                fillcolor="rgba(0,200,150,0.15)",
            )
        if "OUT" in timeline_df.columns:
            fig2.add_scatter(
                x=timeline_df["time"], y=timeline_df["OUT"],
                name="Ra", fill="tozeroy",
                line=dict(color="#4a90d9", width=1.5),
                fillcolor="rgba(74,144,217,0.15)",
            )
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937"),
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────────────────────
# Row 2: Recent Events Table
# ──────────────────────────────────────────────────────────────
st.subheader("Sự kiện gần nhất")
events_df = load_recent_events(100)
if events_df.empty:
    st.info("Chưa có sự kiện nào.")
else:
    events_df["Hướng"] = events_df["direction"].map(
        {"IN": "🟢 Vào", "OUT": "🔵 Ra"}
    )
    events_df = events_df.rename(columns={
        "timestamp": "Thời gian",
        "track_id":  "Track ID",
    })
    st.dataframe(
        events_df[["Thời gian", "Track ID", "Hướng"]],
        use_container_width=True,
        height=300,
        hide_index=True,
    )

# ──────────────────────────────────────────────────────────────
# Auto-refresh
# ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
