import streamlit as st
import cv2
import numpy as np
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import yaml
import time
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH  = ROOT / "data" / "events.db"
CFG_PATH = ROOT / "config.yaml"

#  Page config 
st.set_page_config(
    page_title="SmartVision-Track",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;
  border-radius:10px;padding:14px 18px;}
[data-testid="stImage"]{border-radius:8px;overflow:hidden;}
.stTabs [data-baseweb="tab"]{background:#1f2937;border-radius:8px 8px 0 0;
  padding:8px 20px;color:#9ca3af;}
.stTabs [aria-selected="true"]{background:#374151!important;color:#f9fafb!important;}
.badge-live   {background:#065f46;color:#6ee7b7;padding:3px 12px;
               border-radius:12px;font-size:12px;font-weight:700;}
.badge-stopped{background:#450a0a;color:#fca5a5;padding:3px 12px;
               border-radius:12px;font-size:12px;font-weight:700;}
.badge-loading{background:#1c1917;color:#fbbf24;padding:3px 12px;
               border-radius:12px;font-size:12px;font-weight:700;}
</style>""", unsafe_allow_html=True)


#  CONFIG

def load_cfg() -> dict:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_cfg(cfg: dict):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


#  PIPELINE SESSION STATE

def get_pipeline():
    return st.session_state.get("sv_pipeline", None)

def pipeline_status() -> str:
    p = get_pipeline()
    if p is None:
        return "stopped"
    if not p.is_ready():
        return "loading"
    return "live"

def start_pipeline(cam_idx: int):
    from core.pipeline import SmartVisionPipeline
    stop_pipeline()
    cfg = load_cfg()
    p   = SmartVisionPipeline(cfg)
    try:
        w, h = p.open_source(cam_idx)
    except RuntimeError as e:
        st.error(f"Không mở được camera {cam_idx}: {e}")
        return
    p.start_init_async()
    p.start_pipeline()
    st.session_state["sv_pipeline"]   = p
    st.session_state["sv_start_time"] = time.time()
    st.session_state["sv_cam_w"]      = w
    st.session_state["sv_cam_h"]      = h

def stop_pipeline():
    p = get_pipeline()
    if p:
        p.stop()
    for k in ["sv_pipeline", "sv_start_time", "sv_cam_w", "sv_cam_h"]:
        st.session_state.pop(k, None)


#  DB HELPERS

def _conn():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

def get_totals(date: str) -> dict:
    c = _conn()
    if not c:
        return {"total_in": 0, "total_out": 0}
    r = c.execute("""
        SELECT SUM(CASE WHEN direction='IN' THEN 1 ELSE 0 END),
               SUM(CASE WHEN direction='OUT' THEN 1 ELSE 0 END)
        FROM events WHERE timestamp LIKE ?
    """, (f"{date}%",)).fetchone()
    c.close()
    return {"total_in": r[0] or 0, "total_out": r[1] or 0}

def hourly_df(date: str) -> pd.DataFrame:
    c = _conn()
    if not c:
        return pd.DataFrame()
    df = pd.read_sql("""
        SELECT CAST(strftime('%H',timestamp) AS INT) hour,
               direction, COUNT(*) cnt
        FROM events WHERE timestamp LIKE ?
        GROUP BY hour, direction ORDER BY hour
    """, c, params=(f"{date}%",))
    c.close()
    return df

def timeline_df(date: str, freq: str = "10min") -> pd.DataFrame:
    c = _conn()
    if not c:
        return pd.DataFrame()
    df = pd.read_sql("""
        SELECT timestamp, direction FROM events
        WHERE timestamp LIKE ? ORDER BY timestamp
    """, c, params=(f"{date}%",))
    c.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df["v"] = 1
    pv = df.groupby("direction")["v"].resample(freq).sum().unstack(0).fillna(0)
    pv.index.name = "time"
    return pv.reset_index()

def recent_events(limit: int = 60) -> pd.DataFrame:
    c = _conn()
    if not c:
        return pd.DataFrame()
    df = pd.read_sql("""
        SELECT timestamp, track_id, direction FROM events
        ORDER BY id DESC LIMIT ?
    """, c, params=(limit,))
    c.close()
    return df


#  SIDEBAR

with st.sidebar:
    st.markdown("## SmartVision-Track")
    st.divider()

    st.markdown("### Camera")
    cam_idx = st.number_input("Webcam index", 0, 9, 0, 1)

    status = pipeline_status()
    badge  = {"stopped": "badge-stopped", "loading": "badge-loading", "live": "badge-live"}[status]
    label  = {"stopped": "⏹ Stopped", "loading": "Loading…", "live": "🟢 Live"}[status]
    st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start", use_container_width=True, disabled=(status != "stopped")):
            start_pipeline(cam_idx)
            st.rerun()
    with c2:
        if st.button("⏹ Stop", use_container_width=True, disabled=(status == "stopped")):
            stop_pipeline()
            st.rerun()

    if status == "live":
        if st.button("Reset counts", use_container_width=True):
            p = get_pipeline()
            if p and p.counter:
                p.counter.reset_counts()

    st.divider()

    #  Crossing line 
    st.markdown("### Crossing Line")
    cfg_now = load_cfg()
    ls = cfg_now.get("counter", {}).get("line_start", [0.0, 0.5])
    le = cfg_now.get("counter", {}).get("line_end",   [1.0, 0.5])

    st.caption("Điểm đầu → Điểm cuối  (0.0 – 1.0)")
    ca, cb = st.columns(2)
    with ca:
        lx1 = st.number_input("X1", 0.0, 1.0, float(ls[0]), 0.05, key="lx1")
        ly1 = st.number_input("Y1", 0.0, 1.0, float(ls[1]), 0.05, key="ly1")
    with cb:
        lx2 = st.number_input("X2", 0.0, 1.0, float(le[0]), 0.05, key="lx2")
        ly2 = st.number_input("Y2", 0.0, 1.0, float(le[1]), 0.05, key="ly2")

    if st.button("✅ Áp dụng line", use_container_width=True):
        cfg_now["counter"]["line_start"] = [lx1, ly1]
        cfg_now["counter"]["line_end"]   = [lx2, ly2]
        save_cfg(cfg_now)
        p = get_pipeline()
        if p and p.counter:
            p.counter.update_line((lx1, ly1), (lx2, ly2))
        st.success("Đã cập nhật!")

    st.divider()

    #  Analytics settings 
    st.markdown("### Analytics")
    selected_date = st.date_input("Ngày", value=datetime.today(),
                                  max_value=datetime.today())
    date_str   = selected_date.strftime("%Y-%m-%d")
    chart_type = st.selectbox("Biểu đồ", ["Cột nhóm", "Cột xếp chồng", "Đường"])
    tl_freq    = st.selectbox("Timeline", ["5min", "10min", "30min", "1h"], index=1)

    st.divider()
    st.markdown(f"{'🟢 DB ready' if DB_PATH.exists() else '🔴 DB chưa có'}")


#  MAIN — TABS

tab_live, tab_analytics = st.tabs(["Live Camera", "Analytics"])


#  TAB LIVE

with tab_live:
    p      = get_pipeline()
    status = pipeline_status()

    # Header
    h_l, h_r = st.columns([4, 1])
    with h_l:
        st.markdown("### Camera Realtime")
    with h_r:
        badge  = {"stopped": "badge-stopped", "loading": "badge-loading", "live": "badge-live"}[status]
        label  = {"stopped": "⏹ Stopped", "loading": "Loading…", "live": "🟢 Live"}[status]
        st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)

    #  KPI metrics — luôn lấy trực tiếp từ pipeline (không cache) 
    if p is not None:
        live_stats = p.get_stats()
    else:
        live_stats = {"fps": 0, "count_in": 0, "count_out": 0,
                      "occupancy": 0, "active_tracks": 0}

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("⚡ AI FPS",          live_stats["fps"])
    k2.metric("🚶 Vào (session)",   live_stats["count_in"])
    k3.metric("🚪 Ra (session)",    live_stats["count_out"])
    k4.metric("👥 Trong khu vực",   live_stats["occupancy"])

    st.divider()

    #  Video frame placeholder 
    frame_slot = st.empty()

    if status == "stopped":
        frame_slot.markdown("""
        <div style="background:#111827;border:2px dashed #374151;border-radius:12px;
                    height:400px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:12px;">
            <div style="font-size:52px">📷</div>
            <div style="color:#6b7280;font-size:16px">
                Chọn Webcam index → nhấn <b style="color:#d1d5db">▶ Start</b>
            </div>
        </div>""", unsafe_allow_html=True)

    elif status == "loading":
        elapsed = time.time() - st.session_state.get("sv_start_time", time.time())
        msgs = [
            "Loading YOLOv8 model…",
            "Loading DeepSORT Re-ID (MobileNet)…",
            "Warming up GPU…",
            "Almost ready…",
        ]
        msg = msgs[min(int(elapsed / 5), len(msgs) - 1)]

        # Lấy raw frame từ camera để hiển thị live trong lúc load
        try:
            raw = list(p.raw_queue.queue)[-1].copy() if p.raw_queue.queue else None
        except Exception:
            raw = None

        if raw is not None:
            frame_disp = raw.copy()
            fh, fw     = frame_disp.shape[:2]
            ov = frame_disp.copy()
            cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.5, frame_disp, 0.5, 0, frame_disp)

            bw, bh = min(460, fw - 40), 110
            bx = (fw - bw) // 2
            by = (fh - bh) // 2
            cv2.rectangle(frame_disp, (bx, by), (bx+bw, by+bh), (20, 20, 20), -1)
            cv2.rectangle(frame_disp, (bx-1, by-1), (bx+bw+1, by+bh+1), (0, 180, 90), 2)

            # Spinner
            for i in range(8):
                a  = math.radians((elapsed * 280) % 360 + i * 45)
                px = int(bx + 38 + 14 * math.cos(a))
                py = int(by + bh // 2 + 14 * math.sin(a))
                cv2.circle(frame_disp, (px, py), 3,
                           (0, int(255 * (i + 1) / 8), 60), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_disp, msg,
                        (bx + 65, by + 40), font, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame_disp, f"{elapsed:.1f}s elapsed",
                        (bx + 65, by + 68), font, 0.48, (120, 120, 120), 1, cv2.LINE_AA)

            # Pulse bar
            bpw = bw - 16
            bpx = bx + 8
            bpy = by + bh - 14
            cv2.rectangle(frame_disp, (bpx, bpy), (bpx+bpw, bpy+6), (40, 40, 40), -1)
            pulse = int((math.sin(elapsed * 2) + 1) / 2 * bpw)
            cv2.rectangle(frame_disp, (bpx, bpy), (bpx+pulse, bpy+6), (0, 200, 80), -1)

            rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, use_container_width=True, caption=msg)
        else:
            frame_slot.info(f" {msg}  ({elapsed:.1f}s)")

        time.sleep(0.12)
        st.rerun()

    else:  # LIVE
        #  Ambil frame terbaru 
        frame = p.get_frame()

        # Nếu không có frame mới (AI đang xử lý), dùng frame cũ từ session
        if frame is None:
            frame = st.session_state.get("sv_last_frame")
        else:
            # Lưu frame mới nhất
            st.session_state["sv_last_frame"] = frame

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps_display = live_stats["fps"]
            tracks_disp = live_stats["active_tracks"]
            frame_slot.image(
                rgb,
                use_container_width=True,
                caption=f"🟢 Live  |  {fps_display} FPS  |  {tracks_disp} tracks đang theo dõi",
            )
        else:
            frame_slot.info("⏳ Đang nhận frame đầu tiên từ AI thread…")

        # Thông tin thêm
        if live_stats["active_tracks"] == 0 and live_stats["fps"] > 0:
            st.caption(" Chưa phát hiện người trong frame — hãy bước vào vùng camera")

        #  Rerun để lấy frame tiếp (target ~12 FPS display) 
        time.sleep(0.08)
        st.rerun()


#  TAB ANALYTICS

with tab_analytics:
    st.markdown(f"### Phân tích — {date_str}")

    totals   = get_totals(date_str)
    t_in     = totals["total_in"]
    t_out    = totals["total_out"]
    t_occ    = max(0, t_in - t_out)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Tổng vào",  f"{t_in:,}")
    a2.metric("Tổng ra",   f"{t_out:,}")
    a3.metric("Còn trong", f"{t_occ:,}")
    a4.metric("Ngày",      selected_date.strftime("%d/%m/%Y"))

    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Lưu lượng theo giờ")
        hdf = hourly_df(date_str)
        if hdf.empty:
            st.info("Chưa có dữ liệu. Bật Live Camera và bước qua vạch để ghi nhận.")
        else:
            pv = hdf.pivot_table(index="hour", columns="direction",
                                 values="cnt", fill_value=0).reset_index()
            all_h = pd.DataFrame({"hour": range(24)})
            pv    = all_h.merge(pv, on="hour", how="left").fillna(0)
            for c in ["IN", "OUT"]:
                if c not in pv.columns:
                    pv[c] = 0

            cin, cout = "#00c896", "#4a90d9"
            fig = go.Figure()
            if chart_type == "Cột nhóm":
                fig.add_bar(x=pv["hour"], y=pv["IN"],  name="Vào", marker_color=cin, opacity=0.85)
                fig.add_bar(x=pv["hour"], y=pv["OUT"], name="Ra",  marker_color=cout, opacity=0.85)
                fig.update_layout(barmode="group")
            elif chart_type == "Cột xếp chồng":
                fig.add_bar(x=pv["hour"], y=pv["IN"],  name="Vào", marker_color=cin)
                fig.add_bar(x=pv["hour"], y=pv["OUT"], name="Ra",  marker_color=cout)
                fig.update_layout(barmode="stack")
            else:
                fig.add_scatter(x=pv["hour"], y=pv["IN"],  name="Vào",
                                mode="lines+markers", line=dict(color=cin, width=2))
                fig.add_scatter(x=pv["hour"], y=pv["OUT"], name="Ra",
                                mode="lines+markers", line=dict(color=cout, width=2))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Giờ", tickmode="linear", dtick=1, gridcolor="#1f2937"),
                yaxis=dict(title="Số người", gridcolor="#1f2937"),
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                height=340, margin=dict(l=8, r=8, t=30, b=8),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Timeline")
        tdf = timeline_df(date_str, tl_freq)
        if tdf.empty:
            st.info("Chưa có dữ liệu.")
        else:
            fig2 = go.Figure()
            if "IN" in tdf.columns:
                fig2.add_scatter(x=tdf["time"], y=tdf["IN"], name="Vào",
                                 fill="tozeroy",
                                 line=dict(color="#00c896", width=1.5),
                                 fillcolor="rgba(0,200,150,0.12)")
            if "OUT" in tdf.columns:
                fig2.add_scatter(x=tdf["time"], y=tdf["OUT"], name="Ra",
                                 fill="tozeroy",
                                 line=dict(color="#4a90d9", width=1.5),
                                 fillcolor="rgba(74,144,217,0.12)")
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#1f2937"),
                yaxis=dict(gridcolor="#1f2937"),
                legend=dict(orientation="h", y=1.02),
                height=340, margin=dict(l=8, r=8, t=30, b=8),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Sự kiện gần nhất")
    rdf = recent_events(80)
    if rdf.empty:
        st.info("Chưa có sự kiện crossing. Hãy bước qua vạch đếm khi Live Camera đang chạy.")
    else:
        rdf["Hướng"] = rdf["direction"].map({"IN": "🟢 Vào", "OUT": "🔵 Ra"})
        rdf = rdf.rename(columns={"timestamp": "Thời gian", "track_id": "Track ID"})
        st.dataframe(
            rdf[["Thời gian", "Track ID", "Hướng"]],
            use_container_width=True,
            height=280,
            hide_index=True,
        )

    # Analytics tab refresh chậm hơn (không cần realtime)
    time.sleep(4)
    st.rerun()
