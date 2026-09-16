"""
camera_agent.py — VisionQA Camera Agent (Streamlit) — Professional UI v5.0

Run:
    streamlit run camera_agent.py

Requires:
    pip install opencv-python-headless requests streamlit pandas Pillow
"""

import base64
import json
import os
import smtplib
import sqlite3
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# ── Config ─────────────────────────────────────────────────────
LANGFLOW_FLOW_ID = "a4468afd-7e5f-43e8-a509-59bbb024d2eb"
LANGFLOW_URL     = f"http://localhost:7860/api/v1/run/{LANGFLOW_FLOW_ID}?stream=false"
DETECT_URL       = "http://localhost:8000/api/detect"
BASE_API_URL     = "http://localhost:8000/api"
INTERVAL         = 2.0
CAMERA_IDX       = 0
DB_PATH          = "visionqa.db"

ALERT_EMAIL_ENABLED = False
ALERT_EMAIL_FROM    = "you@example.com"
ALERT_EMAIL_TO      = "team@example.com"
ALERT_SMTP_HOST     = "smtp.example.com"
ALERT_SMTP_PORT     = 587
ALERT_SMTP_USER     = ""
ALERT_SMTP_PASS     = ""

CLR_PASS = (0,   230, 118)
CLR_FAIL = (61,  61,  255)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


# ── Custom CSS ──────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Global resets ── */
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
    }
    [data-testid="stSidebar"] {
        background: #1a1d27 !important;
        border-right: 1px solid #2a2d3e;
    }
    .stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }
    h1, h2, h3 { letter-spacing: -0.02em; }

    /* ── Metric cards ── */
    .metric-card {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #3a3d5e; }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        line-height: 1;
        color: #f1f5f9;
    }
    .metric-card.pass  { border-color: #10b981; }
    .metric-card.fail  { border-color: #ef4444; }
    .metric-card.alert { border-color: #f59e0b; }

    /* ── Status badge ── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-pass  { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
    .badge-fail  { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
    .badge-true  { background: #78350f; color: #fbbf24; border: 1px solid #92400e; }
    .badge-false { background: #1e293b; color: #94a3b8; border: 1px solid #334155; }

    /* ── Feed panel ── */
    .feed-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    .feed-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #6b7280;
    }
    .live-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #10b981;
        display: inline-block;
        margin-right: 6px;
        animation: pulse 1.4s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ── Upload zone ── */
    .upload-zone {
        background: #1e2130;
        border: 2px dashed #3a3d5e;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .upload-zone:hover { border-color: #6366f1; }

    /* ── Result row ── */
    .result-row {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
    }

    /* ── Sidebar tweaks ── */
    .sidebar-section {
        background: #12151f;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .sidebar-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4b5563;
        margin-bottom: 8px;
    }

    /* ── Status bar ── */
    .status-bar {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 13px;
        color: #94a3b8;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
    }
    .status-item strong { color: #f1f5f9; }

    /* ── Streamlit overrides ── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }
    div[data-testid="stFileUploader"] > section {
        border: 2px dashed #3a3d5e !important;
        border-radius: 12px !important;
        background: #1e2130 !important;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: #6366f1 !important;
    }
    div[data-testid="stMetric"] {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ── SQLite ──────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS counters (
            id      INTEGER PRIMARY KEY DEFAULT 1,
            total   INTEGER DEFAULT 0,
            passed  INTEGER DEFAULT 0,
            failed  INTEGER DEFAULT 0,
            alerted INTEGER DEFAULT 0,
            skipped INTEGER DEFAULT 0
        )
    """)
    for col in ("alerted", "skipped"):
        try:
            cur.execute(f"ALTER TABLE counters ADD COLUMN {col} INTEGER DEFAULT 0")
        except Exception:
            pass
    cur.execute("INSERT OR IGNORE INTO counters (id) VALUES (1)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT,
            status   TEXT,
            branch   TEXT,
            defects  TEXT,
            avg_conf TEXT,
            count    INTEGER,
            source   TEXT DEFAULT 'camera'
        )
    """)
    cur.execute("PRAGMA table_info(history)")
    existing = {row[1] for row in cur.fetchall()}
    for col, defn in {
        "branch": "TEXT DEFAULT 'FALSE'",
        "defects": "TEXT DEFAULT 'None'",
        "avg_conf": "TEXT DEFAULT '—'",
        "count": "INTEGER DEFAULT 0",
        "source": "TEXT DEFAULT 'camera'",
    }.items():
        if col not in existing:
            try:
                cur.execute(f"ALTER TABLE history ADD COLUMN {col} {defn}")
            except Exception:
                pass
    con.commit()
    con.close()


def db_increment(status, branch):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    if status == "PASS":
        cur.execute("UPDATE counters SET total=total+1, passed=passed+1, skipped=skipped+1 WHERE id=1")
    elif str(branch).upper() == "TRUE":
        cur.execute("UPDATE counters SET total=total+1, failed=failed+1, alerted=alerted+1 WHERE id=1")
    else:
        cur.execute("UPDATE counters SET total=total+1, failed=failed+1, skipped=skipped+1 WHERE id=1")
    con.commit(); con.close()


def db_add_history(ts, status, branch, defects, avg_conf, count, source="camera"):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO history (ts, status, branch, defects, avg_conf, count, source) VALUES (?,?,?,?,?,?,?)",
        (ts, status, branch, defects, avg_conf, count, source),
    )
    con.commit(); con.close()


def db_get_counters():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT total, passed, failed, alerted, skipped FROM counters WHERE id=1")
    row = cur.fetchone()
    con.close()
    return row or (0, 0, 0, 0, 0)


def db_get_history(limit=20):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT ts, status, branch, defects, avg_conf, count, source "
        "FROM history ORDER BY id DESC LIMIT ?", (limit,),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def db_reset():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE counters SET total=0, passed=0, failed=0, alerted=0, skipped=0 WHERE id=1")
    cur.execute("DELETE FROM history")
    con.commit(); con.close()


# ── API helpers ─────────────────────────────────────────────────
def action_save_image_api(jpeg_bytes):
    try:
        # FIX 3: use milliseconds to avoid filename collision
        filename = f"{int(time.time() * 1000)}.jpg"

        resp = requests.post(
            f"{BASE_API_URL}/save-defect",
            files={"image": (filename, jpeg_bytes, "image/jpeg")},
            timeout=10,
        )

        print(f"[save-defect] Status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            print(f"[save-defect] ✅ Saved: {data}")
            return data.get("path", "")
        else:
            print(f"[save-defect] ❌ Error: {resp.text}")

    except Exception as e:
        print(f"[save-defect] ❌ {e}")

    return ""


def action_alert_api(defects, conf):
    try:
        requests.post(f"{BASE_API_URL}/alert", json={
            "defects": defects, "confidence": conf,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, timeout=10)
    except Exception as e:
        print(f"[alert] {e}")


def action_log_api(status, branch, conf, defects):
    try:
        payload = {
            "status": status,
            "branch": branch,
            "conf": conf,
            "defects": defects,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        resp = requests.post(
            f"{BASE_API_URL}/log",
            json=payload,
            timeout=10
        )

        if resp.status_code == 200:
            print(f"[log] ✅ Success: {resp.json()}")
        else:
            print(f"[log] ❌ Failed: {resp.status_code} | {resp.text}")

    except requests.exceptions.ConnectionError:
        print("[log] ❌ Connection error: API not running?")
    except requests.exceptions.Timeout:
        print("[log] ❌ Timeout: API too slow")
    except Exception as e:
        print(f"[log] ❌ Error: {e}")


def check_langflow():
    try:
        return requests.get("http://localhost:7860/health", timeout=5).status_code == 200
    except Exception:
        return False


def check_detect_api():
    try:
        return requests.get(f"{BASE_API_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


# ── Detection calls ─────────────────────────────────────────────
def call_langflow(frame_bgr):
    ok, enc = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg_bytes = enc.tobytes()
    b64 = base64.b64encode(jpeg_bytes).decode()
    resp = requests.post(
        LANGFLOW_URL,
        files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        data={"input_value": b64, "input_type": "chat", "output_type": "chat", "tweaks": "{}"},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    try:
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Non-JSON response: {resp.text[:200]!r}") from exc


def call_detect_direct(frame_bgr):
    ok, enc = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg_bytes = enc.tobytes()
    resp = requests.post(
        DETECT_URL,
        files={"image": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        params={"conf": 0.35, "iou": 0.50, "lightweight": False},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    try:
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Non-JSON response: {resp.text[:200]!r}") from exc


def call_detect_bytes(jpeg_bytes, conf=0.35, iou=0.50):
    resp = requests.post(
        DETECT_URL,
        files={"image": ("upload.jpg", jpeg_bytes, "image/jpeg")},
        params={"conf": conf, "iou": iou, "lightweight": False},
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def parse_langflow_response(data):
    try:
        msg_text = data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        result = json.loads(msg_text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        result = data
    return {
        "status":           result.get("status", "FAIL"),
        "branch":           result.get("branch", "FALSE"),
        "detections":       result.get("detections", []),
        "total_detections": result.get("total_detections", 0),
        "annotated_image":  result.get("annotated_image"),
    }


def annotate_frame(frame_bgr, detections, status, total_det):
    display = frame_bgr.copy()
    for d in detections:
        box = d.get("box", {})
        if box:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cv2.rectangle(display, (x1, y1), (x2, y2), CLR_FAIL, 2)
            label = f"{d['class']} {d['confidence']*100:.0f}%"
            cv2.putText(display, label, (x1, y1 - 6), FONT, 0.5, CLR_FAIL, 1, cv2.LINE_AA)
    h, w = display.shape[:2]
    colour = CLR_PASS if status == "PASS" else CLR_FAIL
    cv2.rectangle(display, (0, 0), (w, 36), (10, 15, 20), -1)
    ts_short = datetime.now().strftime("%H:%M:%S")
    cv2.putText(
        display,
        f"{status}  [{total_det} det]  {ts_short}",
        (10, 24), FONT, 0.65, colour, 2, cv2.LINE_AA,
    )
    return display


# ── Process camera frame ────────────────────────────────────────
def process_frame(frame_bgr, use_langflow=True):
    ts_full  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_short = datetime.now().strftime("%H:%M:%S")

    ok, enc = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return frame_bgr, None, f"[{ts_short}] ERROR: JPEG encode failed"
    jpeg_bytes = enc.tobytes()

    try:
        if use_langflow:
            raw  = call_langflow(frame_bgr)
            data = parse_langflow_response(raw)
        else:
            data = call_detect_direct(frame_bgr)
    except requests.exceptions.ConnectionError:
        return frame_bgr, None, f"[{ts_short}] Connection lost"
    except requests.exceptions.Timeout:
        return frame_bgr, None, f"[{ts_short}] Request timed out"
    except Exception as e:
        return frame_bgr, None, f"[{ts_short}] {e}"

    detections = data.get("detections", [])
    status     = data.get("status", "FAIL" if detections else "PASS")
    branch     = data.get("branch", "FALSE")
    total_det  = data.get("total_detections", len(detections))
    avg_conf   = (sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0)
    defect_str = ", ".join(sorted({d["class"] for d in detections})) if detections else "None"
    conf_str   = f"{avg_conf * 100:.1f}%" if detections else "—"

    if data.get("annotated_image"):
        arr     = np.frombuffer(base64.b64decode(data["annotated_image"]), dtype=np.uint8)
        display = cv2.imdecode(arr, cv2.IMREAD_COLOR) or annotate_frame(frame_bgr, detections, status, total_det)
    else:
        display = annotate_frame(frame_bgr, detections, status, total_det)

    # ── FIX 1: Save on any FAIL, not just when branch=="TRUE" ──
    saved_path = ""
    if status == "FAIL":
        saved_path = action_save_image_api(jpeg_bytes)
        action_alert_api(defect_str, conf_str)

    # Always log regardless of status
    action_log_api(status, branch, conf_str, defect_str)

    db_increment(status, branch)
    db_add_history(ts_full, status, branch, defect_str, conf_str, total_det, "camera")

    info = {
        "status": status, "branch": branch, "total_det": total_det,
        "conf": conf_str, "defects": defect_str, "ts": ts_short, "saved": saved_path,
    }
    return display, info, None


def process_uploaded_image(uploaded_file, conf=0.35, iou=0.50):
    ts_full  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_short = datetime.now().strftime("%H:%M:%S")

    # Reset file pointer (fix duplicate read issue)
    uploaded_file.seek(0)

    file_bytes = uploaded_file.read()
    nparr      = np.frombuffer(file_bytes, np.uint8)
    frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return None, None, "Could not decode image"

    # Encode again for API
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return None, None, "JPEG encode failed"
    jpeg_bytes = enc.tobytes()

    # Call detection API
    try:
        data = call_detect_bytes(jpeg_bytes, conf=conf, iou=iou)
    except Exception as e:
        return None, None, str(e)

    # Parse output
    detections = data.get("detections", [])
    has_defect = data.get("has_defect", False)

    status     = "FAIL" if has_defect else "PASS"
    total_det  = data.get("total_detections", len(detections))
    avg_conf   = (sum(d["confidence"] for d in detections) / len(detections)) if detections else 0.0

    defect_str = ", ".join(sorted({d["class"] for d in detections})) if detections else "None"
    conf_str   = f"{avg_conf * 100:.1f}%" if detections else "—"
    branch     = "TRUE" if (detections and avg_conf > 0.8) else "FALSE"

    # Render image
    if data.get("annotated_image"):
        arr     = np.frombuffer(base64.b64decode(data["annotated_image"]), dtype=np.uint8)
        display = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        display = annotate_frame(frame, detections, status, total_det)

    # ── FIX 1 (upload path): Save on any FAIL ──
    if status == "FAIL":
        action_save_image_api(jpeg_bytes)
        action_alert_api(defect_str, conf_str)

    # Log always
    action_log_api(status, branch, conf_str, defect_str)

    # DB save (always)
    db_increment(status, branch)
    db_add_history(ts_full, status, branch, defect_str, conf_str, total_det, "upload")

    info = {
        "status": status,
        "branch": branch,
        "total_det": total_det,
        "conf": conf_str,
        "defects": defect_str,
        "ts": ts_short,
        "filename": uploaded_file.name,
    }

    return display, info, None


# ── UI helpers ──────────────────────────────────────────────────
def render_metric_card(label, value, css_class=""):
    st.markdown(
        f'<div class="metric-card {css_class}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_status_info(info):
    if not info:
        return
    s_badge = f'<span class="badge badge-pass">PASS</span>' if info["status"] == "PASS" else f'<span class="badge badge-fail">FAIL</span>'
    b_badge = f'<span class="badge badge-true">TRUE</span>' if info["branch"] == "TRUE" else f'<span class="badge badge-false">FALSE</span>'
    st.markdown(
        f'<div class="status-bar">'
        f'<span>Result {s_badge}</span>'
        f'<span>Branch {b_badge}</span>'
        f'<span>Detections: <strong>{info["total_det"]}</strong></span>'
        f'<span>Conf: <strong>{info["conf"]}</strong></span>'
        f'<span>Defects: <strong>{info["defects"]}</strong></span>'
        f'<span style="color:#6b7280">{info.get("ts","")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_history_table(rows):
    if not rows:
        st.markdown('<p style="color:#6b7280; font-size:13px; text-align:center; padding:1rem;">No inspection history yet</p>', unsafe_allow_html=True)
        return

    for ts, status, branch, defects, conf, count, *src in rows:
        source = src[0] if src else "camera"
        s_badge = '<span class="badge badge-pass">PASS</span>' if status == "PASS" else '<span class="badge badge-fail">FAIL</span>'
        b_badge = '<span class="badge badge-true">TRUE</span>' if branch == "TRUE" else '<span class="badge badge-false">FALSE</span>'
        src_icon = "📷" if source == "camera" else "📁"
        st.markdown(
            f'<div class="result-row">'
            f'<span style="color:#6b7280;font-size:11px;min-width:130px">{ts}</span>'
            f'{s_badge} {b_badge}'
            f'<span style="flex:1;color:#cbd5e1">{defects}</span>'
            f'<span style="color:#6b7280">{conf}</span>'
            f'<span style="color:#4b5563;font-size:16px">{src_icon}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main UI ──────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="VisionQA",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_db()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1.5rem;">
            <div style="font-size:22px; font-weight:700; color:#f1f5f9; letter-spacing:-0.03em">
                🔬 VisionQA
            </div>
            <div style="font-size:12px; color:#4b5563; margin-top:2px;">AI Visual Inspection v5.0</div>
        </div>
        """, unsafe_allow_html=True)

        # Service status
        st.markdown('<div class="sidebar-label">Service Status</div>', unsafe_allow_html=True)
        lf_ok  = check_langflow()
        api_ok = check_detect_api()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div style="background:{"#064e3b" if lf_ok else "#450a0a"};border:1px solid {"#065f46" if lf_ok else "#7f1d1d"};border-radius:8px;padding:8px;text-align:center">'
                f'<div style="font-size:10px;color:#6b7280;margin-bottom:4px">LANGFLOW</div>'
                f'<div style="font-size:12px;font-weight:700;color:{"#34d399" if lf_ok else "#f87171"}">{"ONLINE" if lf_ok else "OFFLINE"}</div>'
                f'</div>', unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div style="background:{"#064e3b" if api_ok else "#450a0a"};border:1px solid {"#065f46" if api_ok else "#7f1d1d"};border-radius:8px;padding:8px;text-align:center">'
                f'<div style="font-size:10px;color:#6b7280;margin-bottom:4px">DETECT API</div>'
                f'<div style="font-size:12px;font-weight:700;color:{"#34d399" if api_ok else "#f87171"}">{"ONLINE" if api_ok else "OFFLINE"}</div>'
                f'</div>', unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Mode selector
        st.markdown('<div class="sidebar-label">Inspection Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "mode", ["📷  Live Camera", "📁  Browse & Upload"],
            label_visibility="collapsed",
        )

        st.divider()

        # Detection settings
        st.markdown('<div class="sidebar-label">Detection Settings</div>', unsafe_allow_html=True)
        conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05)
        iou_thresh  = st.slider("IoU threshold", 0.10, 0.95, 0.50, 0.05)

        if "camera" in mode.lower():
            interval     = st.slider("Capture interval (s)", 1.0, 10.0, INTERVAL, 0.5)
            use_langflow = st.toggle("Route via Langflow", value=True)
            st.divider()
            running = st.toggle("▶ Start Inspection", value=False)
        else:
            running      = False
            use_langflow = False
            interval     = INTERVAL

        st.divider()

        # Session controls
        st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
        if st.button("🗑 Reset counters", use_container_width=True):
            db_reset()
            st.rerun()

    # ── Main content ─────────────────────────────────────────────
    # Top metrics row
    total, passed, failed, alerted, skipped = db_get_counters()
    yield_pct = f"{(passed/total*100):.1f}%" if total > 0 else "—"

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card("Total Inspected", total)
    with m2:
        render_metric_card("Passed", passed, "pass")
    with m3:
        render_metric_card("Failed", failed, "fail")
    with m4:
        render_metric_card("Alerts Triggered", alerted, "alert")
    with m5:
        render_metric_card("Yield", yield_pct)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MODE: Browse & Upload ────────────────────────────────────
    if "upload" in mode.lower():
        col_upload, col_history = st.columns([3, 2])

        with col_upload:
            st.markdown('<div class="feed-title">📁 Image Upload & Inspection</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if not api_ok:
                st.error("Detect API is OFFLINE. Run: `uvicorn api:app --port 8000`")
                return

            uploaded_files = st.file_uploader(
                "Drop images here or click Browse",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            if uploaded_files:
                st.markdown(f'<p style="font-size:13px;color:#6b7280;margin:8px 0">{len(uploaded_files)} file(s) selected</p>', unsafe_allow_html=True)

                run_btn = st.button(
                    f"🔍  Run Inspection on {len(uploaded_files)} Image(s)",
                    type="primary",
                    use_container_width=True,
                )

                if run_btn:
                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    for i, uf in enumerate(uploaded_files):
                        status_text.markdown(f'<p style="font-size:13px;color:#94a3b8">Processing <strong>{uf.name}</strong> ({i+1}/{len(uploaded_files)})…</p>', unsafe_allow_html=True)

                        display, info, err = process_uploaded_image(uf, conf=conf_thresh, iou=iou_thresh)
                        progress_bar.progress((i + 1) / len(uploaded_files))

                        with st.container():
                            img_col, res_col = st.columns([2, 1])
                            with img_col:
                                if display is not None:
                                    st.image(
                                        cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                                        caption=uf.name,
                                        use_container_width=True,
                                    )
                            with res_col:
                                if err:
                                    st.error(err)
                                elif info:
                                    render_status_info(info)
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    if info["detections"] if "detections" in info else info.get("total_det", 0) > 0:
                                        pass

                    status_text.empty()
                    st.success(f"✅ Inspection complete — {len(uploaded_files)} image(s) processed")

            else:
                st.markdown("""
                <div style="text-align:center;padding:3rem 1rem;color:#4b5563">
                    <div style="font-size:48px;margin-bottom:1rem">📂</div>
                    <div style="font-size:16px;font-weight:600;color:#6b7280;margin-bottom:8px">No files selected</div>
                    <div style="font-size:13px">Use the file picker above to browse images from your computer.</div>
                    <div style="font-size:13px;margin-top:4px">Supports JPG, PNG, BMP, WebP</div>
                </div>
                """, unsafe_allow_html=True)

        with col_history:
            st.markdown('<div class="feed-title">📋 Recent Results</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            rows = db_get_history(30)
            render_history_table(rows)

    # ── MODE: Live Camera ────────────────────────────────────────
    else:
        col_feed, col_right = st.columns([3, 2])

        with col_feed:
            dot = '<span class="live-dot"></span>' if running else ""
            st.markdown(
                f'<div class="feed-header">'
                f'<div class="feed-title">{dot} Live Camera Feed</div>'
                f'<div style="font-size:12px;color:#4b5563">Camera {CAMERA_IDX}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            frame_slot  = st.empty()
            status_slot = st.empty()

            if not running:
                frame_slot.markdown("""
                <div style="background:#1e2130;border:1px solid #2a2d3e;border-radius:12px;
                            height:400px;display:flex;flex-direction:column;align-items:center;
                            justify-content:center;color:#4b5563;gap:12px;">
                    <div style="font-size:40px">📷</div>
                    <div style="font-size:15px;font-weight:600;color:#6b7280">Camera paused</div>
                    <div style="font-size:13px">Toggle <strong style="color:#f1f5f9">Start Inspection</strong> in the sidebar</div>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="feed-title">📋 Recent Results</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            history_slot = st.empty()

            rows = db_get_history(15)
            with history_slot.container():
                render_history_table(rows)

        if not running:
            return

        # Live camera services check
        if use_langflow and not lf_ok:
            st.error("Langflow is OFFLINE. Start it or disable 'Route via Langflow'.")
            return
        if not api_ok:
            st.error("Detect API is OFFLINE. Run: `uvicorn api:app --port 8000`")
            return

        cap = cv2.VideoCapture(CAMERA_IDX)
        if not cap.isOpened():
            st.error(f"Cannot open camera index {CAMERA_IDX}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    status_slot.warning("Camera read failed — retrying…")
                    time.sleep(0.5)
                    continue

                display, info, err = process_frame(frame, use_langflow=use_langflow)
                frame_slot.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_container_width=True)

                if err:
                    status_slot.error(err)
                elif info:
                    with status_slot.container():
                        render_status_info(info)
                    if info["status"] == "FAIL" and info.get("saved"):
                        st.toast(f"Defect saved → {info['saved']}", icon="🚨")

                # Update right panel
                total, passed, failed, alerted, skipped = db_get_counters()
                rows = db_get_history(15)
                with history_slot.container():
                    render_history_table(rows)

                time.sleep(interval)

        finally:
            cap.release()


if __name__ == "__main__":
    main()
