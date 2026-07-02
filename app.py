"""
Neuro-Symbolic AI Safety Inspector — Streamlit Application
=============================================================
Main entry point.  Integrates:
  • Video upload & playback
  • YOLOv8 real-time detection with low-light enhancement
  • Per-person tracking (random persistent IDs)
  • Prolog symbolic reasoning per individual
  • Violation log: Person ID → Rule violated (clean, minimal)

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import csv
import io
import os
from datetime import datetime
from PIL import Image

# ── Local modules ──
from vision.detect import SafetyDetector
from logic.prolog_interface import PrologReasoner
from utils.video_processing import (
    get_video_info,
    extract_frames,
    resize_frame,
    save_temp_video,
    format_duration,
)


# ══════════════════════════════════════════════════════════════════
# Page Config & Custom CSS
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Neuro-Symbolic AI Safety Inspector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Dark card ── */
    .inspector-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* ── Log table ── */
    .log-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    .log-table th {
        background: rgba(102,126,234,0.18);
        color: #aab4f0;
        font-weight: 600;
        padding: 0.55rem 0.9rem;
        text-align: left;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        letter-spacing: 0.03em;
        font-size: 0.78rem;
        text-transform: uppercase;
    }
    .log-table td {
        padding: 0.5rem 0.9rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        vertical-align: middle;
    }
    .log-table tr:last-child td { border-bottom: none; }
    .log-table tr:hover td { background: rgba(255,255,255,0.03); }

    /* ── Person ID pill ── */
    .pid {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(102,126,234,0.18);
        color: #a0aaff;
        border: 1px solid rgba(102,126,234,0.35);
        border-radius: 6px;
        padding: 0.18rem 0.55rem;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.08em;
    }

    /* ── Rule text ── */
    .rule-text {
        color: #ccd;
        font-size: 0.84rem;
    }

    /* ── Severity badges ── */
    .sev-critical {
        background: rgba(231,76,60,0.2); color: #e74c3c;
        border: 1px solid rgba(231,76,60,0.4);
        border-radius: 5px; padding: 0.15rem 0.5rem;
        font-weight: 700; font-size: 0.72rem; text-transform: uppercase;
    }
    .sev-high {
        background: rgba(230,126,34,0.2); color: #e67e22;
        border: 1px solid rgba(230,126,34,0.4);
        border-radius: 5px; padding: 0.15rem 0.5rem;
        font-weight: 700; font-size: 0.72rem; text-transform: uppercase;
    }
    .sev-medium {
        background: rgba(241,196,15,0.2); color: #f1c40f;
        border: 1px solid rgba(241,196,15,0.4);
        border-radius: 5px; padding: 0.15rem 0.5rem;
        font-weight: 700; font-size: 0.72rem; text-transform: uppercase;
    }

    /* ── Violation rule ID ── */
    .rule-id {
        font-family: 'JetBrains Mono', monospace;
        color: #667eea;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ── Metric card ── */
    .metric-card {
        background: linear-gradient(135deg, #1e1e30 0%, #25253d 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #888;
        font-size: 0.78rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        font-weight: 400;
        margin-top: -0.5rem;
    }

    /* ── Info badge ── */
    .info-badge {
        display: inline-block;
        background: rgba(52,152,219,0.15);
        color: #3498db;
        border: 1px solid rgba(52,152,219,0.3);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* ── Arch ── */
    .arch-flow {
        display: flex; align-items: center; justify-content: center;
        gap: 0.5rem; flex-wrap: wrap; padding: 1rem;
    }
    .arch-step {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem; font-weight: 600; color: #ccc;
    }
    .arch-arrow { color: #667eea; font-size: 1.2rem; font-weight: 700; }

    /* ── Live person panel ── */
    .person-row {
        display: flex; align-items: center; gap: 0.8rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
    }

    /* ── Pulse on violation ── */
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(231,76,60,0); }
        50%       { box-shadow: 0 0 0 6px rgba(231,76,60,0.3); }
    }
    .violation-pulse {
        animation: pulse-red 1.4s ease-in-out infinite;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "detector":          None,
        "reasoner":          None,
        "processing":        False,
        "stop_processing":   False,
        "violation_logs":    [],       # only rule-breach entries
        "frame_count":       0,
        "total_violations":  0,
        "total_persons":     0,
        "fps_display":       0.0,
        "current_persons":   [],       # latest frame person list
        "unique_ids":        set(),    # all person IDs seen so far
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ══════════════════════════════════════════════════════════════════
# Model Loading (cached)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models():
    detector = SafetyDetector(
        model_path="auto",      # auto-selects yolov8m → yolov8n
        confidence=0.45,
        simulate_gear=True,
        enhance_light=True,
        use_tta=True,
        multiscale=True,
    )
    reasoner = PrologReasoner()
    return detector, reasoner


# ══════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        frame_interval = st.slider(
            "Process every Nth frame",
            min_value=1, max_value=30, value=5,
            help="Lower = more frames analysed, higher accuracy.",
        )
        confidence = st.slider(
            "Non-person confidence threshold",
            min_value=0.10, max_value=0.90, value=0.45, step=0.05,
            help="Person class always uses 0.25 for maximum recall.",
        )

        st.markdown("**Accuracy boosters**")
        enhance_light = st.toggle(
            "🌙 Low-light enhancement",
            value=True,
            help="Gamma + bilateral + CLAHE + unsharp mask.",
        )
        use_tta = st.toggle(
            "🔁 Test-time augmentation (TTA)",
            value=True,
            help="Multi-flip/scale fusion — best accuracy, slower.",
        )
        multiscale = st.toggle(
            "🔍 Multi-scale second pass",
            value=True,
            help="1.3× upscale pass catches small/distant persons.",
        )

        st.markdown("---")
        st.markdown("### 🔧 System Info")
        detector, reasoner = load_models()
        info = reasoner.get_engine_info()
        model_name = getattr(detector.model, 'ckpt_path', 'yolov8m/n')
        model_name = os.path.basename(str(model_name)) if model_name else 'YOLOv8'

        st.markdown(
            f"""
            <div class="inspector-card" style="font-size:0.82rem;">
                <b>CV Model:</b> {model_name}<br/>
                <b>Person conf:</b> 0.25 (max recall)<br/>
                <b>Tracker:</b> IoU centroid · random IDs<br/>
                <b>Low-light:</b> Gamma + CLAHE + Unsharp<br/>
                <b>TTA:</b> {'✅' if use_tta else '❌'} &nbsp;
                <b>Multi-scale:</b> {'✅' if multiscale else '❌'}<br/>
                <b>Reasoning:</b> {info['engine']}<br/>
                <b>Rules file:</b> {'✅ Found' if info['rules_file_exists'] else '❌ Missing'}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 🏗️ Pipeline")
        st.markdown(
            """
            <div class="arch-flow">
                <div class="arch-step">📹 Video</div>
                <span class="arch-arrow">→</span>
                <div class="arch-step">🌙 Enhance</div>
                <span class="arch-arrow">→</span>
                <div class="arch-step">🤖 YOLOv8m</div>
                <span class="arch-arrow">→</span>
                <div class="arch-step">🔁 TTA</div>
                <span class="arch-arrow">→</span>
                <div class="arch-step">🔖 Track</div>
                <span class="arch-arrow">→</span>
                <div class="arch-step">🧠 Prolog</div>
                <span class="arch-arrow">→</span>
                <div class="arch-step">📋 Logs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return frame_interval, confidence, enhance_light, use_tta, multiscale


# ══════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════

def render_header():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            '<p class="main-header">🛡️ Neuro-Symbolic AI Safety Inspector</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sub-header">Maximum accuracy · Per-person tracking · Symbolic rule reasoning · Low-light robust</p>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="text-align:right; padding-top:1rem;">
                <span class="info-badge">YOLOv8</span>
                <span class="info-badge">Prolog</span>
                <span class="info-badge">CLAHE</span>
                <span class="info-badge">IoU-Tracker</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

def render_metrics():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>Frames Processed</h3>'
            f'<div class="value" style="color:#667eea;">'
            f'{st.session_state.frame_count}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h3>Rule Violations</h3>'
            f'<div class="value" style="color:#e74c3c;">'
            f'{st.session_state.total_violations}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h3>Unique Persons</h3>'
            f'<div class="value" style="color:#2ecc71;">'
            f'{len(st.session_state.unique_ids)}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><h3>Processing FPS</h3>'
            f'<div class="value" style="color:#f39c12;">'
            f'{st.session_state.fps_display:.1f}</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# Live Persons Panel (right column)
# ══════════════════════════════════════════════════════════════════

def render_live_persons(persons: list, violations: list):
    """Show a card per tracked person with current gear status."""
    if not persons:
        st.markdown(
            '<div class="inspector-card" style="text-align:center;color:#666;">'
            'Waiting for detections…</div>',
            unsafe_allow_html=True,
        )
        return

    # Build quick lookup: person_id → list of violation keys
    vio_map: dict = {}
    for v in violations:
        pid = v["person_id"]
        vio_map.setdefault(pid, []).append(v["violation"])

    for p in persons:
        pid   = p["id"]
        viols = vio_map.get(pid, [])
        compliant = len(viols) == 0

        h_icon = "⛑️ ✓" if p["helmet"] else "⛑️ ✗"
        v_icon = "🦺 ✓" if p["vest"]   else "🦺 ✗"
        h_col  = "#2ecc71" if p["helmet"] else "#e74c3c"
        v_col  = "#2ecc71" if p["vest"]   else "#e74c3c"

        pulse_cls = "" if compliant else "violation-pulse"
        border_col = "rgba(46,204,113,0.25)" if compliant else "rgba(231,76,60,0.4)"

        viol_html = ""
        for vk in viols:
            label = vk.replace("_", " ").title()
            viol_html += f'<span style="color:#ff7675;font-size:0.78rem;">⚠ {label}</span> '

        st.markdown(
            f"""
            <div class="person-row {pulse_cls}"
                 style="border-color:{border_col};">
                <span class="pid">{pid}</span>
                <span style="color:{h_col};font-size:0.85rem;">{h_icon}</span>
                <span style="color:{v_col};font-size:0.85rem;">{v_icon}</span>
                <span style="flex:1;">{viol_html if viol_html else '<span style="color:#2ecc71;font-size:0.78rem;">✓ Compliant</span>'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# Violation Log Table
# ══════════════════════════════════════════════════════════════════

def render_logs():
    """
    Clean log table: Person ID | Rule ID | Rule Violated | Severity | Frame
    Only shows actual violations — no noise entries.
    """
    logs = st.session_state.violation_logs

    if not logs:
        st.caption("No violations detected yet.")
        return

    # ── Download button ──
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["timestamp", "frame", "person_id", "rule_id", "violation", "reason", "severity"],
    )
    writer.writeheader()
    writer.writerows(logs)
    st.download_button(
        "📥 Download Violation Log (CSV)",
        data=buf.getvalue(),
        file_name="violation_log.csv",
        mime="text/csv",
    )

    # ── Table ──
    SEV_CLASS = {
        "critical": "sev-critical",
        "high":     "sev-high",
        "medium":   "sev-medium",
    }

    rows_html = ""
    for log in reversed(logs[-80:]):
        sev_cls  = SEV_CLASS.get(log["severity"], "sev-high")
        rule_lbl = log["violation"].replace("_", " ").title()
        rows_html += f"""
        <tr>
            <td><span class="pid">{log['person_id']}</span></td>
            <td><span class="rule-id">{log['rule_id']}</span></td>
            <td class="rule-text">{log['reason']}</td>
            <td><span class="{sev_cls}">{log['severity']}</span></td>
            <td style="color:#666;font-size:0.78rem;font-family:'JetBrains Mono',monospace;">
                #{log['frame']}
            </td>
        </tr>
        """

    st.markdown(
        f"""
        <div style="overflow-x:auto;">
        <table class="log-table">
            <thead>
                <tr>
                    <th>Person ID</th>
                    <th>Rule</th>
                    <th>Violation</th>
                    <th>Severity</th>
                    <th>Frame</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# Processing Pipeline
# ══════════════════════════════════════════════════════════════════

def process_video(
    video_path:       str,
    frame_interval:   int,
    confidence:       float,
    enhance_light:    bool,
    use_tta:          bool,
    multiscale:       bool,
    frame_placeholder,
    progress_bar,
):
    """Main processing loop: detect → track → reason → log."""
    detector, reasoner = load_models()
    detector.confidence    = confidence
    detector.enhance_light = enhance_light
    detector.use_tta       = use_tta
    detector.multiscale    = multiscale

    video_info   = get_video_info(video_path)
    total_frames = video_info["total_frames"]

    # Reset state
    st.session_state.processing      = True
    st.session_state.stop_processing = False
    st.session_state.frame_count     = 0
    st.session_state.total_violations = 0
    st.session_state.violation_logs  = []
    st.session_state.current_persons = []
    st.session_state.unique_ids      = set()

    for frame_no, frame, timestamp in extract_frames(video_path, frame_interval):
        if st.session_state.stop_processing:
            break

        t0 = time.time()

        # ── 1. Detect + track ──
        persons, annotated, _ = detector.detect(frame)

        # ── 2. Reason per person ──
        violations = reasoner.reason_all(persons)

        # ── 3. Update state ──
        st.session_state.frame_count += 1
        st.session_state.total_violations += len(violations)
        st.session_state.current_persons   = persons

        for p in persons:
            st.session_state.unique_ids.add(p["id"])

        # ── 4. Log ONLY violations (person_id + rule) ──
        for v in violations:
            st.session_state.violation_logs.append({
                "timestamp": timestamp,
                "frame":     frame_no,
                "person_id": v["person_id"],
                "rule_id":   v["rule_id"],
                "violation": v["violation"],
                "reason":    v["reason"],
                "severity":  v["severity"],
            })

        # ── 5. FPS ──
        elapsed = time.time() - t0
        st.session_state.fps_display = 1.0 / elapsed if elapsed > 0 else 0

        # ── 6. Update frame display ──
        rgb = cv2.cvtColor(
            resize_frame(annotated, max_width=720), cv2.COLOR_BGR2RGB
        )
        frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

        # ── 7. Progress ──
        progress = min(frame_no / max(total_frames, 1), 1.0)
        progress_bar.progress(
            progress,
            text=f"Frame {frame_no}/{total_frames} — {len(persons)} person(s) tracked",
        )

    st.session_state.processing = False
    progress_bar.progress(1.0, text="✅ Processing complete!")


# ══════════════════════════════════════════════════════════════════
# Main Layout
# ══════════════════════════════════════════════════════════════════

def main():
    frame_interval, confidence, enhance_light, use_tta, multiscale = render_sidebar()

    render_header()
    st.markdown("---")

    render_metrics()
    st.markdown("")

    col_left, col_right = st.columns([3, 2])

    # ──────────────────────────
    # LEFT — Video + Frames
    # ──────────────────────────
    with col_left:
        st.markdown("### 📹 Video Input")

        uploaded = st.file_uploader(
            "Upload workplace video",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded:
            temp_path = save_temp_video(uploaded)
            info      = get_video_info(temp_path)

            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Resolution", f"{info['width']}×{info['height']}")
            ic2.metric("FPS",        f"{info['fps']:.1f}")
            ic3.metric("Frames",     info["total_frames"])
            ic4.metric("Duration",   format_duration(info["duration_seconds"]))

            st.markdown("---")

            bc1, bc2 = st.columns(2)
            start_btn = bc1.button("▶️ Start Processing", use_container_width=True, type="primary")
            stop_btn  = bc2.button("⏹️ Stop",             use_container_width=True)

            if stop_btn:
                st.session_state.stop_processing = True

            st.markdown("---")
            st.markdown("### 🖼️ Processed Frame")
            frame_placeholder = st.empty()
            progress_bar      = st.empty()

            if start_btn:
                process_video(
                    video_path=temp_path,
                    frame_interval=frame_interval,
                    confidence=confidence,
                    enhance_light=enhance_light,
                    use_tta=use_tta,
                    multiscale=multiscale,
                    frame_placeholder=frame_placeholder,
                    progress_bar=progress_bar,
                )
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        else:
            st.markdown(
                """
                <div class="inspector-card" style="text-align:center;padding:3rem;">
                    <div style="font-size:3rem;margin-bottom:1rem;">📹</div>
                    <div style="color:#888;font-size:1.1rem;">
                        Upload a workplace video to begin safety inspection
                    </div>
                    <div style="color:#666;font-size:0.85rem;margin-top:0.5rem;">
                        Supports MP4, AVI, MOV, MKV · Works in dim light
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ──────────────────────────
    # RIGHT — Live Person Status
    # ──────────────────────────
    with col_right:
        st.markdown("### 👷 Live Person Status")

        # Build latest violations for display
        live_violations = []
        if st.session_state.violation_logs:
            # Grab violations from the most recently processed frame
            latest_frame = st.session_state.violation_logs[-1]["frame"]
            live_violations = [
                v for v in st.session_state.violation_logs
                if v["frame"] == latest_frame
            ]

        render_live_persons(
            st.session_state.current_persons,
            live_violations,
        )

    # ──────────────────────────
    # FULL-WIDTH — Violation Log
    # ──────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Violation Log — Person ID · Rule Violated")
    render_logs()

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#555;font-size:0.8rem;padding:1rem;">
            <b>Neuro-Symbolic AI Safety Inspector</b>
            — YOLOv8 · IoU Tracker · Prolog Rules · CLAHE Low-light Enhancement<br/>
            Built with Streamlit · Ultralytics · SWI-Prolog
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
