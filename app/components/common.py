"""Shared dashboard pieces: input panel, persistent state, status bar, needle gauge, banners.
Display only - every result comes from pdm.pipeline.analyze()."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from pdm.config import path  # noqa: E402

HEALTH_COLORS = {"Healthy": "#1f9d7a", "Warning": "#e0a43a", "Critical": "#d1495b"}
HEALTH_TEXT = {"Healthy": "#ffffff", "Warning": "#2b1d00", "Critical": "#ffffff"}
REPORT_KEY = "report"
HOME_PAGE = "views/overview.py"


# ------------------------------------------------------------------ cached data
@st.cache_data
def samples() -> list[dict]:
    p = path("samples_dir") / "samples.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


@st.cache_data
def metrics(name: str):
    p = path("metrics_dir") / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else p


@st.cache_resource(show_spinner="Loading models...")
def warm_up() -> bool:
    """Load the models once per server process (pdm.pipeline caches them)."""
    from pdm import pipeline
    pipeline._predictor(); pipeline._agent(); pipeline._obs_model(); pipeline._reference()
    return True


# ------------------------------------------------------------------ persistent input state
# Streamlit forgets widget values when you switch pages. The real values live in "inp" (a plain
# session-state dict); widgets are re-filled from it every time the Home page is shown again.
def inp() -> dict:
    if "inp" not in st.session_state:
        labels = [s["label"] for s in samples()]
        st.session_state["inp"] = {"source": "Bundled sample", "sample": labels[2] if len(labels) > 2 else
                                   (labels[0] if labels else None), "fs": 12000.0, "rpm": 1750.0, "load": 2.0,
                                   "upload_name": None, "upload_bytes": None}
        _apply_sample(st.session_state["inp"])
    return st.session_state["inp"]


def _apply_sample(state: dict) -> None:
    s = next((s for s in samples() if s["label"] == state.get("sample")), None)
    if s:
        state.update({"fs": float(s["fs"]), "rpm": float(s["rpm"]), "load": float(s["load_hp"])})


def _sync(key: str) -> None:
    state = inp()
    state[key] = st.session_state[f"w_{key}"]
    if key == "sample":
        _apply_sample(state)
        for k in ("fs", "rpm", "load"):
            st.session_state[f"w_{k}"] = state[k]


def _on_upload() -> None:
    f = st.session_state.get("w_upload")
    if f is not None:
        inp().update({"upload_name": f.name, "upload_bytes": f.getvalue()})


def input_panel() -> None:
    """The one place where a recording is chosen and analysed (centre of the Home page)."""
    state = inp()
    for k in ("source", "sample", "fs", "rpm", "load"):
        if f"w_{k}" not in st.session_state:
            st.session_state[f"w_{k}"] = state[k]

    with st.container(border=True):
        st.markdown("#### 1 · Choose a vibration recording")
        st.segmented_control("Source", ["Bundled sample", "Upload a file"], key="w_source",
                             on_change=_sync, args=("source",), label_visibility="collapsed")
        file_bytes, name = None, None
        if state["source"] == "Upload a file":
            st.file_uploader("Signal file (.mat, .csv or .npy)", type=["mat", "csv", "npy"], key="w_upload",
                             on_change=_on_upload)
            if state["upload_bytes"] is not None:
                file_bytes, name = state["upload_bytes"], state["upload_name"]
                c1, c2 = st.columns([4, 1], vertical_alignment="center")
                c1.caption(f"Current file: **{name}** ({len(file_bytes) / 1024:.0f} KB)")
                if c2.button("Remove file", use_container_width=True):
                    state.update({"upload_name": None, "upload_bytes": None})
                    st.rerun()
            st.caption("Tip: genuine recordings the model never saw are in `data/unseen/` (CWRU outer-race faults "
                       "@3 and @12 o'clock): 12000 Hz, 1797 RPM, 0 HP.")
        else:
            labels = [s["label"] for s in samples()]
            if not labels:
                st.warning("No bundled samples found - run scripts/01_download_data.py")
            else:
                st.selectbox("Sample recording", labels, key="w_sample", on_change=_sync, args=("sample",))
                s = next(s for s in samples() if s["label"] == state["sample"])
                name = s["file"]
                file_bytes = (path("samples_dir") / name).read_bytes()
                st.caption(f"CWRU recording {s['cwru_recording']} — {s['note']}")

        c = st.columns([1, 1, 1, 1.2], vertical_alignment="bottom")
        c[0].number_input("Sampling rate (Hz)", 1000.0, 200000.0, step=1000.0, key="w_fs", on_change=_sync,
                          args=("fs",))
        c[1].number_input("Shaft speed (RPM)", 100.0, 10000.0, step=1.0, key="w_rpm", on_change=_sync, args=("rpm",))
        c[2].number_input("Load (HP)", 0.0, 10.0, step=1.0, key="w_load", on_change=_sync, args=("load",))
        if c[3].button("Analyze", type="primary", use_container_width=True, disabled=file_bytes is None,
                       key="analyze_btn"):
            run_analysis(file_bytes, name)


def run_analysis(file_bytes: bytes, name: str) -> None:
    from pdm.data.loader import load_upload
    from pdm.pipeline import analyze

    warm_up()
    state = inp()
    try:
        with st.spinner("Analysing..."):
            x = load_upload(file_bytes, name)
            st.session_state[REPORT_KEY] = analyze(x, state["fs"], state["rpm"], state["load"], source=name)
            st.session_state["chat"] = []
    except Exception as exc:
        st.session_state.pop(REPORT_KEY, None)
        st.error(f"Could not analyse the file: {exc}")


# ------------------------------------------------------------------ report access + status bar
def get_report(show_bar: bool = True):
    rep = st.session_state.get(REPORT_KEY)
    if rep is None:
        with st.container(border=True):
            st.markdown("**No recording analysed yet.** Choose a recording and click **Analyze** on the Home page.")
            st.page_link(HOME_PAGE, label="Go to Home to analyze a recording", icon="🏠")
        return None
    if show_bar:
        status_bar(rep)
    return rep


def status_bar(rep) -> None:
    d, h, r = rep.diagnosis, rep.health, rep.recommendation
    with st.container(border=True):
        c = st.columns([2.2, 1.2, 2, 2.4, 1.3], vertical_alignment="center")
        c[0].markdown(f"**Recording**  \n`{rep.input['source']}` · {rep.input['rpm']:g} RPM · {rep.input['load_hp']:g} HP")
        c[1].html(badge(h["status"]))
        c[2].markdown(f"**{d['fault']}**  \n{d['confidence']:.0%} · {d['decision_source']}")
        c[3].markdown(f"**Recommended:** {r['action'] if r else '—'}  \n<span style='opacity:.7'>simulation-trained agent</span>",
                      unsafe_allow_html=True)
        c[4].page_link(HOME_PAGE, label="Change", icon="🔁")
    for w in rep.warnings:
        st.warning(w, icon="⚠️")


def badge(status: str, low_conf: bool = False) -> str:
    extra = " · low confidence" if low_conf else ""
    return (f"<span style='display:inline-block;background:{HEALTH_COLORS[status]};color:{HEALTH_TEXT[status]};"
            f"padding:.3rem .85rem;border-radius:999px;font-weight:700;font-size:.95rem'>{status}{extra}</span>")


# ------------------------------------------------------------------ needle gauge (speedometer)
def gauge_html(index: int, status: str, stage: str, low_conf: bool = False) -> str:
    """Semicircle gauge with an animated needle. `index` (0-100) comes from pdm.health (rule-based)."""
    import math

    cx, cy, R, w = 150, 150, 112, 26

    def arc(a0: float, a1: float) -> str:  # angles in degrees, 180 = left, 0 = right
        p0 = (cx + R * math.cos(math.radians(a0)), cy - R * math.sin(math.radians(a0)))
        p1 = (cx + R * math.cos(math.radians(a1)), cy - R * math.sin(math.radians(a1)))
        return f"M {p0[0]:.1f} {p0[1]:.1f} A {R} {R} 0 0 1 {p1[0]:.1f} {p1[1]:.1f}"

    zones = [(180, 120.6, HEALTH_COLORS["Healthy"]), (119.4, 61.2, HEALTH_COLORS["Warning"]),
             (60, 0, HEALTH_COLORS["Critical"])]
    paths = "".join(f'<path d="{arc(a0, a1)}" stroke="{c}" stroke-width="{w}" fill="none"/>' for a0, a1, c in zones)
    ticks = ""
    for v in range(0, 101, 10):
        a = math.radians(180 - v * 1.8)
        r0, r1 = R - w / 2 - 6, R - w / 2 - (14 if v % 50 == 0 else 10)
        ticks += (f'<line x1="{cx + r0 * math.cos(a):.1f}" y1="{cy - r0 * math.sin(a):.1f}" '
                  f'x2="{cx + r1 * math.cos(a):.1f}" y2="{cy - r1 * math.sin(a):.1f}" stroke="currentColor" '
                  f'stroke-opacity=".45" stroke-width="{2 if v % 50 == 0 else 1}"/>')
    labels = [("Healthy", 150), ("Warning", 90), ("Critical", 30)]
    lab = "".join(f'<text x="{cx + (R + 28) * math.cos(math.radians(a)):.1f}" y="{cy - (R + 22) * math.sin(math.radians(a)):.1f}" '
                  f'text-anchor="middle" font-size="12" fill="currentColor" fill-opacity=".8">{t}</text>' for t, a in labels)
    angle = (index - 50) * 1.8  # 0 -> -90 deg (left), 100 -> +90 deg (right)
    color = HEALTH_COLORS[status]
    flag = '<div style="font-size:.85rem;margin-top:2px">⚠️ low confidence — re-measure / inspect</div>' if low_conf else ""
    return f"""
<style>
@keyframes pdm-sweep {{ from {{ transform: rotate(-90deg); }} to {{ transform: rotate({angle:.1f}deg); }} }}
.pdm-needle {{ transform-origin: {cx}px {cy}px; transform: rotate({angle:.1f}deg);
               animation: pdm-sweep 1.4s cubic-bezier(.2,.9,.25,1.12) both; }}
</style>
<div style="text-align:center;color:inherit">
<svg viewBox="0 0 300 175" style="width:100%;max-width:340px" role="img"
     aria-label="Health gauge: {status}, condition index {index} of 100">
  {paths}{ticks}{lab}
  <g class="pdm-needle">
    <polygon points="{cx - 5},{cy} {cx},{cy - R + 18} {cx + 5},{cy}" fill="currentColor"/>
  </g>
  <circle cx="{cx}" cy="{cy}" r="11" fill="{color}" stroke="currentColor" stroke-width="2"/>
</svg>
<div style="font-size:1.6rem;font-weight:800;color:{color};margin-top:-6px">{status}</div>
<div style="opacity:.8;font-size:.9rem">{stage} · condition index {index}/100</div>{flag}
<div style="opacity:.6;font-size:.75rem;margin-top:4px">Rule-based estimate (needle position is fixed per rule outcome)</div>
</div>"""


def show_gauge(index: int, status: str, stage: str, low_conf: bool = False) -> None:
    """Render the needle gauge in its own frame (st.html strips SVG). The frame does not inherit the
    app's colours, so the current light/dark theme is passed in explicitly."""
    import streamlit.components.v1 as components

    dark = getattr(st.context.theme, "type", None) == "dark"
    fg = "#e6e9ef" if dark else "#1d2433"
    body = gauge_html(index, status, stage, low_conf)
    components.html(f"<html><body style='margin:0;font-family:sans-serif;color:{fg};background:transparent'>"
                    f"{body}</body></html>", height=300)


def simulation_banner() -> None:
    st.info("**SIMULATION** — the maintenance agent, its costs, failure probabilities and what-if risks are simulated "
            "with illustrative assumptions, not plant data.", icon="🧪")


def disclaimer() -> None:
    with st.expander("Data & model limitations (please read)"):
        from pdm.pipeline import CAVEATS
        for c in CAVEATS:
            st.markdown(f"- {c}")
