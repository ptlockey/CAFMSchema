
import json, base64, io, uuid
from dataclasses import dataclass, asdict
from typing import List, Optional

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="NHS Schematic (Plotly robust)", layout="wide")

# ---------------- Models ----------------
@dataclass
class Marker:
    id: str
    kind: str  # "tap" | "shower"
    x: float   # relative [0..1]
    y: float   # relative [0..1]

# ---------------- State ----------------
def init_state():
    st.session_state.setdefault("image", None)         # PIL image
    st.session_state.setdefault("img_w", None)
    st.session_state.setdefault("img_h", None)
    st.session_state.setdefault("image_data_uri", None)
    st.session_state.setdefault("markers", [])         # list[Marker]
    st.session_state.setdefault("uirevision", "keep")  # persist zoom/pan

init_state()

# ---------------- Utilities ----------------
def decode_image_from_json(obj) -> Optional[Image.Image]:
    if not isinstance(obj, dict):
        return None
    node = obj.get("pages",{}).get("0", obj)
    b64 = node.get("image_b64") or node.get("image_base64") or node.get("image")
    if not isinstance(b64, str):
        return None
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    except Exception:
        return None

def encode_image_to_data_uri(img: Image.Image, mime="image/png") -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG" if mime.endswith("png") else "JPEG")
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def make_figure(img: Image.Image, markers: List[Marker]) -> go.Figure:
    w, h = img.size
    fig = go.Figure()
    # Background image
    fig.add_layout_image(
        dict(
            source=img, x=0, y=0, sizex=w, sizey=h,
            xref="x", yref="y", layer="below", sizing="stretch"
        )
    )
    # Axes: image coordinate space
    fig.update_xaxes(range=[0, w], visible=False, constrain="domain")
    fig.update_yaxes(range=[h, 0], visible=False, scaleanchor="x", scaleratio=1)
    # Markers
    taps_x, taps_y, taps_ids = [], [], []
    shows_x, shows_y, shows_ids = [], [], []
    for m in markers:
        X = m.x * w
        Y = m.y * h
        if m.kind == "tap":
            taps_x.append(X); taps_y.append(Y); taps_ids.append(m.id)
        else:
            shows_x.append(X); shows_y.append(Y); shows_ids.append(m.id)

    if taps_x:
        fig.add_trace(go.Scatter(
            x=taps_x, y=taps_y, mode="markers",
            marker=dict(size=12, color="blue", symbol="circle"),
            name="Taps",
            text=[f"Tap {i}" for i in taps_ids],
            hovertemplate="Tap %{text}<br>x=%{x:.0f}, y=%{y:.0f}<extra></extra>"
        ))
    if shows_x:
        fig.add_trace(go.Scatter(
            x=shows_x, y=shows_y, mode="markers",
            marker=dict(size=12, color="red", symbol="diamond"),
            name="Showers",
            text=[f"Shower {i}" for i in shows_ids],
            hovertemplate="Shower %{text}<br>x=%{x:.0f}, y=%{y:.0f}<extra></extra>"
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",   # <-- critical for plotly_events
        uirevision=st.session_state.uirevision,  # keep view on rerun
    )
    return fig

def export_json() -> bytes:
    data = {
        "pages": {
            "0": {
                "mime": "image/png",
                "image_b64": st.session_state.image_data_uri,
                "markers": [asdict(m) for m in st.session_state.markers],
            }
        }
    }
    return json.dumps(data, indent=2).encode("utf-8")

# ---------------- Sidebar ----------------
st.sidebar.header("Load schematic")
source = st.sidebar.radio("Source", ["JSON (with base64 image)", "Image file (PNG/JPG)"], index=0)

if source == "JSON (with base64 image)":
    j = st.sidebar.file_uploader("Upload JSON", type=["json"])
    if j is not None:
        try:
            obj = json.loads(j.read().decode("utf-8"))
            img = decode_image_from_json(obj)
            if img is None:
                st.sidebar.error("Could not decode base64 image from JSON.")
            else:
                st.session_state.image = img
                st.session_state.img_w, st.session_state.img_h = img.size
                st.session_state.image_data_uri = encode_image_to_data_uri(img, "image/png")
                # optional: load existing markers
                node = obj.get("pages",{}).get("0",{})
                markers = node.get("markers") or []
                good = []
                for m in markers:
                    if isinstance(m, dict) and {"id","kind","x","y"} <= set(m.keys()):
                        try:
                            good.append(Marker(id=str(m["id"]), kind=str(m["kind"]), x=float(m["x"]), y=float(m["y"])))
                        except Exception:
                            pass
                if good:
                    st.session_state.markers = good
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")
else:
    f = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if f is not None:
        img = Image.open(f).convert("RGBA")
        st.session_state.image = img
        st.session_state.img_w, st.session_state.img_h = img.size
        st.session_state.image_data_uri = encode_image_to_data_uri(img, "image/png")
        st.session_state.markers = []

# Tools
st.sidebar.header("Marker tool")
tool = st.sidebar.radio("Add", ["Tap", "Shower", "None"], index=2, horizontal=True)
if st.sidebar.button("Clear markers"):
    st.session_state.markers = []

# ---------------- Main ----------------
if st.session_state.image is None:
    st.info("Upload a schematic (JSON with base64 image, or a PNG/JPG).")
    st.stop()

fig = make_figure(st.session_state.image, st.session_state.markers)

st.write("Click on the image to drop a marker. Drag to pan, scroll to zoom (Plotly controls).")

# IMPORTANT: plotly_events requires clickmode='event+select' AND returning event metadata.
events = plotly_events(
    fig,
    click_event=True, hover_event=False, select_event=False,
    override_height=min(900, st.session_state.img_h),
    key="plotly_clicks"
)

# Debug panel (optional): show last raw event so you can verify x/y coming through
with st.expander("Debug: last click event"):
    st.write(events[-1] if events else None)

if events and tool in ("Tap","Shower"):
    ev = events[-1]
    # Some environments return None if click missed data; guard it.
    if ev.get("x") is not None and ev.get("y") is not None:
        w, h = st.session_state.img_w, st.session_state.img_h
        x_rel = min(max(float(ev["x"]) / w, 0.0), 1.0)
        y_rel = min(max(float(ev["y"]) / h, 0.0), 1.0)
        st.session_state.markers.append(Marker(id=str(uuid.uuid4())[:8], kind=("tap" if tool=="Tap" else "shower"), x=x_rel, y=y_rel))

# Re-render with potential new markers
fig = make_figure(st.session_state.image, st.session_state.markers)
st.plotly_chart(fig, use_container_width=True, height=min(900, st.session_state.img_h))

# Summary + table
tap_count = sum(1 for m in st.session_state.markers if m.kind=="tap")
shower_count = sum(1 for m in st.session_state.markers if m.kind=="shower")
col1, col2 = st.columns(2)
col1.metric("Taps", tap_count)
col2.metric("Showers", shower_count)

if st.session_state.markers:
    st.subheader("Markers")
    table = [asdict(m) | {"x_px": int(m.x*st.session_state.img_w), "y_px": int(m.y*st.session_state.img_h)} for m in st.session_state.markers]
    st.dataframe(table, use_container_width=True)

# Export
st.download_button("Download JSON (image + markers)", data=export_json(), file_name="schematic_with_markers.json", mime="application/json")

