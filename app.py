import base64
import io
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Schematic + Taps/Showers", layout="wide")

# ---------------- Models ----------------
@dataclass
class Marker:
    kind: str       # "tap" or "shower"
    x: float        # relative [0..1]
    y: float        # relative [0..1]

# ---------------- State init ----------------
def init_state():
    st.session_state.setdefault("image", None)            # PIL image
    st.session_state.setdefault("image_b64", None)        # data uri or raw b64
    st.session_state.setdefault("mime", "image/png")
    st.session_state.setdefault("markers", [])            # list[Marker]
    st.session_state.setdefault("scale", 1.0)

init_state()

# ---------------- Utilities ----------------
def decode_image_from_json(obj: Dict) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Accepts several JSON layouts and returns (PIL image, mime)."""
    if not isinstance(obj, dict):
        return None, None
    node = obj
    # Common containers: {"pages":{"0":{...}}} or {"image_b64":...}
    if "pages" in obj:
        node = obj["pages"].get("0") or obj["pages"].get(0) or obj["pages"]
    if not isinstance(node, dict):
        return None, None

    mime = node.get("mime") or "image/png"
    # Key variants
    val = node.get("image_b64") or node.get("image_base64") or node.get("image")
    if not isinstance(val, str):
        return None, None

    # Strip data URI header if present
    if val.startswith("data:image"):
        try:
            header, b64 = val.split(",", 1)
            mime = header.split(";")[0].split(":")[1] or mime
        except Exception:
            b64 = val
    else:
        b64 = val

    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
        return img, mime
    except Exception:
        return None, None

def encode_image_to_data_uri(img: Image.Image, mime: str = "image/png") -> str:
    buf = io.BytesIO()
    fmt = "PNG" if mime.endswith("png") else "JPEG"
    img.save(buf, format=fmt)
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def draw_markers(base: Image.Image, markers: List[Marker]) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    r = max(6, int(min(W, H) * 0.01))

    for m in markers:
        x = int(m.x * W)
        y = int(m.y * H)
        color = (0, 150, 255, 220) if m.kind == "tap" else (255, 0, 120, 220)
        # Circle
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        # Label
        label = "T" if m.kind == "tap" else "S"
        draw.text((x + r + 2, y - r - 2), label, fill=(0, 0, 0, 255))
    return img

def export_json() -> bytes:
    # Build a compact export with embedded base64 and markers
    data_uri = st.session_state.image_b64 or encode_image_to_data_uri(st.session_state.image, st.session_state.mime)
    export = {
        "pages": {
            "0": {
                "mime": st.session_state.mime,
                "image_b64": data_uri,
                "markers": [asdict(m) for m in st.session_state.markers]
            }
        }
    }
    return json.dumps(export, indent=2).encode("utf-8")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Load schematic")

choice = st.sidebar.selectbox("Source", ["JSON (with base64 image)", "Image file (PNG/JPG)"])

if choice == "JSON (with base64 image)":
    j = st.sidebar.file_uploader("Upload JSON", type=["json"])
    if j is not None:
        try:
            obj = json.loads(j.read().decode("utf-8"))
        except Exception:
            st.sidebar.error("Not valid JSON.")
            obj = None
        if obj:
            img, mime = decode_image_from_json(obj)
            if img is None:
                st.sidebar.error("JSON did not include a valid base64-encoded image.")
            else:
                st.session_state.image = img
                st.session_state.mime = mime or "image/png"
                # Load any markers if present
                node = obj.get("pages",{}).get("0",{})
                mlist = node.get("markers") or []
                st.session_state.markers = [Marker(**m) for m in mlist if isinstance(m, dict) and "kind" in m and "x" in m and "y" in m]
                st.session_state.image_b64 = node.get("image_b64") or node.get("image_base64") or node.get("image")

else:
    f = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if f is not None:
        st.session_state.image = Image.open(f).convert("RGBA")
        st.session_state.mime = "image/png"
        st.session_state.image_b64 = None  # regenerate on export

# Marker tools
st.sidebar.header("Tools")
tool = st.sidebar.radio("Add marker", ["Tap","Shower","None"], index=2, horizontal=True)
clear = st.sidebar.button("Clear markers")

if clear:
    st.session_state.markers = []

# ---------------- Main canvas ----------------
if st.session_state.image is None:
    st.info("Upload a schematic (JSON with base64 image, or a PNG/JPG).")
    st.stop()

img = draw_markers(st.session_state.image, st.session_state.markers)

st.write("Click on the image to add markers. Zoom with the browser if needed.")
res = streamlit_image_coordinates(img, key="img_click", width=min(1200, img.width))

if res and tool in ("Tap","Shower"):
    # Convert absolute pixel to relative
    W, H = img.size
    x_rel = res["x"] / W
    y_rel = res["y"] / H
    kind = "tap" if tool == "Tap" else "shower"
    st.session_state.markers.append(Marker(kind=kind, x=x_rel, y=y_rel))

# Show a table
if st.session_state.markers:
    st.subheader("Markers")
    st.dataframe([asdict(m) for m in st.session_state.markers])

# Export
st.download_button("Download JSON with image + markers", data=export_json(), file_name="schematic_with_markers.json", mime="application/json")
