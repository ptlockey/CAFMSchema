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
    id: str         # unique identifier e.g. T-001
    kind: str       # "tap" or "shower"
    x: float        # relative [0..1]
    y: float        # relative [0..1]

# ---------------- State init ----------------
def init_state():
    st.session_state.setdefault("image", None)            # Display image with markers
    st.session_state.setdefault("base_image", None)       # Original PIL image
    st.session_state.setdefault("image_b64", None)        # data uri or raw b64
    st.session_state.setdefault("mime", "image/png")
    st.session_state.setdefault("markers", [])            # list[Marker]
    st.session_state.setdefault("id_counters", {"tap": 0, "shower": 0})
    st.session_state.setdefault("scale", 1.0)
    st.session_state.setdefault("pan_x", 0.0)
    st.session_state.setdefault("pan_y", 0.0)
    st.session_state.setdefault("tool_choice", "None")
    st.session_state.setdefault("last_click", None)

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

def _numeric_suffix(identifier: str) -> Optional[int]:
    digits = "".join(ch for ch in identifier if ch.isdigit())
    return int(digits) if digits else None


def _reset_id_counters(markers: List[Marker]):
    counters = {"tap": 0, "shower": 0}
    for marker in markers:
        suffix = _numeric_suffix(marker.id) if isinstance(marker.id, str) else None
        if suffix is not None:
            counters.setdefault(marker.kind, 0)
            counters[marker.kind] = max(counters[marker.kind], suffix)
    st.session_state.id_counters = counters


def next_marker_id(kind: str, existing: Optional[set] = None) -> str:
    """Return next unique marker id for the given kind."""
    if existing is None:
        existing = {m.id for m in st.session_state.markers}
    prefix = "T" if kind == "tap" else "S"
    counters = st.session_state.id_counters
    counters.setdefault(kind, 0)
    candidate = ""
    while not candidate or candidate in existing:
        counters[kind] += 1
        candidate = f"{prefix}-{counters[kind]:03d}"
    return candidate


def load_markers_from_json(raw_list: List[Dict]) -> List[Marker]:
    markers: List[Marker] = []
    counters = {"tap": 0, "shower": 0}
    used_ids = {"tap": set(), "shower": set()}
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("kind")
        if kind not in ("tap", "shower"):
            continue
        try:
            x = float(raw["x"])
            y = float(raw["y"])
        except (KeyError, TypeError, ValueError):
            continue
        identifier = raw.get("id") if isinstance(raw.get("id"), str) else ""
        identifier = identifier.strip()
        if not identifier or identifier in used_ids[kind]:
            identifier = ""
        suffix = _numeric_suffix(identifier) if identifier else None
        if suffix is not None:
            counters[kind] = max(counters[kind], suffix)
        if not identifier:
            counters[kind] += 1
            identifier = f"{'T' if kind == 'tap' else 'S'}-{counters[kind]:03d}"
        used_ids[kind].add(identifier)
        markers.append(Marker(id=identifier, kind=kind, x=x, y=y))
    st.session_state.id_counters = counters
    return markers


def draw_markers(base: Image.Image, markers: List[Marker]) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    W, H = img.size
    r = max(6, int(min(W, H) * 0.012))

    for m in markers:
        x = int(m.x * W)
        y = int(m.y * H)
        if not (0 <= x <= W and 0 <= y <= H):
            continue
        color = (220, 32, 32, 230) if m.kind == "tap" else (30, 108, 255, 230)
        label_color = (0, 0, 0, 255)
        # Circle
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0, 200), width=1)
        # Label with simple background for readability
        text = m.id
        text_size = draw.textbbox((0, 0), text, font=font)
        text_w = text_size[2] - text_size[0]
        text_h = text_size[3] - text_size[1]
        pad = 2
        box = (x + r + pad, y - text_h // 2 - pad, x + r + pad + text_w + pad, y - text_h // 2 + text_h + pad)
        draw.rectangle(box, fill=(255, 255, 255, 220))
        draw.text((box[0] + pad, box[1] + pad // 2), text, fill=label_color, font=font)
    return img


def build_view_image(base: Image.Image, markers: List[Marker], scale: float, pan_x: float, pan_y: float) -> Tuple[Image.Image, Dict[str, float]]:
    annotated = draw_markers(base, markers)
    W, H = annotated.size
    if scale <= 1.0:
        return annotated, {"offset_x": 0.0, "offset_y": 0.0, "scale": 1.0}

    scaled_size = (int(W * scale), int(H * scale))
    zoomed = annotated.resize(scaled_size, Image.BICUBIC)
    max_x = max(scaled_size[0] - W, 0)
    max_y = max(scaled_size[1] - H, 0)
    offset_x = int(max_x * min(max(pan_x, 0.0), 1.0))
    offset_y = int(max_y * min(max(pan_y, 0.0), 1.0))
    cropped = zoomed.crop((offset_x, offset_y, offset_x + W, offset_y + H))
    return cropped, {"offset_x": offset_x, "offset_y": offset_y, "scale": scale}

def export_json() -> bytes:
    # Build a compact export with embedded base64 and markers
    base_img = st.session_state.base_image or st.session_state.image
    data_uri = st.session_state.image_b64 or encode_image_to_data_uri(base_img, st.session_state.mime)
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
                st.session_state.base_image = img
                st.session_state.mime = mime or "image/png"
                # Load any markers if present
                node = obj.get("pages",{}).get("0",{})
                mlist = node.get("markers") or []
                st.session_state.markers = load_markers_from_json(mlist)
                st.session_state.image_b64 = node.get("image_b64") or node.get("image_base64") or node.get("image")
                st.session_state.last_click = None
                st.session_state.scale = 1.0
                st.session_state.pan_x = 0.0
                st.session_state.pan_y = 0.0

else:
    f = st.sidebar.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if f is not None:
        st.session_state.base_image = Image.open(f).convert("RGBA")
        st.session_state.mime = "image/png"
        st.session_state.image_b64 = None  # regenerate on export
        st.session_state.last_click = None
        st.session_state.markers = []
        _reset_id_counters([])
        st.session_state.scale = 1.0
        st.session_state.pan_x = 0.0
        st.session_state.pan_y = 0.0

# Marker tools
st.sidebar.header("Tools")
tool_options = ["Tap", "Shower", "None"]
tool = st.sidebar.radio(
    "Add marker",
    tool_options,
    key="tool_choice",
    horizontal=True,
)
clear = st.sidebar.button("Clear markers")

if clear:
    st.session_state.markers = []
    st.session_state.last_click = None
    _reset_id_counters([])

# View controls
st.sidebar.header("View controls")
scale = st.sidebar.slider(
    "Zoom",
    min_value=1.0,
    max_value=4.0,
    value=float(st.session_state.scale),
    step=0.1,
)
st.session_state.scale = scale

if scale > 1.0:
    st.session_state.pan_x = st.sidebar.slider(
        "Pan X",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.pan_x),
        step=0.01,
    )
    st.session_state.pan_y = st.sidebar.slider(
        "Pan Y",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.pan_y),
        step=0.01,
    )
else:
    st.session_state.pan_x = 0.0
    st.session_state.pan_y = 0.0
    st.sidebar.caption("Zoom in to enable pan controls.")

# ---------------- Main canvas ----------------
base_image = st.session_state.base_image or st.session_state.image

if base_image is None:
    st.info("Upload a schematic (JSON with base64 image, or a PNG/JPG).")
    st.stop()

display_img, view_params = build_view_image(
    base_image,
    st.session_state.markers,
    st.session_state.scale,
    st.session_state.pan_x,
    st.session_state.pan_y,
)
st.session_state.image = display_img
st.session_state.view_params = view_params

st.write("Click on the image to add markers.")

res = streamlit_image_coordinates(
    st.session_state.image, key="img_click", width=min(1200, st.session_state.image.width)
)

if res and tool in ("Tap", "Shower"):
    base_W, base_H = base_image.size
    scale_factor = max(view_params.get("scale", 1.0), 1e-6)
    offset_x = view_params.get("offset_x", 0.0)
    offset_y = view_params.get("offset_y", 0.0)
    x_display = res.get("x", 0)
    y_display = res.get("y", 0)
    x_rel = (offset_x + x_display) / (scale_factor * base_W)
    y_rel = (offset_y + y_display) / (scale_factor * base_H)
    x_rel = min(max(x_rel, 0.0), 1.0)
    y_rel = min(max(y_rel, 0.0), 1.0)
    kind = "tap" if tool == "Tap" else "shower"
    # Append only if coordinates actually changed
    if not st.session_state.get("last_click") == (x_rel, y_rel, kind):
        identifier = next_marker_id(kind)
        st.session_state.markers.append(Marker(id=identifier, kind=kind, x=x_rel, y=y_rel))
        st.session_state.last_click = (x_rel, y_rel, kind)

# Marker statistics
tap_count = sum(1 for m in st.session_state.markers if m.kind == "tap")
shower_count = sum(1 for m in st.session_state.markers if m.kind == "shower")
col_tap, col_shower = st.columns(2)
col_tap.metric("Taps", tap_count)
col_shower.metric("Showers", shower_count)

# Show a table
if st.session_state.markers:
    st.subheader("Markers")
    base_W, base_H = base_image.size
    table_rows = []
    for marker in sorted(st.session_state.markers, key=lambda m: m.id):
        table_rows.append(
            {
                "ID": marker.id,
                "Type": marker.kind.title(),
                "X (relative)": round(marker.x, 4),
                "Y (relative)": round(marker.y, 4),
                "X (px)": int(round(marker.x * base_W)),
                "Y (px)": int(round(marker.y * base_H)),
            }
        )
    st.dataframe(table_rows, use_container_width=True)

# Export
st.download_button("Download JSON with image + markers", data=export_json(), file_name="schematic_with_markers.json", mime="application/json")
