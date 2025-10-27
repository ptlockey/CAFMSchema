import base64
import io
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image, ImageDraw
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(page_title="Building Schematic Viewer", layout="wide")


@dataclass
class Marker:
    kind: str
    x: float
    y: float


# ------------------------- Session helpers -------------------------

def _init_state() -> None:
    if "markers" not in st.session_state:
        st.session_state.markers: List[Marker] = []
    if "base_image" not in st.session_state:
        st.session_state.base_image: Optional[Image.Image] = None
    if "last_click" not in st.session_state:
        st.session_state.last_click: Optional[Dict[str, float]] = None


_init_state()


# ------------------------- JSON ingestion -------------------------

def _decode_image_from_json(data: Dict) -> Optional[Image.Image]:
    """Return an image from the provided JSON payload."""

    image_entry = (
        data.get("image_base64")
        or data.get("image_bytes_b64")
        or data.get("image")
    )
    if not image_entry:
        return None

    if isinstance(image_entry, dict):
        image_entry = image_entry.get("data")

    if not isinstance(image_entry, str):
        return None

    if image_entry.startswith("data:image"):
        try:
            image_entry = image_entry.split(",", 1)[1]
        except IndexError:
            return None

    if isinstance(image_entry, str):
        normalised = "".join(image_entry.split())
        padding = len(normalised) % 4
        if padding:
            normalised += "=" * (4 - padding)
    else:
        normalised = image_entry

    try:
        image_bytes = base64.b64decode(normalised)
    except (base64.binascii.Error, TypeError, ValueError):
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except (OSError, ValueError):
        return None

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def _markers_from_json(data: Dict) -> List[Marker]:
    markers: List[Marker] = []
    fixtures = data.get("fixtures", [])
    if not isinstance(fixtures, list):
        return markers

    for entry in fixtures:
        if not isinstance(entry, dict):
            continue
        fixture_type = entry.get("type")
        if fixture_type not in {"tap", "shower"}:
            continue

        if "point" in entry and isinstance(entry["point"], (list, tuple)) and len(entry["point"]) >= 2:
            x, y = entry["point"][0], entry["point"][1]
        else:
            x = entry.get("x")
            y = entry.get("y")

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue

        # Normalise coordinates if they look absolute
        if x > 1 or y > 1:
            width = entry.get("image_width") or data.get("image_width")
            height = entry.get("image_height") or data.get("image_height")
            if isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0:
                x = x / float(width)
                y = y / float(height)
            else:
                continue

        markers.append(Marker(kind=fixture_type, x=float(x), y=float(y)))
    return markers


def _load_layout(upload) -> None:
    if upload is None:
        return

    try:
        payload = upload.read()
    except Exception:  # pragma: no cover - streamlit handles IO errors
        st.error("Could not read uploaded file.")
        return

    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        st.error("Uploaded file is not valid JSON.")
        return

    if not isinstance(parsed, dict):
        st.error("Expected the JSON file to contain an object at the root level.")
        return

    image = _decode_image_from_json(parsed)
    if image is None:
        st.error("The JSON file does not include a valid base64 encoded image.")
        return

    st.session_state.base_image = image
    st.session_state.markers = _markers_from_json(parsed)
    st.session_state.last_click = None


# ------------------------- Rendering helpers -------------------------

def _draw_markers(image: Image.Image, markers: List[Marker]) -> Image.Image:
    if not markers:
        return image

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    radius = max(6, int(min(width, height) * 0.01))

    for marker in markers:
        color = "#0099ff" if marker.kind == "tap" else "#ff3366"
        cx = marker.x * width
        cy = marker.y * height
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, fill=color, outline="#1f1f1f", width=2)
    return annotated


def _render_viewer(image: Image.Image, zoom: float, active_tool: str) -> None:
    width, height = image.size
    display_width = int(width * zoom)
    display_height = int(height * zoom)

    annotated = _draw_markers(image, st.session_state.markers)

    st.write("Click on the schematic to add the selected fixture type.")
    result = streamlit_image_coordinates(
        annotated,
        width=display_width,
        key=f"schematic_view_{display_width}_{display_height}",
    )

    if result:
        x = result.get("x")
        y = result.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return

        # Avoid recording duplicate clicks after reruns
        if result != st.session_state.last_click:
            if result.get("width") and result.get("height"):
                display_width = result["width"]
                display_height = result["height"]

            if display_width <= 0 or display_height <= 0:
                return

            x_norm = max(0.0, min(1.0, x / display_width))
            y_norm = max(0.0, min(1.0, y / display_height))

            st.session_state.last_click = dict(result)
            st.session_state.markers.append(
                Marker(kind=active_tool, x=x_norm, y=y_norm)
            )
            st.experimental_rerun()


# ------------------------- UI -------------------------

st.title("Building Schematic Viewer")

with st.sidebar:
    st.header("Load schematic")
    uploaded = st.file_uploader("Upload a layout JSON file", type="json")
    if st.button("Load file", use_container_width=True):
        _load_layout(uploaded)

    st.divider()

    st.header("Tools")
    tool = st.radio("Fixture type", options=("tap", "shower"), index=0, horizontal=True)
    zoom = st.slider("Zoom", min_value=0.25, max_value=3.0, value=1.0, step=0.05)

    if st.button("Clear fixtures", use_container_width=True):
        st.session_state.markers = []
        st.session_state.last_click = None

    if st.session_state.markers:
        st.subheader("Fixtures")
        for idx, marker in enumerate(st.session_state.markers, start=1):
            st.write(f"{idx}. {marker.kind.title()} at ({marker.x:.3f}, {marker.y:.3f})")


if st.session_state.base_image is None:
    st.info("Upload a JSON file that contains a base64 encoded schematic image.")
else:
    try:
        _render_viewer(st.session_state.base_image, zoom, tool)
    except Exception as exc:  # pragma: no cover - runtime safeguards
        st.error(f"Could not render the schematic: {exc}")
