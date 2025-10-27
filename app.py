import base64
import io
import json
import uuid
from typing import Dict, List, Optional, Tuple

from PIL import Image
from shapely.geometry import Point, Polygon
import streamlit as st
from streamlit_drawable_canvas import st_canvas, st_image as _canvas_st_image


def _init_session_state() -> None:
    if "floors" not in st.session_state:
        st.session_state["floors"] = {}
    if "active_floor" not in st.session_state:
        st.session_state["active_floor"] = None
    if "active_room_id" not in st.session_state:
        st.session_state["active_room_id"] = None
    if "drawing_entity" not in st.session_state:
        st.session_state["drawing_entity"] = "Room"
    if "needs_canvas_refresh" not in st.session_state:
        st.session_state["needs_canvas_refresh"] = False
    if "processed_uploads" not in st.session_state:
        st.session_state["processed_uploads"] = set()


_init_session_state()


# streamlit-drawable-canvas expects the deprecated ``st_image.image_to_url`` helper.
# Streamlit 1.41+ moved that functionality to ``streamlit.elements.lib.image_utils``.
# Provide a small compatibility shim so the component can keep working on
# modern Streamlit versions without crashing at import time.
if not hasattr(_canvas_st_image, "image_to_url"):
    from streamlit.elements.lib import image_utils as _image_utils
    from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        width_value = width if width is not None else "content"
        layout_config = _LayoutConfig(width=width_value)
        return _image_utils.image_to_url(
            image,
            layout_config=layout_config,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    _canvas_st_image.image_to_url = _image_to_url_compat

# ---------- Utility conversions ----------

def _load_image(file) -> Image.Image:
    image = Image.open(file)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def _load_image_bytes(data: bytes) -> Image.Image:
    return _load_image(io.BytesIO(data))


def _image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _normalize_point(x: float, y: float, width: float, height: float) -> Tuple[float, float]:
    return x / width, y / height


def _denormalize_point(x: float, y: float, width: float, height: float) -> Tuple[float, float]:
    return x * width, y * height


def _polygon_to_path(points: List[Tuple[float, float]], width: float, height: float) -> List[List[float]]:
    if not points:
        return []
    path: List[List[float]] = []
    for idx, (x, y) in enumerate(points):
        abs_x, abs_y = _denormalize_point(x, y, width, height)
        cmd = "M" if idx == 0 else "L"
        path.append([cmd, abs_x, abs_y])
    # close polygon
    first_x, first_y = _denormalize_point(points[0][0], points[0][1], width, height)
    path.append(["L", first_x, first_y])
    path.append(["Z"])
    return path


def _points_from_path(obj: Dict, width: float, height: float) -> List[Tuple[float, float]]:
    obj_type = obj.get("type")
    points: List[Tuple[float, float]] = []

    if obj_type == "path":
        path = obj.get("path", [])
        scale_x = obj.get("scaleX", 1)
        scale_y = obj.get("scaleY", 1)
        for command in path:
            if not command:
                continue
            opcode = command[0]
            if opcode in ("M", "L") and len(command) >= 3:
                x = command[1] * scale_x
                y = command[2] * scale_y
                points.append(_normalize_point(x, y, width, height))
    elif obj_type in {"polygon", "polyline"}:
        scale_x = obj.get("scaleX", 1)
        scale_y = obj.get("scaleY", 1)
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        offset_x = obj.get("pathOffset", {}).get("x", 0)
        offset_y = obj.get("pathOffset", {}).get("y", 0)
        for pt in obj.get("points", []):
            x = (left + (pt["x"] - offset_x) * scale_x)
            y = (top + (pt["y"] - offset_y) * scale_y)
            points.append(_normalize_point(x, y, width, height))
    elif obj_type == "rect":
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        rect_width = obj.get("width", 0) * obj.get("scaleX", 1)
        rect_height = obj.get("height", 0) * obj.get("scaleY", 1)
        pts = [
            (left, top),
            (left + rect_width, top),
            (left + rect_width, top + rect_height),
            (left, top + rect_height),
        ]
        points = [_normalize_point(x, y, width, height) for x, y in pts]
    return points


def _point_from_circle(obj: Dict, width: float, height: float) -> Tuple[float, float]:
    left = obj.get("left", 0)
    top = obj.get("top", 0)
    radius = obj.get("radius", 0) * obj.get("scaleX", 1)
    # If radius missing, derive from width/height
    if radius == 0:
        radius = (obj.get("width", 0) * obj.get("scaleX", 1)) / 2
    x = left + radius
    y = top + radius
    return _normalize_point(x, y, width, height)


def _circle_object(x: float, y: float, width: float, height: float, color: str, radius: float, data: Dict) -> Dict:
    abs_x, abs_y = _denormalize_point(x, y, width, height)
    return {
        "type": "circle",
        "left": abs_x - radius,
        "top": abs_y - radius,
        "radius": radius,
        "fill": color,
        "stroke": "#1f1f1f",
        "strokeWidth": 2,
        "opacity": 0.9,
        "data": data,
        "selectable": True,
        "evented": True,
    }


def _room_to_object(room: Dict, width: float, height: float, is_selected: bool) -> Dict:
    color = room.get("color", "#1f77b455")
    if is_selected:
        color = room.get("color", "#1f77b4")
    path = _polygon_to_path(room["polygon"], width, height)
    return {
        "type": "path",
        "path": path,
        "fill": color,
        "stroke": "#0f0f0f",
        "strokeWidth": 2 if not is_selected else 4,
        "opacity": 0.5 if not is_selected else 0.75,
        "data": {
            "entity": "room",
            "room_id": room["id"],
        },
        "selectable": True,
        "evented": True,
    }


# ---------- Floor data helpers ----------

def _ensure_floor(name: str, image: Image.Image) -> str:
    floors: Dict[str, Dict] = st.session_state["floors"]
    base_name = name
    index = 1
    while name in floors:
        name = f"{base_name} ({index})"
        index += 1
    floors[name] = {
        "image_bytes": _image_to_bytes(image),
        "image_size": image.size,
        "rooms": {},
    }
    st.session_state["active_floor"] = name
    st.session_state["needs_canvas_refresh"] = True
    return name


def _register_uploaded_file(file) -> bool:
    """Return True if this upload hasn't been processed yet."""

    identifier = getattr(file, "file_id", None)
    if identifier is None:
        size = getattr(file, "size", None)
        identifier = f"{file.name}:{size}"

    processed = st.session_state.setdefault("processed_uploads", set())
    if identifier in processed:
        return False

    processed.add(identifier)
    return True


def _ingest_layout_json(payload: bytes, source_name: str) -> None:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        st.error(f"Could not parse JSON layout from {source_name}.")
        return

    floors_data: List[Dict]
    if isinstance(data, list):
        floors_data = data
    elif isinstance(data, dict):
        floors_data = data.get("floors", [])  # type: ignore[assignment]
        if not isinstance(floors_data, list):
            st.error(f"JSON layout in {source_name} is not formatted correctly.")
            return
    else:
        st.error(f"JSON layout in {source_name} is not formatted correctly.")
        return

    if not floors_data:
        st.warning(f"No floors found in {source_name}.")
        return

    for index, entry in enumerate(floors_data):
        if not isinstance(entry, dict):
            continue
        floor_name = entry.get("floor") or f"Imported floor {index + 1}"
        image_b64 = entry.get("image_bytes_b64")
        if not image_b64:
            st.warning(f"Skipping floor '{floor_name}' – missing schematic image data.")
            continue
        try:
            image_bytes = base64.b64decode(image_b64)
        except (base64.binascii.Error, TypeError):
            st.warning(f"Skipping floor '{floor_name}' – schematic image data is invalid.")
            continue

        image = _load_image_bytes(image_bytes)
        floor_key = _ensure_floor(floor_name, image)
        floor = st.session_state["floors"][floor_key]

        rooms = entry.get("rooms", [])
        if isinstance(rooms, list):
            keyed_rooms: Dict[str, Dict] = {}
            for room in rooms:
                if not isinstance(room, dict):
                    continue
                room_id = room.get("id")
                if not room_id:
                    continue
                room.setdefault("fixtures", [])
                keyed_rooms[room_id] = room
            floor["rooms"] = keyed_rooms
        else:
            floor["rooms"] = {}

        st.session_state["needs_canvas_refresh"] = True


def _get_floor_image(floor_key: str) -> Image.Image:
    floor = st.session_state["floors"][floor_key]
    return Image.open(io.BytesIO(floor["image_bytes"]))


def _get_floor_image_data_url(floor_key: str) -> str:
    """Return the cached floor image as a base64 data URL string."""
    floor = st.session_state["floors"][floor_key]
    encoded = base64.b64encode(floor["image_bytes"]).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_canvas_objects(floor_key: str) -> List[Dict]:
    floor = st.session_state["floors"][floor_key]
    width, height = floor["image_size"]
    objects: List[Dict] = []
    active_room = st.session_state.get("active_room_id")

    for room in floor["rooms"].values():
        objects.append(_room_to_object(room, width, height, room["id"] == active_room))
        for fixture in room.get("fixtures", []):
            color = "#0099ff" if fixture["type"] == "tap" else "#ff3366"
            objects.append(
                _circle_object(
                    fixture["point"][0],
                    fixture["point"][1],
                    width,
                    height,
                    color,
                    radius=8,
                    data={
                        "entity": "fixture",
                        "fixture_id": fixture["id"],
                        "room_id": room["id"],
                        "fixture_type": fixture["type"],
                    },
                )
            )

    return objects


def _serialize_canvas(objects: List[Dict]) -> Dict:
    return {"version": "5.2.4", "objects": objects}


def _create_room_from_object(
    floor_key: str,
    obj: Dict,
) -> Optional[str]:
    floor = st.session_state["floors"][floor_key]
    width, height = floor["image_size"]
    polygon = _points_from_path(obj, width, height)
    if len(polygon) < 3:
        return None

    room_id = str(uuid.uuid4())
    room = {
        "id": room_id,
        "name": f"Room {len(floor['rooms']) + 1}",
        "status": "Unspecified",
        "notes": "",
        "color": "#1f77b455",
        "polygon": polygon,
        "fixtures": [],
    }
    floor["rooms"][room_id] = room
    st.session_state["active_room_id"] = room_id
    return room_id


def _assign_fixture_to_room(floor_key: str, obj: Dict, fixture_type: str) -> Optional[str]:
    floor = st.session_state["floors"][floor_key]
    width, height = floor["image_size"]
    point = _point_from_circle(obj, width, height)
    for room in floor["rooms"].values():
        polygon = Polygon(room["polygon"])
        if polygon.contains(Point(point)):
            fixture_id = str(uuid.uuid4())
            room.setdefault("fixtures", []).append(
                {
                    "id": fixture_id,
                    "type": fixture_type,
                    "point": point,
                }
            )
            return fixture_id
    return None


def _capture_new_objects(floor_key: str, new_objects: List[Dict]) -> None:
    if not new_objects:
        return
    entity = st.session_state.get("drawing_entity", "Room")
    if entity == "Room":
        created = False
        for obj in new_objects:
            if obj.get("type") not in {"path", "polygon", "rect", "polyline"}:
                continue
            room_id = _create_room_from_object(floor_key, obj)
            created = created or room_id is not None
        if created:
            st.session_state["needs_canvas_refresh"] = True
    elif entity in {"Tap", "Shower"}:
        fixture_type = "tap" if entity == "Tap" else "shower"
        created = False
        for obj in new_objects:
            if obj.get("type") != "circle":
                continue
            fixture_id = _assign_fixture_to_room(floor_key, obj, fixture_type)
            created = created or fixture_id is not None
        if created:
            st.session_state["needs_canvas_refresh"] = True


def _sync_existing_entities(floor_key: str, objects: List[Dict]) -> None:
    floor = st.session_state["floors"][floor_key]
    width, height = floor["image_size"]

    seen_rooms = set()
    seen_fixtures = set()
    for obj in objects:
        data = obj.get("data", {})
        entity = data.get("entity")
        if entity == "room":
            room_id = data.get("room_id")
            if not room_id or room_id not in floor["rooms"]:
                continue
            polygon = _points_from_path(obj, width, height)
            if len(polygon) >= 3:
                floor["rooms"][room_id]["polygon"] = polygon
            color = obj.get("fill") or floor["rooms"][room_id].get("color", "#1f77b455")
            floor["rooms"][room_id]["color"] = color
            seen_rooms.add(room_id)
        elif entity == "fixture":
            fixture_id = data.get("fixture_id")
            room_id = data.get("room_id")
            if not (fixture_id and room_id and room_id in floor["rooms"]):
                continue
            point = _point_from_circle(obj, width, height)
            for fixture in floor["rooms"][room_id].get("fixtures", []):
                if fixture["id"] == fixture_id:
                    fixture["point"] = point
                    seen_fixtures.add(fixture_id)
                    break

    # Remove entities that were deleted on the canvas
    for room_id in list(floor["rooms"].keys()):
        if room_id not in seen_rooms:
            del floor["rooms"][room_id]
            if st.session_state.get("active_room_id") == room_id:
                st.session_state["active_room_id"] = None

    for room in floor["rooms"].values():
        room["fixtures"] = [fixture for fixture in room.get("fixtures", []) if fixture["id"] in seen_fixtures]



def _detect_active_room(objects: List[Dict]) -> Optional[str]:
    for obj in objects:
        data = obj.get("data", {})
        if data.get("entity") == "room" and obj.get("shadow"):
            return data.get("room_id")
    return None


def _room_summary(room: Dict) -> str:
    fixtures = room.get("fixtures", [])
    taps = len([f for f in fixtures if f["type"] == "tap"])
    showers = len([f for f in fixtures if f["type"] == "shower"])
    parts = [f"Fixtures: {taps} tap(s) / {showers} shower(s)"]
    if room.get("status"):
        parts.append(f"Status: {room['status']}")
    return " | ".join(parts)


def _floor_to_jsonable(floor_key: str) -> Dict:
    floor = st.session_state["floors"][floor_key]
    return {
        "floor": floor_key,
        "image_bytes_b64": base64.b64encode(floor["image_bytes"]).decode("utf-8"),
        "rooms": list(floor["rooms"].values()),
    }


def _crop_room_preview(image: Image.Image, room: Dict) -> Optional[Image.Image]:
    if not room.get("polygon"):
        return None
    width, height = image.size
    xs = [pt[0] for pt in room["polygon"]]
    ys = [pt[1] for pt in room["polygon"]]
    min_x = max(min(xs) - 0.02, 0)
    max_x = min(max(xs) + 0.02, 1)
    min_y = max(min(ys) - 0.02, 0)
    max_y = min(max(ys) + 0.02, 1)
    left, top = _denormalize_point(min_x, min_y, width, height)
    right, bottom = _denormalize_point(max_x, max_y, width, height)
    left = int(max(0, left))
    top = int(max(0, top))
    right = int(min(width, right))
    bottom = int(min(height, bottom))
    if right - left <= 0 or bottom - top <= 0:
        return None
    return image.crop((left, top, right, bottom))


# ---------- Streamlit layout ----------

def main():
    st.set_page_config(page_title="Building Schematic Mapper", layout="wide")

    st.title("Building Schematic Mapper")
    st.caption(
        "Upload floor schematics, outline rooms, add fixtures, and keep directions handy for wayfinding."
    )

    with st.sidebar:
        st.header("Floors & Rooms")
        uploaded_files = st.file_uploader(
            "Upload schematics (PNG/JPG) or layout JSON",
            type=["png", "jpg", "jpeg", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for file in uploaded_files:
                if not _register_uploaded_file(file):
                    continue

                file_name = file.name
                if file_name.lower().endswith(".json"):
                    payload = file.read()
                    _ingest_layout_json(payload, file_name)
                else:
                    image = _load_image(file)
                    _ensure_floor(file.name, image)

        floors = st.session_state["floors"]
        if floors:
            floor_names = list(floors.keys())
            active_floor = st.selectbox(
                "Select floor",
                floor_names,
                index=floor_names.index(st.session_state.get("active_floor", floor_names[0])),
            )
            st.session_state["active_floor"] = active_floor
        else:
            st.info("Upload at least one floor image to begin.")
            st.stop()

        search = st.text_input("Filter rooms", "")

        active_floor = st.session_state["active_floor"]
        floor = st.session_state["floors"][active_floor]
        rooms = list(floor["rooms"].values())
        if search:
            rooms = [room for room in rooms if search.lower() in room["name"].lower()]

        room_labels = [f"{room['name']}" for room in rooms]
        if rooms:
            selected_label = st.radio("Rooms", room_labels, key="room_radio")
            selected_room_id = None
            for room in rooms:
                if room["name"] == selected_label:
                    selected_room_id = room["id"]
                    break
            st.session_state["active_room_id"] = selected_room_id
        else:
            st.write("No rooms yet.")

        export_payload = json.dumps(
            [_floor_to_jsonable(name) for name in st.session_state["floors"].keys()],
            indent=2,
        ).encode("utf-8")
        st.download_button("Download layout JSON", export_payload, file_name="schematic.json")

    active_floor = st.session_state.get("active_floor")
    if not active_floor:
        st.warning("Select a floor to begin editing.")
        st.stop()

    if active_floor not in st.session_state["floors"]:
        st.warning("The selected floor is no longer available. Please choose another upload.")
        st.stop()

    floor = st.session_state["floors"][active_floor]
    floor_image = _get_floor_image(active_floor)
    if floor_image.mode != "RGBA":
        floor_image = floor_image.convert("RGBA")
    width, height = floor_image.size

    col_canvas, col_info = st.columns([3, 2])

    with col_canvas:
        st.subheader("Interactive map")
        drawing_modes = ["Room", "Tap", "Shower", "Move/Select"]
        default_mode = st.session_state.get("drawing_entity", "Room")
        if default_mode not in drawing_modes:
            default_mode = "Room"
        drawing_entity = st.radio(
            "Drawing mode",
            drawing_modes,
            index=drawing_modes.index(default_mode),
            horizontal=True,
        )
        st.session_state["drawing_entity"] = drawing_entity

        if drawing_entity == "Room":
            drawing_mode = "polygon"
        elif drawing_entity == "Move/Select":
            drawing_mode = "transform"
        else:
            drawing_mode = "point"

        canvas_objects = _build_canvas_objects(active_floor)
        canvas_json = _serialize_canvas(canvas_objects)

        canvas_height = min(900, height)
        canvas_width = min(1200, width)

        canvas_result = st_canvas(
            background_image=floor_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            initial_drawing=canvas_json,
            display_toolbar=True,
            key=f"schematic_canvas_{active_floor}",
        )

        if canvas_result.json_data:
            objects = canvas_result.json_data.get("objects", [])
            _sync_existing_entities(active_floor, objects)
            new_objects = [obj for obj in objects if not obj.get("data")]
            if new_objects:
                _capture_new_objects(active_floor, new_objects)
            detected_room = _detect_active_room(objects)
            if detected_room:
                st.session_state["active_room_id"] = detected_room

        if st.session_state.get("needs_canvas_refresh"):
            st.session_state["needs_canvas_refresh"] = False
            st.experimental_rerun()

        st.write(
            "Use the toolbar to select, move, or delete shapes. Switch drawing modes to add new rooms or fixtures."
        )

    with col_info:
        st.subheader("Room details")
        active_room_id = st.session_state.get("active_room_id")
        active_room = None
        if active_room_id and active_room_id in floor["rooms"]:
            active_room = floor["rooms"][active_room_id]

        if active_room:
            with st.form("room_form"):
                name = st.text_input("Name", active_room["name"])
                status = st.text_input("Status", active_room.get("status", ""))
                color = st.color_picker("Fill colour", active_room.get("color", "#1f77b455"))
                notes = st.text_area("Notes", active_room.get("notes", ""), height=120)
                submitted = st.form_submit_button("Save room")
            if submitted:
                active_room["name"] = name
                active_room["status"] = status
                active_room["color"] = color
                active_room["notes"] = notes
                st.experimental_rerun()

            st.markdown("**Summary**")
            st.write(_room_summary(active_room))

            preview = _crop_room_preview(floor_image, active_room)
            if preview:
                st.image(preview, caption="Focused room view", use_column_width=True)

            if active_room.get("fixtures"):
                st.markdown("**Fixtures**")
                for fixture in active_room["fixtures"]:
                    st.write(f"• {fixture['type'].title()} at {fixture['point']}")
        else:
            st.info("Select a room to edit its details.")

    st.subheader("Data overview")
    for room in floor["rooms"].values():
        st.write(f"**{room['name']}** — {_room_summary(room)}")
        if room.get("notes"):
            st.write(room["notes"])

    st.markdown("---")
    st.caption(
        "All edits remain in your browser session. Use the download button to export your layout as JSON for future reuse."
    )


if __name__ == "__main__":
    main()
