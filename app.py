import base64
import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="Floorplan Room Manager", layout="wide")

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "floorplans.db"


@dataclass
class ParsedRoom:
    """Container for room data extracted from uploaded JSON."""

    room_id: str
    source_name: str
    geometry: Optional[Any]
    attributes: Dict[str, Any]


def init_state() -> None:
    st.session_state.setdefault("rooms_df", None)
    st.session_state.setdefault("room_lookup", {})
    st.session_state.setdefault("floorplan_name", "")
    st.session_state.setdefault("uploaded_filename", "")
    st.session_state.setdefault("parse_message", "")
    st.session_state.setdefault("floorplan_image_bytes", None)
    st.session_state.setdefault("floorplan_image_meta", "")
    st.session_state.setdefault("floorplan_image_path", "")
    st.session_state.setdefault("dept_room_select", [])
    st.session_state.setdefault("clear_dept_room_select", False)


init_state()


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS floorplans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source_filename TEXT,
            uploaded_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            floorplan_id INTEGER NOT NULL REFERENCES floorplans(id) ON DELETE CASCADE,
            room_identifier TEXT NOT NULL,
            source_name TEXT,
            display_name TEXT,
            department TEXT,
            geometry TEXT,
            attributes TEXT
        )
        """
    )
    return conn


def find_room_list(obj: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Search the JSON payload for the most likely room list."""

    if isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
        return obj, "root"

    queue: List[Tuple[Any, str]] = []
    if isinstance(obj, dict):
        queue.extend((value, key) for key, value in obj.items())
    elif isinstance(obj, list):
        queue.extend((value, f"root[{idx}]") for idx, value in enumerate(obj))

    best_match: Tuple[List[Dict[str, Any]], str, int] = ([], "", -1)

    while queue:
        node, path = queue.pop(0)
        if isinstance(node, list):
            if node and all(isinstance(item, dict) for item in node):
                score = score_room_list(node)
                if score > best_match[2]:
                    best_match = (node, path, score)
            for idx, value in enumerate(node):
                queue.append((value, f"{path}[{idx}]"))
        elif isinstance(node, dict):
            for key, value in node.items():
                queue.append((value, f"{path}.{key}" if path else key))

    return best_match[0], best_match[1]


def score_room_list(items: List[Dict[str, Any]]) -> int:
    """Heuristic to determine how likely a list of dicts is a room collection."""

    room_keywords = {"room", "space", "area", "unit"}
    score = 0
    for item in items:
        keys = " ".join(item.keys()).lower()
        for keyword in room_keywords:
            if keyword in keys:
                score += 1
        if any(k in item for k in ("id", "room_id", "name", "number")):
            score += 1
    return score


def decode_base64_image(data: str) -> Optional[Tuple[bytes, str, Tuple[int, int]]]:
    """Attempt to decode a base64 string into image bytes, returning metadata if valid."""

    text = data.strip()
    if not text:
        return None

    mime: Optional[str] = None
    if text.startswith("data:image"):
        try:
            header, payload = text.split(",", 1)
        except ValueError:
            return None
        mime = header.split(";", 1)[0].split(":", 1)[-1]
        text = payload

    try:
        raw = base64.b64decode(text, validate=True)
    except Exception:
        try:
            raw = base64.b64decode(text)
        except Exception:
            return None

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            size = img.size
            fmt = img.format or (mime.split("/", 1)[-1].upper() if mime else "Image")
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    return raw, fmt, size


def find_image_data(obj: Any, path: str = "") -> Optional[Tuple[bytes, str, Tuple[int, int], str]]:
    """Traverse arbitrary JSON payload to locate the first decodable image."""

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, str):
                decoded = decode_base64_image(value)
                if decoded:
                    raw, fmt, size = decoded
                    return raw, fmt, size, new_path
            else:
                result = find_image_data(value, new_path)
                if result:
                    return result
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]"
            result = find_image_data(item, new_path)
            if result:
                return result
    return None


def extract_image_from_payload(payload: Any) -> Tuple[Optional[bytes], Optional[str], str]:
    """Return raw image bytes, a metadata caption, and the JSON path for any embedded image."""

    found = find_image_data(payload)
    if not found:
        return None, None, ""
    raw, fmt, size, path = found
    caption = f"{fmt} {size[0]}×{size[1]}px"
    return raw, caption, path


def coalesce(values: Iterable[Any], default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, str):
            return str(value)
    return default


def parse_rooms(payload: Any) -> Tuple[List[ParsedRoom], str]:
    rooms: List[ParsedRoom] = []
    room_list, path = find_room_list(payload)
    if not room_list:
        return rooms, "Could not find a list of rooms in the uploaded JSON."

    used_ids: Dict[str, int] = {}
    for index, raw in enumerate(room_list, start=1):
        if not isinstance(raw, dict):
            continue
        room_identifier = coalesce(
            [
                raw.get("id"),
                raw.get("room_id"),
                raw.get("roomId"),
                raw.get("guid"),
                raw.get("number"),
                raw.get("name"),
            ],
            default=f"room-{index}",
        )
        base_identifier = room_identifier
        counter = used_ids.get(base_identifier, 0)
        if counter:
            room_identifier = f"{base_identifier}-{counter+1}"
        used_ids[base_identifier] = counter + 1

        source_name = coalesce(
            [
                raw.get("name"),
                raw.get("label"),
                raw.get("displayName"),
                raw.get("number"),
            ]
        )

        geometry = None
        for key in ("geometry", "polygon", "outline", "shape"):
            if key in raw:
                geometry = raw[key]
                break
        if geometry is None:
            for key in ("coordinates", "points", "vertices"):
                if key in raw and isinstance(raw[key], (list, dict)):
                    geometry = {key: raw[key]}
                    break

        rooms.append(
            ParsedRoom(
                room_id=str(room_identifier),
                source_name=str(source_name) if source_name else "",
                geometry=geometry,
                attributes=raw,
            )
        )

    message = "Parsed rooms from path: " + (path or "root")
    return rooms, message


def build_rooms_dataframe(rooms: List[ParsedRoom]) -> pd.DataFrame:
    data = []
    for room in rooms:
        data.append(
            {
                "room_id": room.room_id,
                "source_name": room.source_name,
                "display_name": room.source_name or room.room_id,
                "department": "",
            }
        )
    return pd.DataFrame(data)


def save_floorplan_to_db(name: str, filename: str, df: pd.DataFrame, lookup: Dict[str, ParsedRoom]) -> int:
    conn = get_connection()
    try:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        cursor = conn.execute(
            "INSERT INTO floorplans (name, source_filename, uploaded_at) VALUES (?, ?, ?)",
            (name, filename, timestamp),
        )
        floorplan_id = cursor.lastrowid
        records = df.to_dict(orient="records")
        for record in records:
            room_id = record.get("room_id")
            parsed_room = lookup.get(room_id)
            geometry_json = json.dumps(parsed_room.geometry) if parsed_room and parsed_room.geometry is not None else None
            attributes_json = json.dumps(parsed_room.attributes) if parsed_room else None
            source_name = record.get("source_name")
            display_name = record.get("display_name")
            department = record.get("department")
            if pd.isna(source_name):
                source_name = None
            if pd.isna(display_name):
                display_name = None
            if pd.isna(department):
                department = None
            conn.execute(
                """
                INSERT INTO rooms (floorplan_id, room_identifier, source_name, display_name, department, geometry, attributes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    floorplan_id,
                    room_id,
                    source_name,
                    display_name,
                    department,
                    geometry_json,
                    attributes_json,
                ),
            )
        conn.commit()
        return floorplan_id
    finally:
        conn.close()


def load_saved_floorplans() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["id", "name", "source_filename", "uploaded_at", "room_count"])
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT f.id, f.name, f.source_filename, f.uploaded_at, COUNT(r.id) AS room_count
            FROM floorplans f
            LEFT JOIN rooms r ON r.floorplan_id = f.id
            GROUP BY f.id
            ORDER BY f.uploaded_at DESC
            """,
            conn,
        )
        return df
    finally:
        conn.close()


def load_rooms_for_floorplan(floorplan_id: int) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT room_identifier AS room_id, source_name, display_name, department
            FROM rooms
            WHERE floorplan_id = ?
            ORDER BY room_identifier
            """,
            conn,
            params=(floorplan_id,),
        )
        return df
    finally:
        conn.close()


st.title("Floorplan Room Manager")
st.caption(
    "Upload a floorplan JSON file, name rooms, assign departments, and store the results in a SQLite database."
)

uploaded = st.file_uploader("Upload floorplan JSON", type=["json"], accept_multiple_files=False)
if uploaded is not None:
    try:
        payload = json.loads(uploaded.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        st.error(f"Could not decode JSON: {exc}")
        payload = None
    if payload is not None:
        rooms, message = parse_rooms(payload)
        image_bytes, image_meta, image_path = extract_image_from_payload(payload)
        st.session_state.floorplan_image_bytes = image_bytes
        st.session_state.floorplan_image_meta = image_meta or ""
        st.session_state.floorplan_image_path = image_path
        if not rooms:
            st.error("No rooms could be parsed from the uploaded file.")
            st.session_state.parse_message = message
            st.session_state.rooms_df = None
            st.session_state.room_lookup = {}
            if image_bytes is None:
                st.session_state.floorplan_image_bytes = None
                st.session_state.floorplan_image_meta = ""
                st.session_state.floorplan_image_path = ""
        else:
            st.session_state.rooms_df = build_rooms_dataframe(rooms)
            st.session_state.room_lookup = {room.room_id: room for room in rooms}
            default_name = Path(uploaded.name).stem
            st.session_state.floorplan_name = default_name
            st.session_state.uploaded_filename = uploaded.name
            st.session_state.parse_message = message

if st.session_state.parse_message:
    st.info(st.session_state.parse_message)

if st.session_state.floorplan_image_bytes is not None:
    st.subheader("Floorplan image")
    caption_parts: List[str] = []
    if st.session_state.floorplan_image_meta:
        caption_parts.append(st.session_state.floorplan_image_meta)
    if st.session_state.floorplan_image_path:
        caption_parts.append(f"JSON path: {st.session_state.floorplan_image_path}")
    caption = " • ".join(caption_parts) if caption_parts else None
    st.image(st.session_state.floorplan_image_bytes, caption=caption, use_column_width=True)

if st.session_state.rooms_df is None or st.session_state.rooms_df.empty:
    st.stop()

st.subheader("Room details")

col_rooms, col_departments = st.columns([2.5, 1.5])

with col_rooms:
    st.session_state.floorplan_name = st.text_input(
        "Floorplan name",
        value=st.session_state.floorplan_name,
        help="Name of the floorplan to store in the database.",
    )

    edited_df = st.data_editor(
        st.session_state.rooms_df,
        column_config={
            "room_id": st.column_config.TextColumn(
                "Room ID",
                help="Unique identifier parsed from the uploaded JSON.",
                disabled=True,
            ),
            "source_name": st.column_config.TextColumn(
                "Source name",
                help="Name from the uploaded file.",
            ),
            "display_name": st.column_config.TextColumn(
                "Display name",
                help="Name that will be saved for the room.",
            ),
            "department": st.column_config.TextColumn(
                "Department",
                help="Department assignment for the room.",
            ),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
    )
    if "room_id" not in edited_df.columns:
        edited_df = edited_df.reset_index().rename(columns={"index": "room_id"})
    st.session_state.rooms_df = edited_df

with col_departments:
    st.markdown("### Department assignment")
    with st.form("department_form"):
        dept_name = st.text_input("Department name", key="dept_name_input")
        options: List[str] = []
        option_map: Dict[str, str] = {}
        for _, row in st.session_state.rooms_df.iterrows():
            room_id = str(row["room_id"])
            display_name = row["display_name"] if pd.notna(row["display_name"]) else ""
            label = f"{room_id} — {display_name}" if display_name else room_id
            options.append(label)
            option_map[label] = room_id
        if st.session_state.pop("clear_dept_room_select", False):
            st.session_state["dept_room_select"] = []
        selections = st.multiselect(
            "Select rooms",
            options=options,
            key="dept_room_select",
        )
        submitted = st.form_submit_button("Assign selected rooms")
        if submitted:
            if not dept_name.strip():
                st.warning("Enter a department name before assigning rooms.")
            elif not selections:
                st.warning("Select at least one room to assign.")
            else:
                room_ids = [option_map[sel] for sel in selections]
                updated = st.session_state.rooms_df.copy()
                mask = updated["room_id"].astype(str).isin(room_ids)
                updated.loc[mask, "department"] = dept_name.strip()
                st.session_state.rooms_df = updated
                st.session_state.clear_dept_room_select = True
                st.success(f"Assigned {len(room_ids)} room(s) to '{dept_name.strip()}'.")

    sanitized = st.session_state.rooms_df.copy()
    sanitized["department"] = (
        sanitized["department"].fillna("").astype(str).str.strip()
    )
    assigned_mask = sanitized["department"] != ""
    st.markdown("#### Current departments")
    if not assigned_mask.any():
        st.caption("No departments assigned yet.")
    else:
        for dept, group in sanitized[assigned_mask].groupby("department"):
            st.write(f"**{dept}** ({len(group)} rooms)")
            st.caption(", ".join(group["display_name"].fillna("").tolist()))

save_disabled = not st.session_state.floorplan_name.strip()

if st.button("Save to database", type="primary", disabled=save_disabled):
    floorplan_id = save_floorplan_to_db(
        st.session_state.floorplan_name.strip(),
        st.session_state.uploaded_filename,
        st.session_state.rooms_df,
        st.session_state.room_lookup,
    )
    st.success(f"Floorplan saved with id {floorplan_id}.")

st.markdown("---")

st.subheader("Saved floorplans")
saved = load_saved_floorplans()
if saved.empty:
    st.caption("No floorplans have been saved yet.")
else:
    st.dataframe(saved, use_container_width=True)
    selected_id = st.selectbox(
        "View rooms for saved floorplan",
        options=["None"] + saved["id"].astype(str).tolist(),
        index=0,
    )
    if selected_id != "None":
        rooms_df = load_rooms_for_floorplan(int(selected_id))
        st.dataframe(rooms_df, use_container_width=True)
