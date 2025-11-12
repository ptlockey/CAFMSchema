import base64
import io
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_plotly_events import plotly_events

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


@dataclass
class WallSegment:
    """Metadata describing a single wall edge for selection/deletion."""

    segment_id: str
    room_id: str
    shape: str  # "ring" or "path"
    component_index: int
    start_vertex: int
    end_vertex: int
    start: Tuple[float, float]
    end: Tuple[float, float]


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
    st.session_state.setdefault("geometry_transform", None)
    st.session_state.setdefault("selected_wall_segment", None)
    st.session_state.setdefault("geometry_editor_message", "")


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


def is_geojson_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("type") in {"FeatureCollection", "Feature"}


def normalize_polygon(coords: Any) -> Optional[List[List[float]]]:
    """Return a list of [x, y] pairs for the outer ring of a polygon."""

    if not isinstance(coords, list) or not coords:
        return None
    first = coords[0]
    if not isinstance(first, list):
        return None
    if first and isinstance(first[0], (int, float)):
        points = []
        for pair in coords:
            if (
                isinstance(pair, list)
                and len(pair) >= 2
                and isinstance(pair[0], (int, float))
                and isinstance(pair[1], (int, float))
            ):
                points.append([float(pair[0]), float(pair[1])])
        return points or None
    if first and isinstance(first[0], list):
        # Standard GeoJSON polygon -> coords[0] is exterior ring
        return normalize_polygon(first)
    return None


def normalize_polyline(coords: Any) -> Optional[List[List[float]]]:
    """Return a list of [x, y] pairs for a polyline or line string."""

    if not isinstance(coords, list) or not coords:
        return None
    points: List[List[float]] = []
    for pair in coords:
        if (
            isinstance(pair, (list, tuple))
            and len(pair) >= 2
            and isinstance(pair[0], (int, float))
            and isinstance(pair[1], (int, float))
        ):
            points.append([float(pair[0]), float(pair[1])])
    return points or None


def geometry_to_rings(geometry: Any) -> List[List[List[float]]]:
    """Return a list of polygon rings for the provided geometry."""

    if geometry is None:
        return []
    if isinstance(geometry, dict):
        g_type = geometry.get("type")
        coords = geometry.get("coordinates")
        if g_type == "Polygon":
            ring = normalize_polygon(coords)
            return [ring] if ring else []
        if g_type == "MultiPolygon" and isinstance(coords, list):
            rings: List[List[List[float]]] = []
            for poly in coords:
                ring = normalize_polygon(poly)
                if ring:
                    rings.append(ring)
            return rings
        # Some payloads may wrap geometry
        if isinstance(coords, dict):
            return geometry_to_rings(coords)
    if isinstance(geometry, list):
        ring = normalize_polygon(geometry)
        return [ring] if ring else []
    return []


def geometry_to_paths(geometry: Any) -> List[List[List[float]]]:
    """Return polyline paths for LineString-like geometries."""

    if geometry is None:
        return []
    if isinstance(geometry, dict):
        g_type = geometry.get("type")
        coords = geometry.get("coordinates")
        if g_type == "LineString":
            path = normalize_polyline(coords)
            return [path] if path else []
        if g_type == "MultiLineString" and isinstance(coords, list):
            paths: List[List[List[float]]] = []
            for line in coords:
                path = normalize_polyline(line)
                if path:
                    paths.append(path)
            return paths
        if isinstance(coords, dict):
            return geometry_to_paths(coords)
    if isinstance(geometry, (list, tuple)):
        path = normalize_polyline(list(geometry))
        return [path] if path else []
    return []


def build_wall_preview_figure(
    lookup: Dict[str, "ParsedRoom"],
    transform: Dict[str, float],
    wall_width: float,
) -> Optional[go.Figure]:
    """Return a Plotly figure showing polygon edges and polylines as walls."""

    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []

    for room in lookup.values():
        for ring in geometry_to_rings(room.geometry):
            if len(ring) < 2:
                continue
            pairs = list(zip(ring, ring[1:]))
            if ring[0] != ring[-1]:
                pairs.append((ring[-1], ring[0]))
            for (x1, y1), (x2, y2) in pairs:
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue
                xs.extend([float(x1), float(x2), None])
                ys.extend([float(y1), float(y2), None])
        for path in geometry_to_paths(room.geometry):
            if len(path) < 2:
                continue
            for (x1, y1), (x2, y2) in zip(path, path[1:]):
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue
                xs.extend([float(x1), float(x2), None])
                ys.extend([float(y1), float(y2), None])

    if not xs or not ys:
        return None

    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#000000", width=wall_width),
                hoverinfo="skip",
            )
        ]
    )

    min_x = transform["min_x"]
    max_x = transform["max_x"]
    min_y = transform["min_y"]
    max_y = transform["max_y"]

    fig.update_xaxes(
        range=[min_x, max_x],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title_text="",
        constrain="domain",
    )
    fig.update_yaxes(
        range=[max_y, min_y],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title_text="",
        scaleanchor="x",
        scaleratio=1,
    )

    height = int(max(transform.get("canvas_height", 0), 300))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=20, b=20),
        height=height,
    )

    return fig


def _collect_ring_points(ring: List[List[float]]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for pair in ring:
        if (
            isinstance(pair, list)
            and len(pair) >= 2
            and isinstance(pair[0], (int, float))
            and isinstance(pair[1], (int, float))
            and math.isfinite(pair[0])
            and math.isfinite(pair[1])
        ):
            points.append((float(pair[0]), float(pair[1])))
    if len(points) < 2:
        return []
    if points[0] == points[-1]:
        points = points[:-1]
    return points


def collect_wall_segments(lookup: Dict[str, "ParsedRoom"]) -> Dict[str, WallSegment]:
    segments: Dict[str, WallSegment] = {}
    for room in lookup.values():
        rings = geometry_to_rings(room.geometry)
        for ring_index, ring in enumerate(rings):
            points = _collect_ring_points(ring)
            if len(points) < 2:
                continue
            total = len(points)
            for start_idx in range(total):
                end_idx = (start_idx + 1) % total
                start = points[start_idx]
                end = points[end_idx]
                segment_id = f"{room.room_id}|ring|{ring_index}|{start_idx}->{end_idx}"
                segments[segment_id] = WallSegment(
                    segment_id=segment_id,
                    room_id=room.room_id,
                    shape="ring",
                    component_index=ring_index,
                    start_vertex=start_idx,
                    end_vertex=end_idx,
                    start=start,
                    end=end,
                )
        paths = geometry_to_paths(room.geometry)
        for path_index, path in enumerate(paths):
            points = [
                (float(pair[0]), float(pair[1]))
                for pair in path
                if isinstance(pair, list)
                and len(pair) >= 2
                and isinstance(pair[0], (int, float))
                and isinstance(pair[1], (int, float))
                and math.isfinite(pair[0])
                and math.isfinite(pair[1])
            ]
            if len(points) < 2:
                continue
            for start_idx in range(len(points) - 1):
                end_idx = start_idx + 1
                start = points[start_idx]
                end = points[end_idx]
                segment_id = f"{room.room_id}|path|{path_index}|{start_idx}->{end_idx}"
                segments[segment_id] = WallSegment(
                    segment_id=segment_id,
                    room_id=room.room_id,
                    shape="path",
                    component_index=path_index,
                    start_vertex=start_idx,
                    end_vertex=end_idx,
                    start=start,
                    end=end,
                )
    return segments


def build_wall_editor_figure(
    segments: Dict[str, WallSegment],
    transform: Dict[str, float],
    selected_segment_id: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    for seg in segments.values():
        xs = [seg.start[0], seg.end[0]]
        ys = [seg.start[1], seg.end[1]]
        is_selected = seg.segment_id == selected_segment_id
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#d62728" if is_selected else "#000000", width=4 if is_selected else 1.5),
                hovertemplate=(
                    f"Room {seg.room_id}<br>"
                    f"{'Polygon edge' if seg.shape == 'ring' else 'Line segment'}"
                    "<br>x₁=%{x[0]:.2f}, y₁=%{y[0]:.2f}<br>x₂=%{x[1]:.2f}, y₂=%{y[1]:.2f}<extra></extra>"
                ),
                customdata=[[seg.segment_id], [seg.segment_id]],
            )
        )

    min_x = transform["min_x"]
    max_x = transform["max_x"]
    min_y = transform["min_y"]
    max_y = transform["max_y"]

    fig.update_xaxes(
        range=[min_x, max_x],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title_text="",
        constrain="domain",
    )
    fig.update_yaxes(
        range=[max_y, min_y],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title_text="",
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        clickmode="event+select",
    )
    return fig


def delete_wall_segment(segment: WallSegment, lookup: Dict[str, "ParsedRoom"]) -> bool:
    room = lookup.get(segment.room_id)
    if room is None:
        return False
    geometry = room.geometry
    if geometry is None:
        return False

    if segment.shape == "ring":
        rings = geometry_to_rings(geometry)
        if not rings or segment.component_index >= len(rings):
            return False
        ring = _collect_ring_points(rings[segment.component_index])
        if len(ring) < 2:
            return False
        total = len(ring)
        remove_idx = segment.end_vertex % total
        del ring[remove_idx]
        if len(ring) < 3:
            del rings[segment.component_index]
        else:
            ring.append(ring[0])
            rings[segment.component_index] = ring
        if not rings:
            room.geometry = None
        elif len(rings) == 1:
            room.geometry = {"type": "Polygon", "coordinates": [rings[0]]}
        else:
            room.geometry = {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}
        return True

    paths = geometry_to_paths(geometry)
    if not paths or segment.component_index >= len(paths):
        return False
    path = [
        [float(pair[0]), float(pair[1])]
        for pair in paths[segment.component_index]
        if isinstance(pair, list)
        and len(pair) >= 2
        and isinstance(pair[0], (int, float))
        and isinstance(pair[1], (int, float))
    ]
    if len(path) < 2:
        return False
    if segment.end_vertex < len(path):
        del path[segment.end_vertex]
    if len(path) < 2:
        del paths[segment.component_index]
    else:
        paths[segment.component_index] = path

    if not paths:
        room.geometry = None
    elif len(paths) == 1:
        room.geometry = {"type": "LineString", "coordinates": paths[0]}
    else:
        room.geometry = {"type": "MultiLineString", "coordinates": paths}
    return True


def iter_geometry_points(lookup: Dict[str, "ParsedRoom"]) -> Iterable[Tuple[float, float]]:
    for room in lookup.values():
        rings = geometry_to_rings(room.geometry)
        paths = geometry_to_paths(room.geometry)
        for ring in rings:
            for x, y in ring:
                yield x, y
        for path in paths:
            for x, y in path:
                yield x, y


def compute_geometry_bounds(lookup: Dict[str, "ParsedRoom"]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for x, y in iter_geometry_points(lookup):
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def build_geometry_transform(lookup: Dict[str, "ParsedRoom"], canvas_size: int = 900) -> Optional[Dict[str, float]]:
    bounds = compute_geometry_bounds(lookup)
    if not bounds:
        return None
    min_x, min_y, max_x, max_y = bounds
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    margin = canvas_size * 0.05
    usable_w = canvas_size - 2 * margin
    usable_h = canvas_size - 2 * margin
    scale = min(usable_w / width if width else 1.0, usable_h / height if height else 1.0)
    offset_x = margin
    offset_y = margin
    canvas_width = math.ceil(width * scale + 2 * margin)
    canvas_height = math.ceil(height * scale + 2 * margin)
    return {
        "min_x": float(min_x),
        "max_x": float(max_x),
        "min_y": float(min_y),
        "max_y": float(max_y),
        "scale": float(scale),
        "offset_x": float(offset_x),
        "offset_y": float(offset_y),
        "canvas_width": float(canvas_width),
        "canvas_height": float(canvas_height),
    }


def parse_rooms(payload: Any) -> Tuple[List[ParsedRoom], str]:
    if is_geojson_payload(payload):
        return parse_geojson_rooms(payload)
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


def parse_geojson_rooms(payload: Dict[str, Any]) -> Tuple[List[ParsedRoom], str]:
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        features = []
    rooms: List[ParsedRoom] = []
    message = ""
    used_ids: Dict[str, int] = {}
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        g_type = geometry.get("type")
        if g_type not in {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}:
            continue
        coords = geometry.get("coordinates")
        rings: List[List[List[float]]] = []
        paths: List[List[List[float]]] = []
        if g_type in {"Polygon", "MultiPolygon"}:
            rings = geometry_to_rings({"type": g_type, "coordinates": coords})
            if not rings:
                continue
        else:
            paths = geometry_to_paths({"type": g_type, "coordinates": coords})
            if not paths:
                continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        room_identifier = coalesce(
            [
                properties.get("room_id"),
                properties.get("roomId"),
                properties.get("id"),
                properties.get("guid"),
                properties.get("name"),
                feature.get("id"),
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
                properties.get("name"),
                properties.get("label"),
                properties.get("displayName"),
            ]
        )
        normalized_geometry: Dict[str, Any]
        if g_type in {"Polygon", "MultiPolygon"}:
            if len(rings) == 1:
                normalized_geometry = {"type": "Polygon", "coordinates": [rings[0]]}
            else:
                normalized_geometry = {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}
        else:
            if len(paths) == 1:
                normalized_geometry = {"type": "LineString", "coordinates": paths[0]}
            else:
                normalized_geometry = {"type": "MultiLineString", "coordinates": paths}
        attributes = {
            "properties": properties,
            "feature_id": feature.get("id"),
        }
        rooms.append(
            ParsedRoom(
                room_id=str(room_identifier),
                source_name=str(source_name) if source_name else "",
                geometry=normalized_geometry,
                attributes=attributes,
            )
        )
    if not rooms:
        message = "GeoJSON did not contain any polygon, multipolygon, or line features."
    else:
        message = "Parsed GeoJSON features"
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


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to UTF-8 encoded CSV bytes suitable for download."""

    return df.to_csv(index=False).encode("utf-8")


def room_lookup_to_geojson(lookup: Dict[str, ParsedRoom]) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for room in lookup.values():
        geometry = room.geometry
        if not isinstance(geometry, dict):
            continue
        properties: Dict[str, Any] = {}
        if isinstance(room.attributes, dict):
            if "properties" in room.attributes and isinstance(room.attributes["properties"], dict):
                properties = dict(room.attributes["properties"])
            else:
                properties = dict(room.attributes)
        properties.setdefault("room_id", room.room_id)
        if room.source_name:
            properties.setdefault("name", room.source_name)
        features.append(
            {
                "type": "Feature",
                "id": room.attributes.get("feature_id") if isinstance(room.attributes, dict) else room.room_id,
                "properties": properties,
                "geometry": geometry,
            }
        )
    return {"type": "FeatureCollection", "features": features}


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
    "Upload a floorplan JSON or GeoJSON file, name rooms, assign departments, and store the results in a SQLite database."
)
st.caption("GeoJSON uploads should contain only geometry data – labels such as room numbers can be added later inside the app.")

uploaded = st.file_uploader(
    "Upload floorplan JSON/GeoJSON", type=["json", "geojson"], accept_multiple_files=False
)
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
            st.session_state.geometry_transform = None
            if image_bytes is None:
                st.session_state.floorplan_image_bytes = None
                st.session_state.floorplan_image_meta = ""
                st.session_state.floorplan_image_path = ""
        else:
            st.session_state.rooms_df = build_rooms_dataframe(rooms)
            st.session_state.room_lookup = {room.room_id: room for room in rooms}
            st.session_state.geometry_transform = build_geometry_transform(st.session_state.room_lookup)
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

transform = build_geometry_transform(st.session_state.room_lookup)
if transform:
    st.session_state.geometry_transform = transform
    st.subheader("Floorplan wall preview")
    wall_thickness = st.slider(
        "Wall thickness",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        key="wall_thickness_slider",
        help="Adjust how thick the wall lines appear in the preview.",
    )
    wall_fig = build_wall_preview_figure(st.session_state.room_lookup, transform, wall_thickness)
    if wall_fig:
        st.plotly_chart(wall_fig, use_container_width=True)
    else:
        st.info("No wall geometries available to preview.")
    st.subheader("Floorplan geometry editor")
    st.caption(
        "This editor mirrors the wall preview with thin black lines. Click a wall segment to "
        "select it, then use the button below to delete it."
    )
    if st.session_state.geometry_editor_message:
        st.success(st.session_state.geometry_editor_message)
        st.session_state.geometry_editor_message = ""

    segments = collect_wall_segments(st.session_state.room_lookup)
    selected_segment_id = st.session_state.selected_wall_segment
    if selected_segment_id and selected_segment_id not in segments:
        selected_segment_id = None
        st.session_state.selected_wall_segment = None

    if not segments:
        st.info("No wall segments available for editing.")
    else:
        editor_fig = build_wall_editor_figure(segments, transform, selected_segment_id)
        events = plotly_events(
            editor_fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=int(transform.get("canvas_height", 600)),
            key="wall_editor_events",
        )
        if events:
            last_event = events[-1]
            payload = last_event.get("customdata")
            if isinstance(payload, list) and payload:
                payload = payload[0]
            if isinstance(payload, str) and payload in segments:
                st.session_state.selected_wall_segment = payload
                selected_segment_id = payload
        # Rebuild the figure to show the active selection highlight
        editor_fig = build_wall_editor_figure(segments, transform, selected_segment_id)
        st.plotly_chart(editor_fig, use_container_width=True)

        selected_segment = (
            segments.get(st.session_state.selected_wall_segment)
            if st.session_state.selected_wall_segment
            else None
        )
        if selected_segment:
            st.caption(
                f"Selected wall in room `{selected_segment.room_id}` "
                f"({selected_segment.shape} #{selected_segment.component_index + 1}, "
                f"vertices {selected_segment.start_vertex + 1}→{selected_segment.end_vertex + 1})."
            )
        else:
            st.caption("Click a segment above to select it for deletion.")

        if st.button("Delete selected wall segment", disabled=selected_segment is None):
            if selected_segment and delete_wall_segment(selected_segment, st.session_state.room_lookup):
                st.session_state.geometry_editor_message = (
                    f"Removed wall segment from room `{selected_segment.room_id}`."
                )
                st.session_state.selected_wall_segment = None
                st.session_state.geometry_transform = build_geometry_transform(
                    st.session_state.room_lookup
                )
                st.experimental_rerun()
            else:
                st.warning("Unable to delete the selected wall segment.")
    geojson_bytes = json.dumps(room_lookup_to_geojson(st.session_state.room_lookup), indent=2).encode("utf-8")
    st.download_button(
        "Download edited floorplan (GeoJSON)",
        geojson_bytes,
        file_name=f"{st.session_state.floorplan_name or 'floorplan'}_edited.geojson",
        mime="application/geo+json",
    )
else:
    st.info(
        "No geometric shapes detected in the upload. Upload a GeoJSON file with polygon coordinates to enable drag-and-drop editing."
    )

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
    export_name = st.session_state.floorplan_name.strip() or "rooms"
    st.download_button(
        "Download current rooms (CSV)",
        dataframe_to_csv_bytes(st.session_state.rooms_df),
        file_name=f"{export_name}_rooms.csv",
        mime="text/csv",
    )

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
                st.success(
                    f"Assigned {len(room_ids)} rooms to {dept_name.strip()}"
                )
                st.rerun()

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
    st.download_button(
        "Download saved floorplans (CSV)",
        dataframe_to_csv_bytes(saved),
        file_name="saved_floorplans.csv",
        mime="text/csv",
    )
    selected_id = st.selectbox(
        "View rooms for saved floorplan",
        options=["None"] + saved["id"].astype(str).tolist(),
        index=0,
    )
    if selected_id != "None":
        rooms_df = load_rooms_for_floorplan(int(selected_id))
        st.dataframe(rooms_df, use_container_width=True)
        st.download_button(
            "Download rooms for selected floorplan (CSV)",
            dataframe_to_csv_bytes(rooms_df),
            file_name=f"floorplan_{selected_id}_rooms.csv",
            mime="text/csv",
        )
