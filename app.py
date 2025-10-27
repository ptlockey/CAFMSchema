
import json, base64, io
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide", page_title="JSON Schematic Viewer (Minimal)")
st.title("JSON Schematic Viewer (Minimal)")

j = st.file_uploader("Upload JSON with base64 image", type=["json"])
if not j:
    st.stop()

try:
    data = json.loads(j.read().decode("utf-8"))
except Exception as e:
    st.error(f"Invalid JSON: {e}")
    st.stop()

node = data.get("pages",{}).get("0",{})
b64 = node.get("image_b64") or node.get("image_base64") or node.get("image")
if not isinstance(b64, str):
    st.error("No 'image_b64' or 'image_base64' key found under pages['0'].")
    st.stop()

if b64.startswith("data:image"):
    b64 = b64.split(",",1)[1]

try:
    import base64, io
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
except Exception as e:
    st.error(f"Image decode error: {e}")
    st.stop()

st.image(img, caption=f"{img.size[0]} x {img.size[1]}")
