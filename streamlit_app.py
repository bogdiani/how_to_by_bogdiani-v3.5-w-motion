
import json
from pathlib import Path
import streamlit as st
from pipeline import render_from_json

st.set_page_config(page_title="JSON → Vertical Video", page_icon="🎬", layout="centered")

st.title("🎬 JSON → Vertical Video")
st.write("Upload your JSON script and render a 9:16 video. Optional: set OpenAI & Pexels keys in `.env`.")

uploaded = st.file_uploader("Upload script.json", type=["json"])
fps = st.slider("FPS", 24, 60, 30)
preset = st.selectbox("Preset", ["reels","tiktok","shorts"], index=0)
tts_mode = st.selectbox("TTS Engine", ["auto", "openai", "offline"], index=0)
use_local = st.checkbox("Use local media from assets/media (if present)", value=False)

out = Path("outputs/streamlit_video.mp4")

if st.button("Render"):
    if not uploaded:
        st.error("Please upload a JSON file first.")
    else:
        data = json.load(uploaded)
        with st.spinner("Rendering..."):
            render_from_json(data, output_path=str(out), fps=fps, preset=preset, tts_mode=tts_mode, use_local_media=use_local)
        st.success("Done!")
        st.video(str(out))
        st.download_button("Download MP4", out.read_bytes(), file_name="video.mp4", mime="video/mp4")
