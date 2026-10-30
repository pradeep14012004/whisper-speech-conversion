import streamlit as st
from pathlib import Path

st.title("File Upload Test")

audio = st.file_uploader("Upload audio (WAV, MP3, FLAC, OGG, M4A)", type=None)
video = st.file_uploader("Upload video (MP4, AVI, MOV, MKV)", type=None)

if audio:
    st.success(f"Audio received: {audio.name} ({len(audio.getvalue())} bytes)")
if video:
    st.success(f"Video received: {video.name} ({len(video.getvalue())} bytes)")
