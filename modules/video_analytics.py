import cv2
import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_video_analytics(video_path, audio_path=None):
    if not video_path:
        st.info("Video analytics will appear here after processing a source clip.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error("The supplied video could not be opened for analysis.")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    timestamps = np.linspace(0.0, duration, len(frames)) if len(frames) else np.array([])
    fig_timeline = go.Figure()
    fig_timeline.add_scatter(x=timestamps, y=np.arange(len(frames)), mode="lines+markers", line=dict(color="#7cf0c6", width=2))
    fig_timeline.update_layout(title="Frame extraction timeline", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Frame index")
    st.plotly_chart(fig_timeline, use_container_width=True)

    gallery_cols = st.columns(min(4, max(1, len(frames))))
    for idx, frame in enumerate(frames[:min(8, len(frames))]):
        with gallery_cols[idx % min(4, max(1, len(frames)))]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=f"Frame {idx + 1}", use_container_width=True)

    metrics = {
        "Video resolution": f"{width}x{height}",
        "FPS": fps,
        "Total frames": frame_count,
        "Video duration (s)": round(duration, 3),
    }
    if audio_path:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        audio_duration = len(y) / sr
        metrics["Audio-video duration comparison"] = f"Audio: {audio_duration:.3f}s | Video: {duration:.3f}s"
    st.json(metrics)
