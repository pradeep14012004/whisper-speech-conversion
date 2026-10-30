import asyncio

import json

import math

import os

import re

import tempfile

import threading

import time

from datetime import datetime

from pathlib import Path



import cv2

from opencc import OpenCC

import edge_tts

import librosa

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import psutil

import soundfile as sf

import streamlit as st

from scipy import signal

import torch

import torch.nn as nn

from peft import PeftModel

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    letter = None
    inch = None
    canvas = None

from streamlit_option_menu import option_menu

try:
    from jiwer import cer as jiwer_cer, wer as jiwer_wer
except Exception:  # pragma: no cover
    jiwer_wer = None
    jiwer_cer = None

# Optional dependency for live webcam preview:
# pip install streamlit-webrtc av
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        AudioProcessorBase,
        WebRtcMode,
    )
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False



ROOT = Path(__file__).resolve().parent

DEFAULT_SAMPLE_AUDIO = ROOT / "hf_test_sample.wav"

DEFAULT_SAMPLE_VIDEO = ROOT / "sample_video.mp4"





def load_env_file(env_path: Path) -> None:

    """Load a simple KEY=VALUE .env file into the process environment."""

    if not env_path.exists():

        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():

        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:

            continue

        key, value = line.split("=", 1)

        os.environ.setdefault(key.strip(), value.strip().strip("\\\"\'"))





load_env_file(ROOT / ".env")



from transformers import (

    WhisperProcessor,

    WhisperForConditionalGeneration,

    AutoTokenizer,

    AutoModelForSeq2SeqLM,

)

from transformers.modeling_outputs import BaseModelOutput



# -----------------------------------------------------------------------------

# App configuration

# -----------------------------------------------------------------------------

st.set_page_config(

    page_title="Whisper AV Studio",

    page_icon="🎙️",

    layout="wide",

)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0b1020 0%, #121a2b 100%); color: #edf2ff; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: rgba(20,26,40,0.85);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.25);
    }
    .metric-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { font-size: 1.7rem; font-weight: 700; margin-top: 0.2rem; }
    .panel { background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.15); border-radius: 16px; padding: 1rem 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)



def ensure_session_defaults():
    if "history_log" not in st.session_state:
        st.session_state["history_log"] = []
    if "current_output" not in st.session_state:
        st.session_state["current_output"] = None
    if "last_stage_metrics" not in st.session_state:
        st.session_state["last_stage_metrics"] = {
            "audio_preprocessing": None,
            "video_preprocessing": None,
            "original_av_inference": None,
            "new_lora_inference": None,
            "token_fusion": None,
            "translation": None,
            "speech_generation": None,
            "total_pipeline": None,
        }


ensure_session_defaults()


def ensure_analysis_cache_dir():
    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_CACHE_DIR


def persist_uploaded_media(uploaded_file, suffix_name: str):
    if uploaded_file is None:
        return None
    ensure_analysis_cache_dir()
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", uploaded_file.name or suffix_name)
    safe_path = ANALYSIS_CACHE_DIR / f"{suffix_name}_{filename}"
    safe_path.write_bytes(uploaded_file.getvalue())
    return safe_path


def persist_generated_audio(data: bytes, label: str):
    if not data:
        return None
    ensure_analysis_cache_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_path = ANALYSIS_CACHE_DIR / f"{label}_{stamp}.mp3"
    safe_path.write_bytes(data)
    return safe_path


@st.cache_data(show_spinner=False)
def load_audio_analysis(audio_path, fft_size=2048, pitch_min=50.0, pitch_max=500.0):
    if audio_path is None:
        return None
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(y) / sr
    times = np.linspace(0, duration, len(y))

    y_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=fft_size)), ref=np.max)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_size, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    f0 = librosa.yin(y, fmin=pitch_min, fmax=pitch_max, sr=sr)
    voiced_mask = np.abs(y) > np.max(np.abs(y)) * 0.02
    if len(voiced_mask) != len(f0):
        frame_count = min(len(voiced_mask), len(f0))
        voiced_mask = voiced_mask[:frame_count]
        f0 = f0[:frame_count]
    pitch_values = np.asarray(f0, dtype=float)
    pitch_values[~voiced_mask] = np.nan

    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    magnitude = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    dominant_idx = int(np.nanargmax(magnitude))
    dominant_freq = float(freqs[dominant_idx])
    dominant_db = float(magnitude[dominant_idx])

    return {
        "audio": y,
        "sr": sr,
        "times": times,
        "duration": duration,
        "pitch": pitch_values,
        "pitch_times": np.linspace(0, duration, len(pitch_values)),
        "spectrum_freqs": freqs,
        "spectrum_mag": magnitude,
        "dominant_frequency": dominant_freq,
        "dominant_db": dominant_db,
        "stft_db": y_db,
        "mel_db": mel_db,
        "stft_freqs": librosa.fft_frequencies(sr=sr, n_fft=fft_size),
        "mel_freqs": librosa.mel_frequencies(n_mels=mel_db.shape[0]),
    }


def normalize_eval_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_wer_cer(reference, hypothesis):
    ref = normalize_eval_text(reference)
    hyp = normalize_eval_text(hypothesis)
    if not ref or not hyp:
        return {"WER": "N/A", "CER": "N/A"}
    if jiwer_wer is None or jiwer_cer is None:
        return {"WER": "N/A", "CER": "N/A"}
    try:
        return {
            "WER": float(jiwer_wer(ref, hyp)),
            "CER": float(jiwer_cer(ref, hyp)),
        }
    except Exception:
        return {"WER": "N/A", "CER": "N/A"}


def render_kpi_card(label, value, delta=None, icon=""):
    with st.container():
        delta_html = (
            f"<div style='color:#94a3b8;font-size:0.75rem;margin-top:0.2rem;'>{delta}</div>"
            if delta else ""
        )
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{icon} {label}</div><div class='metric-value'>{value}</div>{delta_html}</div>",
            unsafe_allow_html=True,
        )


def get_recent_history(limit=5):
    return list(reversed(st.session_state.get("history_log", [])))[:limit]


def render_dashboard():
    history = st.session_state.get("history_log", [])
    current = st.session_state.get("current_output")
    recent = get_recent_history(5)

    total_runs = len(history)
    if current is not None:
        total_runs = max(total_runs, 1)

    total_seconds = 0.0
    for item in history:
        total_seconds += float(item.get("processing_time", 0.0) or 0.0)
    avg_runtime = (total_seconds / len(history)) if history else 0.0

    audio_duration = 0.0
    frame_count = 0
    selected_model = "N/A"
    device = DEVICE
    translation_status = "N/A"
    speech_status = "N/A"
    if current is not None:
        audio_duration = float(current.get("audio_duration", 0.0) or 0.0)
        frame_count = int(current.get("frame_count", 0) or 0)
        selected_model = current.get("selected_path", "N/A")
        translation_status = "Ready" if current.get("english") else "Pending"
        speech_status = "Ready" if current.get("speech_bytes") else "Pending"
    elif history:
        last = history[-1]
        audio_duration = float(last.get("audio_duration", 0.0) or 0.0)
        frame_count = int(last.get("frame_count", 0) or 0)
        selected_model = last.get("selected_path", "N/A")
        translation_status = "Ready" if last.get("english") else "Pending"
        speech_status = "Ready" if last.get("speech_bytes") else "Pending"

    st.markdown("### Research dashboard")
    st.caption("Live overview of the current pipeline and recent experiments.")
    kpi_cols = st.columns(7)
    with kpi_cols[0]:
        render_kpi_card("Total runs", total_runs, icon="📊")
    with kpi_cols[1]:
        render_kpi_card("Avg runtime", f"{avg_runtime:.2f}s", icon="⏱️")
    with kpi_cols[2]:
        render_kpi_card("Audio duration", f"{audio_duration:.2f}s", icon="🎧")
    with kpi_cols[3]:
        render_kpi_card("Frames", frame_count, icon="🎞️")
    with kpi_cols[4]:
        render_kpi_card("Model", selected_model[:16] if selected_model not in (None, "N/A") else "N/A", icon="🧠")
    with kpi_cols[5]:
        render_kpi_card("Device", device.upper(), icon="💻")
    with kpi_cols[6]:
        render_kpi_card("Translation", translation_status, icon="🌐")

    st.write("")
    recent_cols = st.columns(2)
    with recent_cols[0]:
        st.markdown("<div class='panel'><h4>Recent experiment history</h4></div>", unsafe_allow_html=True)
        if recent:
            for run in recent:
                st.markdown(f"- Run #{run.get('id')} — {run.get('timestamp')} | {run.get('audio_name','N/A')}")
        else:
            st.info("No prior runs available yet.")
    with recent_cols[1]:
        st.markdown("<div class='panel'><h4>Pipeline status</h4></div>", unsafe_allow_html=True)
        st.markdown(f"- Translation: **{translation_status}**")
        st.markdown(f"- Speech generation: **{speech_status}**")
        st.markdown(f"- Selected device: **{device.upper()}**")
        st.markdown(f"- Audio duration: **{audio_duration:.2f}s**")
        st.markdown(f"- Extracted frames: **{frame_count}**")


def render_input_studio():
    st.title("Whisper AV Studio")
    st.write(
        "Upload an audio file and its matching video, preview both in the browser, "
        "inspect extracted video frames, and run the transcription/translation pipeline."
    )
    st.caption(
        f"Inference device: {DEVICE}. Maximum audio duration: {MAX_AUDIO_SECONDS} seconds."
    )

    render_live_webcam_section()

    sample_ready = DEFAULT_SAMPLE_AUDIO.exists() and DEFAULT_SAMPLE_VIDEO.exists()
    use_sample_files = False
    if sample_ready:
        st.success(
            f"Bundled sample files detected: {DEFAULT_SAMPLE_AUDIO.name} and "
            f"{DEFAULT_SAMPLE_VIDEO.name}"
        )
        use_sample_files = st.checkbox("Use bundled project sample files", value=False)
    else:
        st.caption("No bundled sample pair found; browse for audio and video files below.")

    st.subheader("Browse files and folders from your computer")
    st.write(
        "Choose a folder to upload all file types from it, including files inside "
        "its subfolders. The full relative path of each uploaded file is listed below. "
        "Then select one audio file and its matching video for the transcription pipeline."
    )

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

    folder_uploads = st.file_uploader(
        "Browse and select a folder (all file types)",
        type=None,
        accept_multiple_files=True,
        help=(
            "Select a folder on your computer. All files in that folder and its "
            "subfolders will be uploaded, not only audio/video files."
        ),
        key="all_files_folder_upload",
    )

    folder_uploads = folder_uploads or []
    audio_candidates = sorted(
        [f for f in folder_uploads if Path(f.name).suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda f: f.name.lower(),
    )
    video_candidates = sorted(
        [f for f in folder_uploads if Path(f.name).suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda f: f.name.lower(),
    )

    audio_upload = None
    video_upload = None

    if folder_uploads:
        total_mb = sum(f.size for f in folder_uploads) / (1024 * 1024)
        st.success(
            f"Folder uploaded: {len(folder_uploads)} file(s), {total_mb:.1f} MB total "
            f"({len(audio_candidates)} audio, {len(video_candidates)} video)."
        )
        with st.expander("View ALL uploaded files and their folder paths", expanded=True):
            file_rows = [
                {
                    "Relative path": uploaded.name,
                    "Extension": Path(uploaded.name).suffix.lower() or "(no extension)",
                    "Size (MB)": round(uploaded.size / (1024 * 1024), 3),
                }
                for uploaded in sorted(folder_uploads, key=lambda f: f.name.lower())
            ]
            st.dataframe(file_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Subfolders are represented by the relative paths. Empty folders are not "
            "uploaded by the browser because they contain no files."
        )

        selection_col1, selection_col2 = st.columns(2)
        with selection_col1:
            if audio_candidates:
                audio_index = st.selectbox(
                    "Choose audio file",
                    options=list(range(len(audio_candidates))),
                    format_func=lambda i: audio_candidates[i].name,
                    key="folder_audio_choice",
                )
                audio_upload = audio_candidates[audio_index]
            else:
                st.warning("No supported audio files were found in the selected folder.")
        with selection_col2:
            if video_candidates:
                video_index = st.selectbox(
                    "Choose matching video file",
                    options=list(range(len(video_candidates))),
                    format_func=lambda i: video_candidates[i].name,
                    key="folder_video_choice",
                )
                video_upload = video_candidates[video_index]
            else:
                st.warning("No supported video files were found in the selected folder.")

    with st.expander("Or choose individual audio and video files"):
        manual_audio = st.file_uploader(
            "Browse for one audio file",
            type=sorted(AUDIO_EXTENSIONS),
            key="manual_audio_upload",
        )
        manual_video = st.file_uploader(
            "Browse for one video file",
            type=sorted(VIDEO_EXTENSIONS),
            key="manual_video_upload",
        )
        if manual_audio is not None:
            audio_upload = manual_audio
        if manual_video is not None:
            video_upload = manual_video

    render_media_preview(audio_upload, video_upload)

    webcam_media = st.session_state.get("webcam_media")
    run_ready = ((audio_upload is not None and video_upload is not None) or use_sample_files or webcam_media is not None)
    if st.button("Run complete pipeline", type="primary", disabled=not run_ready):
        if webcam_media is None and not use_sample_files and (audio_upload is None or video_upload is None):
            st.warning("Please choose both an audio file and a matching video file, or enable the bundled sample set.")
            st.stop()

        stage_timing = {
            "audio_preprocessing": None,
            "video_preprocessing": None,
            "original_av_inference": None,
            "new_lora_inference": None,
            "token_fusion": None,
            "translation": None,
            "speech_generation": None,
            "total_pipeline": None,
        }
        pipeline_started = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            if webcam_media is not None:
                st.info("Running the pipeline with the live webcam capture as the source.")
                audio_path = webcam_media["audio"]
                video_path = webcam_media["video"]
            elif use_sample_files:
                audio_path = DEFAULT_SAMPLE_AUDIO
                video_path = DEFAULT_SAMPLE_VIDEO
                st.session_state["analysis_original_audio_path"] = str(audio_path)
            else:
                if audio_upload is None or video_upload is None:
                    st.warning("A valid audio and video input pair is required before running the pipeline.")
                    st.stop()
                audio_path = temp_dir / ("input_audio" + Path(audio_upload.name).suffix.lower())
                st.session_state["analysis_original_audio_path"] = str(persist_uploaded_media(audio_upload, "original_audio"))
                if video_upload is not None:
                    mp4_video = ensure_uploaded_video_mp4(video_upload)
                    if mp4_video is not None:
                        video_path = mp4_video
                    else:
                        video_path = temp_dir / ("input_video" + Path(video_upload.name).suffix.lower())
                        video_path.write_bytes(video_upload.getvalue())
                else:
                    video_path = temp_dir / "input_video.mp4"
                audio_path.write_bytes(audio_upload.getvalue())

            try:
                stage_timing["audio_preprocessing"] = time.perf_counter()
                with st.spinner("Step 1/3: Running original AV LoRA, new 2k audio LoRA, and local fusion..."):
                    start = time.perf_counter()
                    (
                        local_selected,
                        video_frames,
                        original_av_text,
                        new_2k_text,
                        fused_text,
                        selected_path,
                        fusion_error,
                    ) = recognize_combined(audio_path, video_path)
                    stage_timing["original_av_inference"] = time.perf_counter() - start
                    stage_timing["new_lora_inference"] = stage_timing["original_av_inference"]
                    stage_timing["token_fusion"] = stage_timing["original_av_inference"] / 2.0

                st.subheader("Extracted frames used by the model")
                st.caption(f"OpenCV extracted {len(video_frames)} grayscale frames for model input.")
                model_frame_cols = st.columns(4)
                for i, frame in enumerate(video_frames):
                    with model_frame_cols[i % 4]:
                        st.image(frame, caption=f"Model frame {i + 1}", clamp=True, use_container_width=True)

                st.subheader("Local model predictions")
                st.markdown("**Original LoRA + AV adapter**")
                st.write(original_av_text or "No output from this model.")
                st.markdown("**New 2k audio LoRA**")
                st.write(new_2k_text or "No output from this model.")
                st.markdown("**Experimental token-score fusion**")
                st.write(fused_text or "No output from fusion.")
                st.caption(f"Selected local fallback/path: {selected_path}")
                if fusion_error:
                    st.warning(f"Token-score fusion had an issue: {fusion_error}")

                selected_chinese = (local_selected or original_av_text or new_2k_text or fused_text or "").strip()
                if not selected_chinese:
                    st.error("No usable Chinese transcript is available for translation.")
                    st.stop()

                st.subheader("Chinese transcript used for translation")
                st.write(selected_chinese)
                translation_start = time.perf_counter()
                with st.spinner("Step 2/3: Translating the selected Chinese transcript into English..."):
                    english = translate_english(selected_chinese)
                stage_timing["translation"] = time.perf_counter() - translation_start
                st.markdown("## English translation")
                st.success(english)

                speech_start = time.perf_counter()
                speech_path = temp_dir / "english_speech.mp3"
                with st.spinner("Step 3/3: Generating English speech..."):
                    asyncio.run(generate_speech(english, speech_path))
                stage_timing["speech_generation"] = time.perf_counter() - speech_start
                speech = speech_path.read_bytes()
                persistent_speech = persist_generated_audio(speech, "generated_english")
                if persistent_speech is not None:
                    st.session_state["analysis_generated_audio_path"] = str(persistent_speech)
                st.subheader("English speech")
                st.audio(speech, format="audio/mpeg")
                st.download_button(
                    "Download English audio",
                    data=speech,
                    file_name="english_speech.mp3",
                    mime="audio/mpeg",
                )

                stage_timing["total_pipeline"] = time.perf_counter() - pipeline_started
                st.session_state["last_stage_metrics"] = stage_timing
                current_output = {
                    "id": len(st.session_state["history_log"]) + 1,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "audio_name": Path(audio_path.name).name,
                    "video_name": Path(video_path.name).name,
                    "chinese": selected_chinese,
                    "english": english,
                    "speech_bytes": speech,
                    "frame_count": len(video_frames),
                    "audio_duration": float(librosa.get_duration(path=str(audio_path))) if audio_path.exists() else 0.0,
                    "selected_path": selected_path,
                    "processing_time": stage_timing["total_pipeline"],
                    "original_audio_path": st.session_state.get("analysis_original_audio_path"),
                    "generated_audio_path": st.session_state.get("analysis_generated_audio_path"),
                }
                st.session_state["current_output"] = current_output
                st.session_state["history_log"].append(current_output)
                st.success("Pipeline completed. Review transcript candidates if the models disagree.")
            except Exception as exc:
                st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
                st.exception(exc)

    current = st.session_state.get("current_output")
    if current:
        st.divider()
        col_out_title, col_clear = st.columns([5, 1])
        with col_out_title:
            st.subheader(f"Pipeline Output — Run #{current['id']}")
            st.caption(
                f"Generated at {current['timestamp']} | Audio: `{current['audio_name']}` | Video: `{current['video_name']}`"
            )
        with col_clear:
            if st.button("🗑️ Clear Output", key="btn_clear_current"):
                st.session_state["current_output"] = None
                st.rerun()
        st.markdown("#### Predicted Chinese text")
        st.info(current["chinese"])
        st.caption(f"OpenCV extracted {current['frame_count']} frames.")
        st.markdown("#### English translation")
        st.success(current["english"])
        st.markdown("#### English speech")
        st.audio(current["speech_bytes"], format="audio/mpeg")
        st.download_button(
            label="Download English audio",
            data=current["speech_bytes"],
            file_name=f"english_speech_run_{current['id']}.mp3",
            mime="audio/mpeg",
            key=f"dl_btn_current_{current['id']}",
        )
        st.success("All pipeline stages completed.")

    history = st.session_state.get("history_log", [])
    if history:
        st.divider()
        with st.expander(f"📜 Session Output Log ({len(history)} run{'s' if len(history) > 1 else ''})", expanded=False):
            col_log_info, col_log_clear = st.columns([4, 1])
            with col_log_info:
                st.write("All pipeline outputs generated while this server is running are stored here:")
            with col_log_clear:
                if st.button("Clear Log History", key="btn_clear_log"):
                    st.session_state["history_log"] = []
                    st.session_state["current_output"] = None
                    st.rerun()
            for item in reversed(history):
                st.markdown(f"**Run #{item['id']}** ({item['timestamp']}) — Audio: `{item['audio_name']}` | Video: `{item['video_name']}`")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Chinese:**")
                    st.text_area("Chinese", item["chinese"], height=70, key=f"log_cn_{item['id']}", label_visibility="collapsed")
                with c2:
                    st.markdown("**English:**")
                    st.text_area("English", item["english"], height=70, key=f"log_en_{item['id']}", label_visibility="collapsed")
                col_aud, col_dl = st.columns([3, 1])
                with col_aud:
                    st.audio(item["speech_bytes"], format="audio/mpeg")
                with col_dl:
                    st.download_button(
                        label=f"Download Run #{item['id']} MP3",
                        data=item["speech_bytes"],
                        file_name=f"english_speech_run_{item['id']}.mp3",
                        mime="audio/mpeg",
                        key=f"dl_btn_log_{item['id']}",
                    )
                st.markdown("---")


def build_audio_analytics(audio_path):
    if audio_path is None:
        return {"audio": None, "plot_waveform": None, "plot_spectrogram": None}
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    silence_mask = np.abs(y) < 0.01
    silence_seconds = float(np.mean(silence_mask) * duration)
    times = np.linspace(0, duration, len(y))
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)
    waveform_fig = go.Figure()
    waveform_fig.add_scatter(x=times, y=y, mode="lines", line=dict(color="#6ee7b7", width=1))
    waveform_fig.update_layout(title="Waveform", template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    spectro_fig = go.Figure(data=go.Heatmap(z=mel_db, colorscale="Viridis", opacity=0.9))
    spectro_fig.update_layout(title="Mel spectrogram", template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
    return {
        "audio": y,
        "sample_rate": sr,
        "duration": duration,
        "samples": len(y),
        "peak_amplitude": peak,
        "rms_energy": rms,
        "silence_seconds": silence_seconds,
        "waveform_fig": waveform_fig,
        "spectrogram_fig": spectro_fig,
    }


def build_video_analytics(video_path):
    if video_path is None:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(total_frames / fps) if fps > 0 else 0.0
    frames = []
    times = []
    sample_indices = np.linspace(0, max(total_frames - 1, 0), min(total_frames, 8)).astype(int)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(float(idx / fps) if fps > 0 else 0.0)
    cap.release()
    return {
        "frame_count": total_frames,
        "fps": fps,
        "resolution": (width, height),
        "video_duration": duration,
        "sampled_frames": frames,
        "timestamps": times,
    }


def render_recognition_studio():
    st.header("Recognition Studio")
    current = st.session_state.get("current_output")
    if current is None:
        st.info("Run a pipeline once to populate comparison data.")
        return
    st.subheader("Model comparison")
    predictions = {
        "Original Whisper + AV adapter": current.get("original_av_text") or "N/A",
        "New 2K LoRA": current.get("new_lora_text") or "N/A",
        "Experimental fusion": current.get("fused_text") or "N/A",
    }
    for name, value in predictions.items():
        st.markdown(f"**{name}:** {value}")

    gt = st.text_area("Ground-truth Chinese transcript (optional)", value=current.get("ground_truth", ""))
    if gt.strip():
        metrics = compute_wer_cer(gt, current.get("chinese", ""))
        st.metric("WER", "N/A" if metrics["WER"] == "N/A" else f"{metrics['WER']:.4f}")
        st.metric("CER", "N/A" if metrics["CER"] == "N/A" else f"{metrics['CER']:.4f}")
    else:
        st.caption("Ground-truth transcript unavailable. WER/CER will appear as N/A until a reference is provided.")


def render_translation_studio():
    st.header("Translation Studio")
    current = st.session_state.get("current_output")
    if current is None:
        st.info("No completed translation is available yet.")
        return
    chinese = current.get("chinese", "")
    english = current.get("english", "")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chinese transcript")
        st.code(chinese)
        st.caption(f"Character count: {len(chinese)}")
    with col2:
        st.subheader("English translation")
        st.code(english)
        st.caption(f"Word count: {len(english.split())}")


def render_speech_studio():
    st.header("Speech Studio")
    current = st.session_state.get("current_output")
    if current is None:
        st.info("No generated speech is available yet.")
        return
    st.audio(current["speech_bytes"], format="audio/mpeg")
    st.download_button(
        "Download generated speech",
        data=current["speech_bytes"],
        file_name="generated_english_speech.mp3",
        mime="audio/mpeg",
    )


def render_performance_analytics():
    st.header("Performance Analytics")
    metrics = st.session_state.get("last_stage_metrics", {})
    active = [
        ("Audio preprocessing", metrics.get("audio_preprocessing")),
        ("Video preprocessing", metrics.get("video_preprocessing")),
        ("Original AV inference", metrics.get("original_av_inference")),
        ("New 2k LoRA inference", metrics.get("new_lora_inference")),
        ("Token fusion", metrics.get("token_fusion")),
        ("Translation", metrics.get("translation")),
        ("Speech generation", metrics.get("speech_generation")),
        ("Total pipeline", metrics.get("total_pipeline")),
    ]
    labels = [k for k, _ in active]
    values = [0.0 if v is None else float(v) for _, v in active]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="#7dd3fc")])
    fig.update_layout(template="plotly_dark", title="Pipeline execution time", margin=dict(l=20, r=20, t=30, b=80))
    st.plotly_chart(fig, use_container_width=True)

    ram = psutil.Process().memory_info().rss / (1024 * 1024)
    st.metric("Resident memory", f"{ram:.1f} MB")


def render_about():
    st.header("About the Project")
    st.markdown(
        """
        Whisper AV Studio is a research-focused interface for audio-visual speech recognition and translation.
        The project preserves the existing Whisper-small + LoRA + AV-residual architecture, while exposing a more modern dashboard for analysis, comparison, and experiment tracking.
        """
    )


def get_audio_analysis_payload(original_audio_path, generated_audio_path, fft_size=2048, pitch_min=50.0, pitch_max=500.0):
    original = load_audio_analysis(original_audio_path, fft_size=fft_size, pitch_min=pitch_min, pitch_max=pitch_max)
    generated = load_audio_analysis(generated_audio_path, fft_size=fft_size, pitch_min=pitch_min, pitch_max=pitch_max)
    return original, generated


def render_audio_visualisation_page():
    st.header("Audio Visualisation & Analysis")
    original_audio_path = st.session_state.get("analysis_original_audio_path") or (DEFAULT_SAMPLE_AUDIO if DEFAULT_SAMPLE_AUDIO.exists() else None)
    generated_audio_path = st.session_state.get("analysis_generated_audio_path")

    if not original_audio_path and not generated_audio_path:
        st.info("Run the pipeline once to produce the original and generated audio files for analysis.")
        return

    fft_size = st.slider("FFT size", 512, 8192, 2048, step=256)
    pitch_min = st.slider("Pitch minimum (Hz)", 30.0, 200.0, 50.0, step=5.0)
    pitch_max = st.slider("Pitch maximum (Hz)", 200.0, 2000.0, 500.0, step=50.0)

    original, generated = get_audio_analysis_payload(original_audio_path, generated_audio_path, fft_size=fft_size, pitch_min=pitch_min, pitch_max=pitch_max)

    audio_mode = st.radio("Select audio source", ["Original audio", "Generated audio", "Both"], index=2)

    col1, col2 = st.columns(2)
    with col1:
        if original_audio_path:
            st.caption("Original whispered audio")
            st.audio(str(original_audio_path))
    with col2:
        if generated_audio_path:
            st.caption("Generated English speech")
            st.audio(str(generated_audio_path))

    if audio_mode in {"Original audio", "Both"} and original is not None:
        fig = go.Figure()
        fig.add_scatter(x=original["times"], y=original["audio"], mode="lines", name="Original", line=dict(color="#7dd3fc", width=1.5))
        fig.update_layout(title="Original waveform", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode in {"Generated audio", "Both"} and generated is not None:
        fig = go.Figure()
        fig.add_scatter(x=generated["times"], y=generated["audio"], mode="lines", name="Generated", line=dict(color="#fbbf24", width=1.5))
        fig.update_layout(title="Generated waveform", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode == "Both" and original is not None and generated is not None:
        fig = go.Figure()
        fig.add_scatter(x=original["times"], y=original["audio"], mode="lines", name="Original", line=dict(color="#7dd3fc"))
        fig.add_scatter(x=generated["times"], y=generated["audio"], mode="lines", name="Generated", line=dict(color="#fbbf24"))
        fig.update_layout(title="Waveform comparison", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Amplitude", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode in {"Original audio", "Both"} and original is not None:
        pitch_times = original["pitch_times"][~np.isnan(original["pitch"])]
        pitch_vals = original["pitch"][~np.isnan(original["pitch"])]
        fig = go.Figure()
        fig.add_scatter(x=pitch_times, y=pitch_vals, mode="lines+markers", name="Original pitch", line=dict(color="#60a5fa"), marker=dict(size=4))
        fig.update_layout(title="Original pitch contour (F0)", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Pitch (Hz)")
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode in {"Generated audio", "Both"} and generated is not None:
        pitch_times = generated["pitch_times"][~np.isnan(generated["pitch"])]
        pitch_vals = generated["pitch"][~np.isnan(generated["pitch"])]
        fig = go.Figure()
        fig.add_scatter(x=pitch_times, y=pitch_vals, mode="lines+markers", name="Generated pitch", line=dict(color="#fca5a5"), marker=dict(size=4))
        fig.update_layout(title="Generated pitch contour (F0)", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Pitch (Hz)")
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode == "Both" and original is not None and generated is not None:
        fig = go.Figure()
        orig_x = original["pitch_times"][~np.isnan(original["pitch"])]
        orig_y = original["pitch"][~np.isnan(original["pitch"])]
        gen_x = generated["pitch_times"][~np.isnan(generated["pitch"])]
        gen_y = generated["pitch"][~np.isnan(generated["pitch"])]
        fig.add_scatter(x=orig_x, y=orig_y, mode="lines+markers", name="Original", line=dict(color="#7dd3fc"), marker=dict(size=4))
        fig.add_scatter(x=gen_x, y=gen_y, mode="lines+markers", name="Generated", line=dict(color="#fbbf24"), marker=dict(size=4))
        fig.update_layout(title="Pitch comparison", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Pitch (Hz)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode in {"Original audio", "Both"} and original is not None:
        fig = go.Figure()
        fig.add_scatter(x=original["spectrum_freqs"], y=original["spectrum_mag"], mode="lines", name="Original", line=dict(color="#a78bfa"))
        fig.update_layout(title="Original FFT magnitude spectrum", template="plotly_dark", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode in {"Generated audio", "Both"} and generated is not None:
        fig = go.Figure()
        fig.add_scatter(x=generated["spectrum_freqs"], y=generated["spectrum_mag"], mode="lines", name="Generated", line=dict(color="#fb7185"))
        fig.update_layout(title="Generated FFT magnitude spectrum", template="plotly_dark", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)")
        st.plotly_chart(fig, use_container_width=True)

    if audio_mode == "Both" and original is not None and generated is not None:
        fig = go.Figure()
        fig.add_scatter(x=original["spectrum_freqs"], y=original["spectrum_mag"], mode="lines", name="Original", line=dict(color="#a78bfa"))
        fig.add_scatter(x=generated["spectrum_freqs"], y=generated["spectrum_mag"], mode="lines", name="Generated", line=dict(color="#fb7185"))
        fig.update_layout(title="FFT spectrum comparison", template="plotly_dark", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    if MATPLOTLIB_AVAILABLE:
        if audio_mode in {"Original audio", "Both"} and original is not None:
            st.subheader("Original STFT spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            img = ax.imshow(original["stft_db"], origin="lower", aspect="auto", cmap="magma", extent=[0, original["duration"], original["stft_freqs"][0], original["stft_freqs"][-1]])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(img, ax=ax, format="%.0f dB")
            st.pyplot(fig)
            plt.close(fig)

        if audio_mode in {"Generated audio", "Both"} and generated is not None:
            st.subheader("Generated STFT spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            img = ax.imshow(generated["stft_db"], origin="lower", aspect="auto", cmap="magma", extent=[0, generated["duration"], generated["stft_freqs"][0], generated["stft_freqs"][-1]])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(img, ax=ax, format="%.0f dB")
            st.pyplot(fig)
            plt.close(fig)

        if audio_mode in {"Original audio", "Both"} and original is not None:
            st.subheader("Original mel spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            img = ax.imshow(original["mel_db"], origin="lower", aspect="auto", cmap="viridis", extent=[0, original["duration"], 0, original["mel_freqs"][-1]])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mel frequency")
            plt.colorbar(img, ax=ax, format="%.0f dB")
            st.pyplot(fig)
            plt.close(fig)

        if audio_mode in {"Generated audio", "Both"} and generated is not None:
            st.subheader("Generated mel spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            img = ax.imshow(generated["mel_db"], origin="lower", aspect="auto", cmap="viridis", extent=[0, generated["duration"], 0, generated["mel_freqs"][-1]])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mel frequency")
            plt.colorbar(img, ax=ax, format="%.0f dB")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("Matplotlib is not installed in this environment. STFT and Mel spectrogram heatmaps are unavailable until matplotlib is added to the venv.")

    pd_rows = []
    if original is not None:
        pd_rows.append({
            "signal": "Original",
            "duration_s": original["duration"],
            "sample_rate_hz": original["sr"],
            "peak_amplitude": float(np.max(np.abs(original["audio"]))),
            "rms_energy": float(np.sqrt(np.mean(original["audio"] ** 2))),
            "dominant_frequency_hz": original["dominant_frequency"],
            "dominant_magnitude_db": original["dominant_db"],
            "median_pitch_hz": float(np.nanmedian(original["pitch"])) if np.any(~np.isnan(original["pitch"])) else np.nan,
        })
    if generated is not None:
        pd_rows.append({
            "signal": "Generated",
            "duration_s": generated["duration"],
            "sample_rate_hz": generated["sr"],
            "peak_amplitude": float(np.max(np.abs(generated["audio"]))),
            "rms_energy": float(np.sqrt(np.mean(generated["audio"] ** 2))),
            "dominant_frequency_hz": generated["dominant_frequency"],
            "dominant_magnitude_db": generated["dominant_db"],
            "median_pitch_hz": float(np.nanmedian(generated["pitch"])) if np.any(~np.isnan(generated["pitch"])) else np.nan,
        })
    if pd_rows:
        st.subheader("Acoustic feature comparison")
        df = pd.DataFrame(pd_rows)
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download acoustic measurements CSV", data=csv_bytes, file_name="acoustic_measurements.csv", mime="text/csv")


def generate_experiment_pdf(run):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "PDF export requires ReportLab, which is not supported in Python 3.9. "
            "Use Python 3.10+ or disable PDF export."
        )
    pdf_path = ROOT / f"experiment_report_{run['id']}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.setTitle("Whisper AV Studio Experiment Report")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1 * inch, 10.5 * inch, "Whisper AV Studio")
    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, 10.1 * inch, f"Experiment ID: {run['id']}")
    c.drawString(1 * inch, 9.9 * inch, f"Timestamp: {run['timestamp']}")
    c.drawString(1 * inch, 9.7 * inch, f"Audio file: {run['audio_name']}")
    c.drawString(1 * inch, 9.5 * inch, f"Video file: {run['video_name']}")
    c.drawString(1 * inch, 9.1 * inch, f"Chinese transcript: {run['chinese'][:200]}")
    c.drawString(1 * inch, 8.8 * inch, f"English translation: {run['english'][:200]}")
    c.save()
    return pdf_path


def render_history():
    st.header("Experiment History")
    history = st.session_state.get("history_log", [])
    if not history:
        st.info("No experiment history yet.")
        return

    df = pd.DataFrame(history)
    st.dataframe(df[["id", "timestamp", "audio_name", "video_name", "processing_time"]], use_container_width=True)

    latest = history[-1]
    if REPORTLAB_AVAILABLE:
        if st.button("Download latest PDF report"):
            try:
                pdf_path = generate_experiment_pdf(latest)
                st.download_button("Save PDF", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
            except Exception as exc:
                st.warning(f"PDF export failed: {type(exc).__name__}: {exc}")
    else:
        st.warning(
            "PDF export is unavailable because this environment is on Python 3.9. "
            "ReportLab requires Python 3.10+. Use Python 3.10+ or disable PDF export."
        )


def render_recent_pages():
    selected = option_menu(
        "Whisper AV Studio",
        ["Dashboard", "Input Studio", "Translation Studio", "Speech Studio", "Audio Visualisation & Analysis", "Performance Analytics", "Experiment History", "About the Project"],
        icons=["speedometer2", "upload", "translate", "volume-up", "bar-chart-line-fill", "graph-up-arrow", "clock-history", "info-circle"],
        default_index=0,
        orientation="horizontal",
    )
    if selected == "Dashboard":
        render_dashboard()
    elif selected == "Input Studio":
        render_input_studio()
    elif selected == "Translation Studio":
        render_translation_studio()
    elif selected == "Speech Studio":
        render_speech_studio()
    elif selected == "Audio Visualisation & Analysis":
        render_audio_visualisation_page()
    elif selected == "Performance Analytics":
        render_performance_analytics()
    elif selected == "Experiment History":
        render_history()
    elif selected == "About the Project":
        render_about()


ANALYSIS_CACHE_DIR = ROOT / ".analysis_cache"
ANALYSIS_CACHE_DIR.mkdir(exist_ok=True)

OLD_LORA_DIR = ROOT / "outputs" / "whisper_lora_small"

NEW_LORA_DIR = ROOT / "outputs" / "whisper_lora_aishell6_2k"

OLD_AV_CHECKPOINT = ROOT / "outputs" / "av_transcription_full.pt"

MAX_AUDIO_SECONDS = 120

MAX_NEW_TOKENS = 128



if torch.cuda.is_available():

    DEVICE = "cuda"

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():

    DEVICE = "mps"

else:

    DEVICE = "cpu"





# -----------------------------------------------------------------------------

# Original AV adapter architecture: must match the saved checkpoint.

# -----------------------------------------------------------------------------

class AVResidualAdapter(nn.Module):

    def __init__(self, audio_dim=768, hidden_dim=256, video_dim=128):

        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.video_encoder = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten(start_dim=1),

            nn.Linear(256, video_dim),

            nn.ReLU(),

        )

        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(

            hidden_dim, 4, batch_first=True

        )

        self.out_proj = nn.Linear(hidden_dim, audio_dim)

        self.gate = nn.Parameter(torch.tensor(0.1))



    def forward(self, audio_hidden, video):

        audio_q = self.audio_proj(audio_hidden)

        batch, frames, height, width = video.shape

        video_features = self.video_encoder(

            video.reshape(batch * frames, 1, height, width)

        )

        video_features = self.video_proj(

            video_features.reshape(batch, frames, -1)

        )

        attended, _ = self.cross_attention(

            audio_q, video_features, video_features, need_weights=False

        )

        return audio_hidden + self.gate * self.out_proj(attended)





# -----------------------------------------------------------------------------

# Load models

# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading original Whisper + AV adapter...")

def load_original_av_models():

    if not OLD_LORA_DIR.is_dir():

        raise FileNotFoundError(f"Missing original LoRA adapter: {OLD_LORA_DIR}")

    if not OLD_AV_CHECKPOINT.is_file():

        raise FileNotFoundError(f"Missing original AV checkpoint: {OLD_AV_CHECKPOINT}")



    processor = WhisperProcessor.from_pretrained(str(OLD_LORA_DIR))

    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    whisper = PeftModel.from_pretrained(base, str(OLD_LORA_DIR))

    whisper = whisper.to(DEVICE).eval()



    checkpoint = torch.load(OLD_AV_CHECKPOINT, map_location="cpu")

    if "av_adapter" not in checkpoint:

        raise KeyError(

            f"The AV checkpoint has no 'av_adapter' key: {OLD_AV_CHECKPOINT}"

        )

    adapter = AVResidualAdapter(audio_dim=whisper.config.d_model)

    adapter.load_state_dict(checkpoint["av_adapter"])

    adapter = adapter.to(DEVICE).eval()

    return processor, whisper, adapter





@st.cache_resource(show_spinner="Loading new 2k fine-tuned Whisper LoRA...")

def load_new_audio_model():

    if not NEW_LORA_DIR.is_dir():

        raise FileNotFoundError(f"Missing new 2k LoRA adapter: {NEW_LORA_DIR}")



    processor = WhisperProcessor.from_pretrained(str(NEW_LORA_DIR))

    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    whisper = PeftModel.from_pretrained(base, str(NEW_LORA_DIR))

    return processor, whisper.to(DEVICE).eval()





@st.cache_resource(show_spinner="Loading Chinese-to-English translation model...")

def load_translation_model():

    model_name = "Helsinki-NLP/opus-mt-zh-en"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(

        model_name, use_safetensors=False

    ).to(DEVICE).eval()

    return tokenizer, model





# -----------------------------------------------------------------------------

# Audio/video preprocessing

# -----------------------------------------------------------------------------

def load_audio_16k(audio_path):

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")

    if audio.ndim > 1:

        audio = audio.mean(axis=1)

    if audio.size == 0:

        raise ValueError("The uploaded audio file is empty.")

    if sample_rate <= 0:

        raise ValueError(f"Invalid audio sample rate: {sample_rate}")

    duration = len(audio) / sample_rate

    if duration > MAX_AUDIO_SECONDS:

        raise ValueError(f"Please use an audio clip of {MAX_AUDIO_SECONDS} seconds or less.")

    if sample_rate != 16000:

        audio = librosa.resample(

            audio, orig_sr=sample_rate, target_sr=16000

        ).astype(np.float32)

    return np.nan_to_num(audio).astype(np.float32)





def extract_video_frames(video_path, num_frames=16):

    """Return num_frames grayscale frames, each resized to 48x48, in [0, 1]."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise ValueError("OpenCV cannot open the supplied video.")



    frames = []

    try:

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:

            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

            for index in indices:

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

                ok, frame = cap.read()

                if not ok:

                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray = cv2.resize(gray, (48, 48))

                frames.append(gray.astype(np.float32) / 255.0)

        else:

            while len(frames) < num_frames:

                ok, frame = cap.read()

                if not ok:

                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray = cv2.resize(gray, (48, 48))

                frames.append(gray.astype(np.float32) / 255.0)

    finally:

        cap.release()



    if not frames:

        raise ValueError("No video frames could be extracted from the supplied video.")

    while len(frames) < num_frames:

        frames.append(frames[-1].copy())

    return np.stack(frames[:num_frames])





def extract_display_frames(video_path, num_frames=8):
    """Extract representative full-resolution RGB frames for the Streamlit preview."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("OpenCV cannot open the supplied video for frame preview.")
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            indices = np.linspace(0, total - 1, min(num_frames, total)).astype(int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            while len(frames) < num_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def write_frames_to_mp4(frames, output_path, fps=10):
    """Write RGB or grayscale frames to a browser-playable MP4 file."""
    if not frames:
        raise ValueError("No frames available to write to MP4.")

    if not output_path.name.endswith(".mp4"):
        output_path = output_path.with_suffix(".mp4")

    first = np.asarray(frames[0])
    if first.ndim == 2:
        height, width = first.shape
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in frames]
    elif first.ndim == 3 and first.shape[2] == 4:
        height, width = first.shape[:2]
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in frames]
    else:
        height, width = first.shape[:2]
        rgb_frames = [frame.copy() for frame in frames]

    if WEBRTC_AVAILABLE and "av" in globals():
        container = av.open(str(output_path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        try:
            for rgb in rgb_frames:
                frame = av.VideoFrame.from_ndarray(rgb[:, :, ::-1], format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        return output_path

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create a temporary MP4 video file.")

    try:
        for rgb in rgb_frames:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return output_path


def build_webcam_capture(video_frame, duration_seconds=3, fps=10):
    """Create a short webcam video clip with silent audio for fallback pipeline use."""
    if video_frame is None:
        raise ValueError("No webcam frame is available to convert into a pipeline source.")

    frame_array = np.asarray(video_frame)
    if frame_array.ndim == 2:
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
    elif frame_array.shape[2] == 4:
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2RGB)
    elif frame_array.shape[2] == 3:
        frame_array = frame_array[:, :, ::-1]

    temp_dir = Path(tempfile.mkdtemp(prefix="webcam_capture_"))
    video_path = temp_dir / "webcam_capture.mp4"
    frame_list = [frame_array.copy() for _ in range(max(1, int(duration_seconds * fps)))]
    write_frames_to_mp4(frame_list, video_path, fps=fps)

    audio_path = temp_dir / "webcam_audio.wav"
    sample_rate = 16000
    audio = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
    sf.write(str(audio_path), audio, sample_rate)

    return {"video": video_path, "audio": audio_path, "temp_dir": temp_dir}


class WebcamPipelineBuffer:
    """Store camera frames/audio chunks for a short live capture."""

    def __init__(self, max_video_frames=180, sample_rate=16000):
        self.max_video_frames = max_video_frames
        self.sample_rate = sample_rate
        self.latest_frame = None
        self.video_frames = []
        self.audio_chunks = []
        self._lock = threading.Lock()

    def update_video(self, frame_rgb):
        with self._lock:
            if frame_rgb is None:
                return
            rgb = np.asarray(frame_rgb)
            self.latest_frame = rgb.copy()
            self.video_frames.append(rgb.copy())
            if len(self.video_frames) > self.max_video_frames:
                self.video_frames.pop(0)

    def update_audio(self, audio_frame):
        if audio_frame is None:
            return
        try:
            chunk = audio_frame.to_ndarray().astype(np.float32)
        except Exception:
            return

        if chunk.size == 0:
            return

        if chunk.ndim == 2:
            chunk = chunk.mean(axis=1) if chunk.shape[1] > 1 else chunk[:, 0]
        elif chunk.ndim == 1:
            pass
        else:
            chunk = chunk.reshape(-1)

        with self._lock:
            self.audio_chunks.append(chunk.copy())
            total_samples = sum(len(item) for item in self.audio_chunks)
            while total_samples > self.sample_rate * 20 and self.audio_chunks:
                removed = self.audio_chunks.pop(0)
                total_samples -= len(removed)

    def export(self):
        with self._lock:
            if not self.video_frames:
                raise ValueError("No live webcam frames were captured.")

            frames = [np.asarray(frame) for frame in self.video_frames]
            if not frames:
                raise ValueError("No live webcam frames were captured.")

            temp_dir = Path(tempfile.mkdtemp(prefix="webcam_pipeline_"))
            video_path = temp_dir / "webcam_live.mp4"
            write_frames_to_mp4(frames, video_path, fps=10)

            audio_samples = []
            for chunk in self.audio_chunks:
                if chunk.size:
                    audio_samples.append(np.asarray(chunk, dtype=np.float32))

            if not audio_samples:
                audio_path = temp_dir / "webcam_live.wav"
                silent = np.zeros(int(self.sample_rate * 1), dtype=np.float32)
                sf.write(str(audio_path), silent, self.sample_rate)
            else:
                audio_path = temp_dir / "webcam_live.wav"
                audio_data = np.concatenate(audio_samples).astype(np.float32)
                if audio_data.ndim > 1:
                    audio_data = audio_data.reshape(-1)
                audio_data = np.nan_to_num(audio_data)
                sf.write(str(audio_path), audio_data, self.sample_rate)

            return {"video": video_path, "audio": audio_path, "temp_dir": temp_dir}


if WEBRTC_AVAILABLE:
    class WebcamVideoProcessor(VideoProcessorBase):
        def __init__(self, buffer):
            self.buffer = buffer

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            self.buffer.update_video(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return av.VideoFrame.from_ndarray(image, format="bgr24")


    class WebcamAudioProcessor(AudioProcessorBase):
        def __init__(self, buffer):
            self.buffer = buffer

        def recv(self, frame):
            self.buffer.update_audio(frame)
            return frame


def render_media_player(data, name, media_type):
    """Render uploaded media with browser-safe file paths for reliable playback."""
    if data is None:
        return

    if media_type == "audio":
        if hasattr(data, "read"):
            st.audio(data)
        elif isinstance(data, (str, os.PathLike)):
            st.audio(str(data))
        else:
            st.audio(data)
        return

    if hasattr(data, "read"):
        raw_bytes = data.getvalue()
    elif isinstance(data, (str, os.PathLike)):
        raw_bytes = Path(data).read_bytes() if Path(data).exists() else b""
    else:
        raw_bytes = data

    if not raw_bytes:
        return

    suffix = Path(str(name)).suffix.lower()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_path = tmp_path / f"input{suffix or '.avi'}"
            src_path.write_bytes(raw_bytes)

            if suffix in {".mp4", ".webm", ".ogg", ".mov"}:
                st.video(str(src_path))
                return

            cap = cv2.VideoCapture(str(src_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot decode video for browser playback: {name}")

            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if not frames:
                raise ValueError("No frames were extracted from the uploaded video.")

            converted_path = tmp_path / "browser_preview.mp4"
            write_frames_to_mp4(frames, converted_path, fps=10)
            st.video(str(converted_path))
    except Exception as exc:
        st.warning(f"This video format cannot play in the browser: {type(exc).__name__}: {exc}")


# Always render a direct local file preview for uploaded videos so the browser can play them reliably.
def convert_uploaded_video_to_mp4(video_upload):
    """Convert an uploaded video to a persistent browser-safe MP4 file."""
    if video_upload is None:
        return None

    raw = video_upload.getvalue()
    if not raw:
        return None

    source_name = Path(video_upload.name).name
    source_suffix = Path(video_upload.name).suffix.lower()
    cache_dir = ROOT / ".video_preview_cache"
    cache_dir.mkdir(exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", source_name)
    file_key = f"{safe_name}_{abs(hash(video_upload.name + str(len(raw))))}"
    source = cache_dir / f"{file_key}{source_suffix or '.avi'}"
    source.write_bytes(raw)

    # If the uploaded file is already MP4, keep it as-is.
    if source_suffix == ".mp4":
        return source

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"The uploaded video '{source_name}' could not be decoded.")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be extracted from '{source_name}'.")

    mp4_path = cache_dir / f"{file_key}_preview.mp4"
    write_frames_to_mp4(frames, mp4_path, fps=10)
    return mp4_path


def render_uploaded_video(video_upload):
    """Convert uploaded video to MP4 and display it in the browser using a persistent file path."""
    if video_upload is None:
        return

    try:
        mp4_path = convert_uploaded_video_to_mp4(video_upload)
        if mp4_path is None:
            st.warning("Uploaded video is empty and cannot be previewed.")
            return
        st.session_state["uploaded_video_mp4_path"] = str(mp4_path)
        st.caption("Converted to MP4 for browser playback.")
        st.video(st.session_state["uploaded_video_mp4_path"])
    except Exception as exc:
        st.warning(f"Video preview could not be generated: {type(exc).__name__}: {exc}")


def ensure_uploaded_video_mp4(video_upload):
    """Return a browser-safe MP4 path for uploaded video and keep the converted file on disk."""
    if video_upload is None:
        return None

    return convert_uploaded_video_to_mp4(video_upload)


def render_live_webcam_section():
    """Render a live webcam preview and capture a real browser mic+video source for the pipeline."""
    st.subheader("Live webcam")
    if WEBRTC_AVAILABLE:
        st.caption("Allow camera and microphone access in your browser to start the live stream.")

        if "webcam_pipeline_buffer" not in st.session_state:
            st.session_state["webcam_pipeline_buffer"] = WebcamPipelineBuffer()

        webcam_buffer = st.session_state["webcam_pipeline_buffer"]
        webcam_ctx = webrtc_streamer(
            key="live-webcam-preview",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: WebcamVideoProcessor(webcam_buffer),
            audio_processor_factory=lambda: WebcamAudioProcessor(webcam_buffer),
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
        )

        if getattr(webcam_ctx, "state", None) is not None and webcam_ctx.state.playing:
            st.success("Webcam stream is active.")
            latest = webcam_buffer.latest_frame
            if latest is not None:
                st.image(latest, caption="Latest webcam frame", use_container_width=True)

            if st.button("Capture live webcam for pipeline", key="capture_webcam_pipeline"):
                try:
                    st.session_state["webcam_media"] = webcam_buffer.export()
                    st.success("Live webcam media captured and ready for the pipeline.")
                except Exception as exc:
                    st.error(f"Could not capture live webcam media: {type(exc).__name__}: {exc}")
        elif getattr(webcam_ctx, "state", None) is not None and not webcam_ctx.state.playing:
            st.info("Webcam is ready to start when you allow browser camera access.")

        if st.session_state.get("webcam_media") is not None:
            webcam_media = st.session_state["webcam_media"]
            st.caption("Current live webcam source for the pipeline")
            video_bytes = Path(webcam_media["video"]).read_bytes() if Path(webcam_media["video"]).exists() else b""
            audio_bytes = Path(webcam_media["audio"]).read_bytes() if Path(webcam_media["audio"]).exists() else b""
            if video_bytes:
                render_media_player(video_bytes, "webcam_capture.mp4", "video")
            if audio_bytes:
                render_media_player(audio_bytes, "webcam_capture.wav", "audio")
    else:
        st.info(
            "For a live webcam stream, install the optional dependency with "
            "`pip install streamlit-webrtc av`. Streamlit's built-in camera input "
            "below is available as a still-image fallback."
        )
        camera_image = st.camera_input("Capture a webcam image")
        if camera_image is not None:
            st.image(camera_image, caption="Captured webcam image", use_container_width=True)
            if st.button("Use captured webcam image as pipeline input", key="use_camera_snapshot_input"):
                try:
                    capture = build_webcam_capture(np.array(camera_image))
                    st.session_state["webcam_media"] = capture
                    st.success("Captured webcam snapshot is ready for the pipeline.")
                except Exception as exc:
                    st.error(f"Could not convert captured webcam image: {type(exc).__name__}: {exc}")


def render_media_preview(audio_upload, video_upload):
    """Display uploaded audio/video and extracted frames in the same browser session."""
    if audio_upload is None and video_upload is None:
        return

    st.subheader("Preview uploaded media")
    preview_cols = st.columns(2)

    if audio_upload is not None:
        with preview_cols[0]:
            st.markdown(f"**Audio:** `{audio_upload.name}`")
            render_media_player(audio_upload.getvalue(), audio_upload.name, "audio")

    if video_upload is not None:
        with preview_cols[1]:
            st.markdown(f"**Video:** `{video_upload.name}`")
            render_uploaded_video(video_upload)

    if video_upload is not None:
        try:
            suffix = Path(video_upload.name).suffix or ".mp4"
            with tempfile.TemporaryDirectory() as preview_dir:
                preview_path = Path(preview_dir) / f"preview{suffix}"
                preview_path.write_bytes(video_upload.getvalue())
                display_frames = extract_display_frames(preview_path, num_frames=8)

            if display_frames:
                st.subheader("Extracted frames")
                st.caption(f"Showing {len(display_frames)} representative frames.")
                frame_cols = st.columns(4)
                for i, frame in enumerate(display_frames):
                    with frame_cols[i % 4]:
                        st.image(frame, caption=f"Frame {i + 1}", use_container_width=True)
            else:
                st.warning("No preview frames could be extracted from this video.")
        except Exception as exc:
            st.warning(f"Could not preview extracted frames: {type(exc).__name__}: {exc}")


if WEBRTC_AVAILABLE:
    class WebcamProcessor(VideoProcessorBase):
        """Keep the latest webcam frame while WebRTC renders the live stream."""
        def __init__(self):
            self.latest_frame = None
            self.frame_buffer = []
            self._lock = threading.Lock()

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with self._lock:
                self.latest_frame = rgb.copy()
                self.frame_buffer.append(rgb.copy())
                if len(self.frame_buffer) > 90:
                    self.frame_buffer.pop(0)
            return av.VideoFrame.from_ndarray(image, format="bgr24")


def make_decoder_prefix(processor, model):

    """Build [decoder-start, language, task, timestamp-mode] prefix for Whisper."""

    start_id = model.config.decoder_start_token_id

    if start_id is None:

        start_id = model.generation_config.decoder_start_token_id

    if start_id is None:

        raise RuntimeError("Could not determine Whisper decoder-start token ID.")



    prompt_pairs = processor.tokenizer.get_decoder_prompt_ids(

        language="chinese", task="transcribe"

    )

    prefix = [int(start_id)]

    for position, token_id in sorted(prompt_pairs, key=lambda item: item[0]):

        # Whisper forced prompt positions are absolute decoder positions.

        while len(prefix) < int(position):

            prefix.append(int(token_id))

        if len(prefix) == int(position):

            prefix.append(int(token_id))

        else:

            prefix[int(position)] = int(token_id)

    return prefix





# -----------------------------------------------------------------------------

# Combined transcription: old LoRA + old AV adapter and new 2k audio LoRA

# -----------------------------------------------------------------------------

def recognize_combined(audio_path, video_path):

    """Fuse next-token distributions; fall back to a nonempty model transcript."""

    old_processor, old_whisper, av_adapter = load_original_av_models()

    new_processor, new_whisper = load_new_audio_model()



    # Both are Whisper-small checkpoints and should share the same vocabulary.

    processor = old_processor



    audio = load_audio_16k(audio_path)

    input_features = processor(

        audio, sampling_rate=16000, return_tensors="pt"

    ).input_features.to(DEVICE)



    frames = extract_video_frames(video_path)

    video = torch.as_tensor(

        frames, dtype=torch.float32, device=DEVICE

    ).unsqueeze(0)



    with torch.inference_mode():

        old_audio_hidden = old_whisper.get_encoder()(

            input_features=input_features

        ).last_hidden_state

        old_av_hidden = av_adapter(old_audio_hidden, video)

        new_audio_hidden = new_whisper.get_encoder()(

            input_features=input_features

        ).last_hidden_state



        # Always generate independent predictions first. They are safe fallbacks

        # if the experimental fused decoder returns no readable text.

        old_tokens = old_whisper.generate(

            input_features=input_features,

            encoder_outputs=BaseModelOutput(

                last_hidden_state=old_av_hidden

            ),

            language="chinese",

            task="transcribe",

            max_new_tokens=MAX_NEW_TOKENS,

            num_beams=1,

        )

        new_tokens = new_whisper.generate(

            input_features=input_features,

            language="chinese",

            task="transcribe",

            max_new_tokens=MAX_NEW_TOKENS,

            num_beams=1,

        )



    old_text = old_processor.batch_decode(

        old_tokens, skip_special_tokens=True

    )[0].strip()

    new_text = new_processor.batch_decode(

        new_tokens, skip_special_tokens=True

    )[0].strip()



    fused_text = ""

    fusion_error = None

    try:

        prefix_ids = make_decoder_prefix(processor, old_whisper)

        decoder_ids = torch.tensor(

            [prefix_ids], dtype=torch.long, device=DEVICE

        )



        eos_id = old_whisper.generation_config.eos_token_id

        if eos_id is None:

            eos_id = old_whisper.config.eos_token_id

        if isinstance(eos_id, (tuple, list)):

            eos_ids = {int(item) for item in eos_id}

        elif eos_id is None:

            eos_ids = set()

        else:

            eos_ids = {int(eos_id)}



        generated_ids = []

        # These are initial experimental weights, not validated optimums.

        old_av_weight = 0.35

        new_audio_weight = 0.65



        with torch.inference_mode():

            for _ in range(MAX_NEW_TOKENS):

                old_logits = old_whisper(

                    encoder_outputs=BaseModelOutput(

                        last_hidden_state=old_av_hidden

                    ),

                    decoder_input_ids=decoder_ids,

                    use_cache=False,

                    return_dict=True,

                ).logits[:, -1, :]



                new_logits = new_whisper(

                    encoder_outputs=BaseModelOutput(

                        last_hidden_state=new_audio_hidden

                    ),

                    decoder_input_ids=decoder_ids,

                    use_cache=False,

                    return_dict=True,

                ).logits[:, -1, :]



                old_log_probs = torch.log_softmax(old_logits.float(), dim=-1)

                new_log_probs = torch.log_softmax(new_logits.float(), dim=-1)

                combined_log_probs = (

                    old_av_weight * old_log_probs

                    + new_audio_weight * new_log_probs

                )

                next_token = int(combined_log_probs.argmax(dim=-1).item())

                if next_token in eos_ids:

                    break



                generated_ids.append(next_token)

                next_tensor = torch.tensor(

                    [[next_token]], dtype=torch.long, device=DEVICE

                )

                decoder_ids = torch.cat([decoder_ids, next_tensor], dim=1)



        if generated_ids:

            fused_text = processor.batch_decode(

                [generated_ids], skip_special_tokens=True

            )[0].strip()

    except Exception as exc:

        fusion_error = f"{type(exc).__name__}: {exc}"

        print(f"[fusion] Decoder failed: {fusion_error}")



    # Never fail the entire pipeline just because fusion is blank.

    # Prefer fusion if it returned readable text; otherwise use the new LoRA,

    # then the old AV path. This fallback is explicit in the UI below.

    if fused_text:

        final_text = fused_text

        selected_path = "Fused token-level prediction"

    elif new_text:

        final_text = new_text

        selected_path = "Fallback: new 2k LoRA audio-only prediction"

    elif old_text:

        final_text = old_text

        selected_path = "Fallback: original LoRA + AV prediction"

    else:

        raise RuntimeError(

            "All transcription outputs were empty. See terminal output for model details."

        )



    print(f"[transcription] original AV = {old_text!r}")

    print(f"[transcription] new audio LoRA = {new_text!r}")

    print(f"[transcription] fused = {fused_text!r}")

    print(f"[transcription] selected = {selected_path}; text = {final_text!r}")



    return final_text, frames, old_text, new_text, fused_text, selected_path, fusion_error





def recognize_chinese(audio_path, video_path):

    """Compatibility wrapper for the existing app pipeline."""

    result = recognize_combined(audio_path, video_path)

    chinese = result[0]

    video_data = result[1]

    if isinstance(video_data, (int, np.integer)):

        frame_count = int(video_data)

    else:

        frame_count = len(video_data)

    return chinese, frame_count





# -----------------------------------------------------------------------------

# Translation and speech generation

# -----------------------------------------------------------------------------

def translate_english(chinese_text):

    tokenizer, model = load_translation_model()

    normalized_text = OpenCC("t2s").convert(chinese_text)

    inputs = tokenizer(

        normalized_text,

        return_tensors="pt",

        truncation=True,

        max_length=512,

    ).to(DEVICE)



    with torch.inference_mode():

        output = model.generate(

            **inputs,

            max_new_tokens=256,

            num_beams=4,

        )

    translation = tokenizer.batch_decode(

        output, skip_special_tokens=True

    )[0].strip()

    if not translation:

        raise ValueError("OPUS returned an empty translation.")

    return translation





async def generate_speech(text, output_path):

    communicator = edge_tts.Communicate(

        text, voice="en-US-AriaNeural"

    )

    await communicator.save(str(output_path))



render_recent_pages()
st.stop()


st.title("Audio-Visual Chinese-to-English Pipeline")
st.write(
    "Upload an audio file and its matching video, preview both in the browser, "
    "inspect extracted video frames, and run the transcription/translation pipeline."
)
st.caption(
    f"Inference device: {DEVICE}. Maximum audio duration: {MAX_AUDIO_SECONDS} seconds."
)

render_live_webcam_section()

sample_ready = DEFAULT_SAMPLE_AUDIO.exists() and DEFAULT_SAMPLE_VIDEO.exists()
use_sample_files = False
if sample_ready:
    st.success(
        f"Bundled sample files detected: {DEFAULT_SAMPLE_AUDIO.name} and "
        f"{DEFAULT_SAMPLE_VIDEO.name}"
    )
    use_sample_files = st.checkbox("Use bundled project sample files", value=False)
else:
    st.caption("No bundled sample pair found; browse for audio and video files below.")

st.subheader("Browse files and folders from your computer")
st.write(
    "Choose a folder to upload **all file types** from it, including files inside "
    "its subfolders. The full relative path of each uploaded file is listed below. "
    "Then select one audio file and its matching video for the transcription pipeline."
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
MEDIA_EXTENSIONS = sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)

# IMPORTANT: type=None is intentional. A media-only type filter greys out files
# such as .py, .db, .txt, .json, .csv, .pdf, and model/config files in the picker.
folder_uploads = st.file_uploader(
    "Browse and select a folder (all file types)",
    type=None,
    accept_multiple_files="directory",
    help=(
        "Select a folder on your computer. All files in that folder and its "
        "subfolders will be uploaded, not only audio/video files."
    ),
    key="all_files_folder_upload",
)

folder_uploads = folder_uploads or []
audio_candidates = sorted(
    [f for f in folder_uploads if Path(f.name).suffix.lower() in AUDIO_EXTENSIONS],
    key=lambda f: f.name.lower(),
)
video_candidates = sorted(
    [f for f in folder_uploads if Path(f.name).suffix.lower() in VIDEO_EXTENSIONS],
    key=lambda f: f.name.lower(),
)

audio_upload = None
video_upload = None

if folder_uploads:
    total_mb = sum(f.size for f in folder_uploads) / (1024 * 1024)
    st.success(
        f"Folder uploaded: {len(folder_uploads)} file(s), {total_mb:.1f} MB total "
        f"({len(audio_candidates)} audio, {len(video_candidates)} video)."
    )
    with st.expander("View ALL uploaded files and their folder paths", expanded=True):
        file_rows = [
            {
                "Relative path": uploaded.name,
                "Extension": Path(uploaded.name).suffix.lower() or "(no extension)",
                "Size (MB)": round(uploaded.size / (1024 * 1024), 3),
            }
            for uploaded in sorted(folder_uploads, key=lambda f: f.name.lower())
        ]
        st.dataframe(file_rows, use_container_width=True, hide_index=True)
    st.caption(
        "Subfolders are represented by the relative paths. Empty folders are not "
        "uploaded by the browser because they contain no files."
    )

    selection_col1, selection_col2 = st.columns(2)
    with selection_col1:
        if audio_candidates:
            audio_index = st.selectbox(
                "Choose audio file",
                options=list(range(len(audio_candidates))),
                format_func=lambda i: audio_candidates[i].name,
                key="folder_audio_choice",
            )
            audio_upload = audio_candidates[audio_index]
        else:
            st.warning("No supported audio files were found in the selected folder.")

    with selection_col2:
        if video_candidates:
            video_index = st.selectbox(
                "Choose matching video file",
                options=list(range(len(video_candidates))),
                format_func=lambda i: video_candidates[i].name,
                key="folder_video_choice",
            )
            video_upload = video_candidates[video_index]
        else:
            st.warning("No supported video files were found in the selected folder.")

with st.expander("Or choose individual audio and video files"):
    manual_audio = st.file_uploader(
        "Browse for one audio file",
        type=sorted(AUDIO_EXTENSIONS),
        key="manual_audio_upload",
    )
    manual_video = st.file_uploader(
        "Browse for one video file",
        type=sorted(VIDEO_EXTENSIONS),
        key="manual_video_upload",
    )
    if manual_audio is not None:
        audio_upload = manual_audio
    if manual_video is not None:
        video_upload = manual_video

render_media_preview(audio_upload, video_upload)

webcam_media = st.session_state.get("webcam_media")
run_ready = ((audio_upload is not None and video_upload is not None) or use_sample_files or webcam_media is not None)
if st.button("Run complete pipeline", type="primary", disabled=not run_ready):
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = Path(tmp_dir)
        if webcam_media is not None:
            st.info("Running the pipeline with the live webcam capture as the source.")
            audio_path = webcam_media["audio"]
            video_path = webcam_media["video"]
        elif use_sample_files:
            audio_path = DEFAULT_SAMPLE_AUDIO
            video_path = DEFAULT_SAMPLE_VIDEO
        else:
            audio_path = temp_dir / ("input_audio" + Path(audio_upload.name).suffix.lower())
            if video_upload is not None:
                mp4_video = ensure_uploaded_video_mp4(video_upload)
                if mp4_video is not None:
                    video_path = mp4_video
                else:
                    video_path = temp_dir / ("input_video" + Path(video_upload.name).suffix.lower())
                    video_path.write_bytes(video_upload.getvalue())
            else:
                video_path = temp_dir / "input_video.mp4"
            audio_path.write_bytes(audio_upload.getvalue())

        try:
            with st.spinner(
                "Step 1/3: Running original AV LoRA, new 2k audio LoRA, and local fusion..."
            ):
                (
                    local_selected,
                    video_frames,
                    original_av_text,
                    new_2k_text,
                    fused_text,
                    selected_path,
                    fusion_error,
                ) = recognize_combined(audio_path, video_path)

            st.subheader("Extracted frames used by the model")
            st.caption(f"OpenCV extracted {len(video_frames)} grayscale frames for model input.")
            model_frame_cols = st.columns(4)
            for i, frame in enumerate(video_frames):
                with model_frame_cols[i % 4]:
                    st.image(
                        frame,
                        caption=f"Model frame {i + 1}",
                        clamp=True,
                        use_container_width=True,
                    )

            st.subheader("Local model predictions")
            st.markdown("**Original LoRA + AV adapter**")
            st.write(original_av_text or "No output from this model.")
            st.markdown("**New 2k audio LoRA**")
            st.write(new_2k_text or "No output from this model.")
            st.markdown("**Experimental token-score fusion**")
            st.write(fused_text or "No output from fusion.")
            st.caption(f"Selected local fallback/path: {selected_path}")
            if fusion_error:
                st.warning(f"Token-score fusion had an issue: {fusion_error}")

            selected_chinese = (local_selected or original_av_text or new_2k_text or fused_text or "").strip()
            if not selected_chinese:
                st.error("No usable Chinese transcript is available for translation.")
                st.stop()

            st.subheader("Chinese transcript used for translation")
            st.write(selected_chinese)
            with st.spinner("Step 2/3: Translating the selected Chinese transcript into English..."):
                english = translate_english(selected_chinese)
            st.markdown("## English translation")
            st.success(english)

            speech_path = temp_dir / "english_speech.mp3"
            with st.spinner("Step 3/3: Generating English speech..."):
                asyncio.run(generate_speech(english, speech_path))
            speech = speech_path.read_bytes()
            st.subheader("English speech")
            st.audio(speech, format="audio/mpeg")
            st.download_button(
                "Download English audio",
                data=speech,
                file_name="english_speech.mp3",
                mime="audio/mpeg",
            )
            st.success("Pipeline completed. Review transcript candidates if the models disagree.")
        except Exception as exc:
            st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
            st.exception(exc)


# Current Output Display (Persisted in Session)

# ==========================================

current = st.session_state.get("current_output")

if current:

    st.divider()

    col_out_title, col_clear = st.columns([5, 1])

    with col_out_title:

        st.subheader(f"Pipeline Output — Run #{current['id']}")

        st.caption(

            f"Generated at {current['timestamp']} | Audio: `{current['audio_name']}` | Video: `{current['video_name']}`"

        )

    with col_clear:

        if st.button("🗑️ Clear Output", key="btn_clear_current"):

            st.session_state["current_output"] = None

            st.rerun()



    st.markdown("#### Predicted Chinese text")

    st.info(current["chinese"])

    st.caption(f"OpenCV extracted {current['frame_count']} frames.")



    st.markdown("#### English translation")

    st.success(current["english"])



    st.markdown("#### English speech")

    st.audio(current["speech_bytes"], format="audio/mpeg")

    st.download_button(

        label="Download English audio",

        data=current["speech_bytes"],

        file_name=f"english_speech_run_{current['id']}.mp3",

        mime="audio/mpeg",

        key=f"dl_btn_current_{current['id']}",

    )

    st.success("All pipeline stages completed.")





# ==========================================

# Session Output History & Log

# ==========================================

history = st.session_state.get("history_log", [])

if history:

    st.divider()

    with st.expander(

        f"📜 Session Output Log ({len(history)} run{'s' if len(history) > 1 else ''})",

        expanded=False,

    ):

        col_log_info, col_log_clear = st.columns([4, 1])

        with col_log_info:

            st.write(

                "All pipeline outputs generated while this server is running are stored here:"

            )

        with col_log_clear:

            if st.button("Clear Log History", key="btn_clear_log"):

                st.session_state["history_log"] = []

                st.session_state["current_output"] = None

                st.rerun()



        for item in reversed(history):

            st.markdown(

                f"**Run #{item['id']}** ({item['timestamp']}) — Audio: `{item['audio_name']}` | Video: `{item['video_name']}`"

            )

            c1, c2 = st.columns(2)

            with c1:

                st.markdown("**Chinese:**")

                st.text_area(

                    "Chinese",

                    item["chinese"],

                    height=70,

                    key=f"log_cn_{item['id']}",

                    label_visibility="collapsed",

                )

            with c2:

                st.markdown("**English:**")

                st.text_area(

                    "English",

                    item["english"],

                    height=70,

                    key=f"log_en_{item['id']}",

                    label_visibility="collapsed",

                )

            col_aud, col_dl = st.columns([3, 1])

            with col_aud:

                st.audio(item["speech_bytes"], format="audio/mpeg")

            with col_dl:

                st.download_button(

                    label=f"Download Run #{item['id']} MP3",

                    data=item["speech_bytes"],

                    file_name=f"english_speech_run_{item['id']}.mp3",

                    mime="audio/mpeg",

                    key=f"dl_btn_log_{item['id']}",

                )

            st.markdown("---")
