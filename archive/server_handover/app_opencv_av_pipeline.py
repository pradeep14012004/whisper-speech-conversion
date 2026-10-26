
from pathlib import Path
import subprocess
import tempfile

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import imageio_ffmpeg

from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs/verified_av_manifest.csv"
ADAPTER = ROOT / "outputs/whisper_lora_small"
CHECKPOINT = ROOT / "outputs/av_transcription_full.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AVResidualAdapter(nn.Module):
    def __init__(self, audio_dim=768, hidden_dim=256, video_dim=128):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
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
        b, t, h, w = video.shape
        vf = self.video_encoder(video.reshape(b * t, 1, h, w))
        vf = self.video_proj(vf.reshape(b, t, -1))
        attended, _ = self.cross_attention(
            audio_q, vf, vf, need_weights=False
        )
        return audio_hidden + self.gate * self.out_proj(attended)


@st.cache_resource
def load_models():
    if not ADAPTER.exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {ADAPTER}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing AV checkpoint: {CHECKPOINT}")

    processor = WhisperProcessor.from_pretrained(str(ADAPTER))
    base = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    whisper = PeftModel.from_pretrained(base, str(ADAPTER))
    whisper = whisper.to(DEVICE).eval()

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    av = AVResidualAdapter(
        audio_dim=whisper.config.d_model
    ).to(DEVICE)
    av.load_state_dict(ckpt["av_adapter"])
    av.eval()
    return processor, whisper, av


def resolve_path(value):
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def extract_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        if count > 0:
            indices = np.linspace(
                0, count - 1, num_frames
            ).astype(int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
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
        raise RuntimeError("No video frames could be decoded.")
    frames.extend([frames[-1].copy()] * (num_frames - len(frames)))
    return np.stack(frames[:num_frames]).astype(np.float32)


def recognize(audio_path, video_path):
    processor, whisper, av = load_models()
    audio, sr = sf.read(str(audio_path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio)

    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    frames = extract_video(video_path)
    video = torch.from_numpy(frames).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        hidden = whisper.get_encoder()(
            input_features=features
        ).last_hidden_state
        fused = av(hidden, video)
        tokens = whisper.generate(
            input_features=features,
            encoder_outputs=BaseModelOutput(
                last_hidden_state=fused
            ),
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
        )

    text = processor.batch_decode(
        tokens, skip_special_tokens=True
    )[0].strip()
    return text, frames.shape


def run_script(args):
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )


st.set_page_config(
    page_title="Audio-Visual Chinese Recognition",
    layout="wide",
)
st.title("Audio-Visual Chinese Recognition → English")
st.caption(
    "OpenCV video frames → AV transcription → English translation → English TTS"
)
st.info(
    "The noisy Chinese speech-conversion stage is intentionally excluded."
)

if not MANIFEST.exists():
    st.error(f"Manifest not found: {MANIFEST}")
    st.stop()

df = pd.read_csv(MANIFEST).fillna("")
required = {
    "whisper_audio", "whisper_video", "transcript",
    "split", "speaker_id", "whisper_id",
}
if not required.issubset(df.columns):
    st.error(f"Manifest missing columns: {required - set(df.columns)}")
    st.stop()
if df.empty:
    st.error("The dataset manifest is empty.")
    st.stop()

labels = [
    f"{r['speaker_id']} | {r['whisper_id']} | "
    f"{str(r['transcript'])[:30]}"
    for _, r in df.iterrows()
]
selected = st.selectbox(
    "Select a dataset sample",
    range(len(df)),
    format_func=lambda i: labels[i],
)
row = df.iloc[selected]
audio_path = resolve_path(row["whisper_audio"])
video_path = resolve_path(row["whisper_video"])

left, right = st.columns(2)
with left:
    st.subheader("Original video")
    if video_path.is_file():
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            with tempfile.TemporaryDirectory() as tmp:
                mp4 = Path(tmp) / "preview.mp4"
                subprocess.run(
                    [
                        ffmpeg, "-y", "-i", str(video_path),
                        "-an", "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-movflags", "+faststart",
                        str(mp4),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                st.video(mp4.read_bytes(), format="video/mp4")
        except Exception as e:
            st.warning(f"Video preview failed: {e}")
    else:
        st.error(f"Video not found: {video_path}")

with right:
    st.subheader("Whispered audio")
    if audio_path.is_file():
        st.audio(audio_path.read_bytes(), format="audio/wav")
    else:
        st.error(f"Audio not found: {audio_path}")

st.caption(f"Speaker: {row['speaker_id']} · Split: {row['split']}")

if st.button(
    "Recognize Chinese, translate, and generate English speech",
    type="primary",
):
    if not audio_path.is_file() or not video_path.is_file():
        st.error("Selected audio or video file is missing.")
        st.stop()

    try:
        with st.spinner("Loading models and recognizing Chinese..."):
            chinese, frame_shape = recognize(audio_path, video_path)
        st.session_state["av_text"] = chinese
        st.session_state["av_shape"] = frame_shape
        st.session_state["sample_id"] = str(row["whisper_id"])
    except Exception as e:
        st.error(f"Recognition failed: {type(e).__name__}: {e}")
        st.exception(e)
        st.stop()

    st.write("OpenCV frame tensor:", frame_shape)
    st.write("**Predicted Chinese text:**", chinese or "(empty)")
    if not chinese:
        st.error("No text was predicted; translation was skipped.")
        st.stop()

    with st.spinner("Translating Chinese to English..."):
        tr = run_script(["python", "translate_one.py", chinese])
    if tr.returncode != 0:
        st.error("Translation script failed.")
        st.code((tr.stdout + "\n" + tr.stderr)[-5000:])
        st.stop()

    english_file = ROOT / "outputs/demo_english.txt"
    if not english_file.exists():
        st.error("Translation output file was not created.")
        st.stop()
    english = english_file.read_text(encoding="utf-8").strip()
    if not english:
        st.error("Translation output is empty.")
        st.stop()

    with st.spinner("Generating English speech..."):
        tts = run_script(["python", "english_tts_one.py", english])
    if tts.returncode != 0:
        st.error("English TTS failed.")
        st.code((tts.stdout + "\n" + tts.stderr)[-5000:])
        st.stop()

    st.session_state["english_text"] = english
    st.session_state["completed_sample"] = str(row["whisper_id"])
    st.success("All three pipeline stages completed.")

if st.session_state.get("completed_sample") == str(row["whisper_id"]):
    st.subheader("Pipeline results")
    st.markdown("**Predicted Chinese text**")
    st.write(st.session_state.get("av_text", ""))
    st.markdown("**English translation**")
    st.write(st.session_state.get("english_text", ""))

    speech = ROOT / "outputs/demo_english.mp3"
    if speech.is_file():
        st.markdown("**Generated English speech**")
        st.audio(speech.read_bytes(), format="audio/mpeg")
        st.download_button(
            "Download English speech",
            speech.read_bytes(),
            file_name="english_speech.mp3",
            mime="audio/mpeg",
        )
