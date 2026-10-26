
import asyncio
import tempfile
from pathlib import Path

import cv2
import edge_tts
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parent
LORA_DIR = ROOT / "outputs" / "whisper_lora_small"
AV_CHECKPOINT = ROOT / "outputs" / "av_transcription_full.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


@st.cache_resource
def load_asr():
    if not LORA_DIR.exists():
        raise FileNotFoundError(f"Missing Whisper LoRA adapter: {LORA_DIR}")
    if not AV_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing AV checkpoint: {AV_CHECKPOINT}")

    processor = WhisperProcessor.from_pretrained(str(LORA_DIR))
    base = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    )
    whisper = PeftModel.from_pretrained(base, str(LORA_DIR))
    whisper = whisper.to(DEVICE).eval()

    checkpoint = torch.load(AV_CHECKPOINT, map_location="cpu")
    adapter = AVResidualAdapter(audio_dim=whisper.config.d_model)
    adapter.load_state_dict(checkpoint["av_adapter"])
    adapter = adapter.to(DEVICE).eval()
    return processor, whisper, adapter


@st.cache_resource
def load_translation():
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(DEVICE).eval()
    tokenizer.src_lang = "zho_Hans"
    return tokenizer, model


def extract_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("OpenCV cannot open the supplied video.")

    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            indices = np.linspace(0, total - 1, num_frames).astype(int)
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
        raise ValueError("No video frames could be extracted.")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:num_frames])


def recognize_chinese(audio_path, video_path):
    processor, whisper, adapter = load_asr()
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("The audio file is empty.")
    if len(audio) / sr > 120:
        raise ValueError("Please use an audio clip of 120 seconds or less.")
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    frames = extract_video_frames(video_path)
    video = torch.tensor(
        frames, dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)

    with torch.inference_mode():
        encoder_hidden = whisper.get_encoder()(
            input_features=features
        ).last_hidden_state
        fused_hidden = adapter(encoder_hidden, video)
        tokens = whisper.generate(
            input_features=features,
            encoder_outputs=BaseModelOutput(
                last_hidden_state=fused_hidden
            ),
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
        )

    text = processor.batch_decode(
        tokens, skip_special_tokens=True
    )[0].strip()
    return text, len(frames)


def translate_english(chinese_text):
    tokenizer, model = load_translation()
    tokenizer.src_lang = "zho_Hans"
    inputs = tokenizer(
        chinese_text, return_tensors="pt",
        truncation=True, max_length=512
    ).to(DEVICE)

    english_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            forced_bos_token_id=english_id,
            max_new_tokens=256,
            num_beams=4,
        )
    return tokenizer.batch_decode(
        output, skip_special_tokens=True
    )[0].strip()


async def generate_speech(text, output_path):
    tts = edge_tts.Communicate(
        text, voice="en-US-AriaNeural"
    )
    await tts.save(str(output_path))


st.set_page_config(
    page_title="Audio-Visual Speech Pipeline",
    layout="wide",
)
st.title("Audio-Visual Chinese-to-English Pipeline")
st.write(
    "Upload matching whispered Chinese audio and video. "
    "The system predicts Chinese text, translates it into English, "
    "and generates English speech."
)
st.caption("No Chinese speech-conversion stage is used.")

audio_upload = st.file_uploader(
    "Upload whispered audio",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
)
video_upload = st.file_uploader(
    "Upload matching video",
    type=["avi", "mp4", "mov", "mkv", "webm"],
)

if st.button(
    "Run complete pipeline",
    type="primary",
    disabled=not (audio_upload and video_upload),
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio_path = tmp / (
            "input_audio" + Path(audio_upload.name).suffix.lower()
        )
        video_path = tmp / (
            "input_video" + Path(video_upload.name).suffix.lower()
        )
        audio_path.write_bytes(audio_upload.getvalue())
        video_path.write_bytes(video_upload.getvalue())

        try:
            with st.spinner("Step 1/3: Predicting Chinese text..."):
                chinese, frame_count = recognize_chinese(
                    audio_path, video_path
                )
            if not chinese:
                st.error("The model returned no Chinese text.")
                st.stop()

            st.subheader("Predicted Chinese text")
            st.write(chinese)
            st.caption(f"OpenCV extracted {frame_count} frames.")

            with st.spinner("Step 2/3: Translating into English..."):
                english = translate_english(chinese)
            if not english:
                st.error("Translation returned empty text.")
                st.stop()

            st.subheader("English translation")
            st.write(english)

            speech_path = tmp / "english_speech.mp3"
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
            st.success("All pipeline stages completed.")
        except Exception as exc:
            st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
