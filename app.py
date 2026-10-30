import asyncio

import os

import re

import tempfile

import threading

from datetime import datetime

from pathlib import Path



import cv2

from opencc import OpenCC

import edge_tts

import librosa

import numpy as np

import soundfile as sf

import streamlit as st

import torch

import torch.nn as nn

from peft import PeftModel

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

    page_title="Audio-Visual Chinese-to-English Pipeline",

    layout="wide",

)



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

st.subheader("Choose files from your computer")
with st.container():
    audio_upload = st.file_uploader(
        "Browse and upload audio",
        type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
        help="Choose an audio file from your computer. The browser opens a file picker.",
        key="audio_upload",
    )
    video_upload = st.file_uploader(
        "Browse and upload video",
        type=["avi", "mp4", "mov", "mkv", "webm", "mpeg", "mpg"],
        help="Choose a video file from your computer. The browser opens a file picker.",
        key="video_upload",
    )

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
