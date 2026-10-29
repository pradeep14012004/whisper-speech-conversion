import asyncio
import tempfile
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
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForMaskedLM,
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

ROOT = Path(__file__).resolve().parent
OLD_LORA_DIR = ROOT / "outputs" / "whisper_lora_small"
NEW_LORA_DIR = ROOT / "outputs" / "whisper_lora_aishell6_2k"
OLD_AV_CHECKPOINT = ROOT / "outputs" / "av_transcription_full.pt"
MACBERT_MODEL_NAME = "shibing624/macbert4csc-base-chinese"
MACBERT_FINETUNED_DIR = ROOT / "outputs" / "macbert_asr_corrector"
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


@st.cache_resource(show_spinner="Loading MacBERT Chinese spelling corrector...")
def load_macbert_corrector():
    """Prefer the ASR-error-fine-tuned corrector when training has produced it."""
    has_finetuned_weights = (
        (MACBERT_FINETUNED_DIR / "config.json").is_file()
        and any(
            (MACBERT_FINETUNED_DIR / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )
    )
    model_source = str(MACBERT_FINETUNED_DIR) if has_finetuned_weights else MACBERT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForMaskedLM.from_pretrained(model_source)
    model = model.to(DEVICE).eval()
    return tokenizer, model, model_source


def correct_chinese_text(text):
    """Apply MacBERT CSC correction and return (corrected_text, changes).

    The released checkpoint is a general Chinese spelling-correction model,
    not one trained specifically for Whisper errors. We preserve the original
    text separately in the UI so any overcorrection is visible.
    """
    if not text or not text.strip():
        return text, []

    # The released MacBERT CSC checkpoint is trained for Simplified Chinese.
    source_text = OpenCC("t2s").convert(text.strip())
    tokenizer, model, _model_source = load_macbert_corrector()
    encoded = tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.inference_mode():
        logits = model(**encoded).logits
        predicted_ids = logits.argmax(dim=-1)[0]

    # MacBERT CSC predicts a replacement token at each original token position.
    # Decode its predicted token sequence, then follow the model card's
    # character-length alignment convention for Chinese text.
    decoded = tokenizer.decode(
        predicted_ids, skip_special_tokens=True
    ).replace(" ", "").strip()
    corrected = decoded[:len(source_text)]

    # Safety guard: do not let the corrector return an empty sentence or
    # silently drop source characters if decoding yields a shorter string.
    if not corrected:
        corrected = source_text
    elif len(corrected) < len(source_text):
        corrected = corrected + source_text[len(corrected):]

    # Record character substitutions for transparency.
    changes = []
    max_len = min(len(source_text), len(corrected))
    for idx in range(max_len):
        if source_text[idx] != corrected[idx]:
            changes.append((idx, source_text[idx], corrected[idx]))

    return corrected, changes


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


st.set_page_config(
    page_title="Audio-Visual Speech Pipeline",
    layout="wide",
)
st.title("Audio-Visual Chinese-to-English Pipeline")
st.write(
    "Upload matching whispered Chinese audio and video. The app runs the "
    "original LoRA + AV adapter and the new 2k audio LoRA, fuses their "
    "token scores experimentally, translates the selected Chinese text, "
    "and generates English speech."
)
st.caption(f"Inference device: {DEVICE}. Maximum audio duration: {MAX_AUDIO_SECONDS} seconds.")

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
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = Path(tmp_dir)
        audio_path = temp_dir / (
            "input_audio" + Path(audio_upload.name).suffix.lower()
        )
        video_path = temp_dir / (
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
            st.markdown("## English translation")
            st.success(english)

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


# ==========================================
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
