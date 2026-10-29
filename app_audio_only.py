import asyncio
import tempfile
from pathlib import Path

import edge_tts
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from opencc import OpenCC
from peft import PeftModel
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

ROOT = Path(__file__).resolve().parent
LORA_DIR = ROOT / "outputs" / "whisper_lora_aishell6_2k"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@st.cache_resource
def load_asr():
    if not LORA_DIR.is_dir():
        raise FileNotFoundError(f"Missing fine-tuned adapter: {LORA_DIR}")

    processor = WhisperProcessor.from_pretrained(str(LORA_DIR))
    base = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    )
    model = PeftModel.from_pretrained(base, str(LORA_DIR))
    model = model.to(DEVICE).eval()

    return processor, model


@st.cache_resource
def load_translation():
    name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name, use_safetensors=False
    )
    model = model.to(DEVICE).eval()
    return tokenizer, model


def transcribe_chinese(audio_path):
    processor, model = load_asr()

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError("The uploaded audio is empty.")

    if len(audio) / sample_rate > 120:
        raise ValueError("Please upload an audio clip of 120 seconds or less.")

    if sample_rate != 16000:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=16000
        ).astype(np.float32)

    features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(DEVICE)

    with torch.inference_mode():
        tokens = model.generate(
            input_features=features,
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
            num_beams=5,
        )

    return processor.batch_decode(
        tokens, skip_special_tokens=True
    )[0].strip()


def translate_english(chinese_text):
    tokenizer, model = load_translation()

    # Normalize any Traditional Chinese output before translation.
    chinese_text = OpenCC("t2s").convert(chinese_text)

    inputs = tokenizer(
        chinese_text,
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

    return tokenizer.batch_decode(
        output, skip_special_tokens=True
    )[0].strip()


async def generate_speech(text, output_path):
    tts = edge_tts.Communicate(
        text, voice="en-US-AriaNeural"
    )
    await tts.save(str(output_path))


st.set_page_config(
    page_title="Fine-tuned Mandarin Transcription",
    layout="wide",
)

st.title("Mandarin Whisper Transcription → English Speech")

st.write(
    "This version uses the AISHELL6-Whisper fine-tuned Whisper-small "
    "LoRA adapter for Chinese transcription, OPUS for English translation, "
    "and Edge TTS for speech generation."
)

st.info(
    "Audio-only version: it does not use video or the existing AV adapter. "
    "This lets us evaluate the fine-tuned audio model independently."
)

st.caption(f"Inference device: {DEVICE}")
st.caption(f"Whisper adapter: {LORA_DIR.relative_to(ROOT)}")

audio_upload = st.file_uploader(
    "Upload whispered Mandarin audio",
    type=["wav", "flac", "ogg"],
)

reference = st.text_input(
    "Optional: paste the correct Chinese transcript to measure CER"
)

if st.button(
    "Run audio-only pipeline",
    type="primary",
    disabled=audio_upload is None,
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio_path = tmp / "input_audio"  # WAV/FLAC/OGG content is preserved below
        suffix = Path(audio_upload.name).suffix.lower()
        audio_path = tmp / ("input_audio" + suffix)
        audio_path.write_bytes(audio_upload.getvalue())

        try:
            with st.spinner("Step 1/3: Transcribing Chinese..."):
                chinese = transcribe_chinese(audio_path)

            if not chinese:
                st.error("The transcription model returned empty text.")
                st.stop()

            st.subheader("Predicted Chinese transcription")
            st.write(chinese)

            if reference.strip():
                import editdistance
                import unicodedata

                converter = OpenCC("t2s")

                def normalize(text):
                    text = converter.convert(
                        unicodedata.normalize("NFKC", text).lower()
                    )
                    return "".join(c for c in text if c.isalnum())

                ref = normalize(reference)
                pred = normalize(chinese)

                if ref:
                    cer = editdistance.eval(ref, pred) / len(ref)
                    st.metric("Character Error Rate (CER)", f"{cer * 100:.2f}%")
                    st.caption("Lower CER is better. This comparison uses your supplied reference.")

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
            st.subheader("Generated English speech")
            st.audio(speech, format="audio/mpeg")

            st.download_button(
                "Download English speech",
                data=speech,
                file_name="english_speech.mp3",
                mime="audio/mpeg",
            )

            st.success("All three pipeline stages completed.")

        except Exception as exc:
            st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
