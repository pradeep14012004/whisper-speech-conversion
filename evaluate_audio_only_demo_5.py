from pathlib import Path
import csv
import gc
import unicodedata

import editdistance
import librosa
import numpy as np
import soundfile as sf
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
DEMO = ROOT / "demo_samples_5pairs"
MANIFEST = DEMO / "manifest.csv"
ADAPTER = ROOT / "outputs/whisper_lora_aishell6_2k"
OUTPUT = ROOT / "outputs/evaluation/audio_only_demo_5_results.csv"

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
if not ADAPTER.is_dir():
    raise FileNotFoundError(f"Missing adapter: {ADAPTER}")

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    manifest_rows = list(csv.DictReader(f))

rows = [
    r for r in manifest_rows
    if r.get("sample") in {"1", "2", "3", "4", "5"}
]

if len(rows) != 5:
    raise RuntimeError(f"Expected 5 manifest samples, found {len(rows)}")

for row in rows:
    if not (DEMO / row["audio_file"]).is_file():
        raise FileNotFoundError(DEMO / row["audio_file"])
    if not row.get("reference_transcript", "").strip():
        raise ValueError(f"Missing reference for sample {row['sample']}")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
converter = OpenCC("t2s")

def normalize(text):
    text = converter.convert(unicodedata.normalize("NFKC", text).lower())
    return "".join(c for c in text if c.isalnum())

def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio):
        raise ValueError(f"Empty audio file: {path}")
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)
    return audio

print("Device:", DEVICE)
print("Loading fine-tuned Whisper + LoRA...")
processor = WhisperProcessor.from_pretrained(str(ADAPTER))
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
)
asr = PeftModel.from_pretrained(base, str(ADAPTER)).to(DEVICE).eval()

print("Loading OPUS Chinese-to-English translator...")
translation_name = "Helsinki-NLP/opus-mt-zh-en"
translator_tokenizer = AutoTokenizer.from_pretrained(translation_name)
translator = AutoModelForSeq2SeqLM.from_pretrained(
    translation_name, use_safetensors=False
).to(DEVICE).eval()

results = []
total_edits = 0
total_reference_chars = 0
exact_matches = 0

for row in rows:
    sample_id = row["sample"]
    audio_path = DEMO / row["audio_file"]
    reference = row["reference_transcript"]

    audio = load_audio(audio_path)
    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.inference_mode():
        tokens = asr.generate(
            input_features=features,
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
            num_beams=1,
        )
    prediction = processor.batch_decode(
        tokens, skip_special_tokens=True
    )[0].strip()

    ref_norm = normalize(reference)
    pred_norm = normalize(prediction)
    edits = editdistance.eval(ref_norm, pred_norm)
    cer = edits / max(1, len(ref_norm))

    total_edits += edits
    total_reference_chars += len(ref_norm)
    exact_matches += int(ref_norm == pred_norm)

    # Translate the model prediction, not the reference.
    source_text = converter.convert(prediction)
    inputs = translator_tokenizer(
        source_text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    with torch.inference_mode():
        translated_tokens = translator.generate(
            **inputs, max_new_tokens=128, num_beams=4
        )
    english = translator_tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )[0].strip()

    results.append({
        "sample": sample_id,
        "audio_file": row["audio_file"],
        "reference_chinese": reference,
        "predicted_chinese": prediction,
        "cer_percent": round(cer * 100, 2),
        "english_translation": english,
    })

    print(f"\n--- Sample {sample_id} ---")
    print("Reference:", reference)
    print("Prediction:", prediction)
    print(f"CER: {cer * 100:.2f}%")
    print("English:", english)

    del audio, features, tokens, inputs, translated_tokens

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\n========== SUMMARY ==========")
print("Samples evaluated:", len(results))
print(
    "Corpus CER:",
    f"{100 * total_edits / max(1, total_reference_chars):.2f}%"
)
print(
    "Exact matches:",
    f"{exact_matches}/{len(results)}"
)
print("Results saved to:", OUTPUT)

del asr, base, processor, translator, translator_tokenizer
gc.collect()
if DEVICE == "mps":
    torch.mps.empty_cache()
