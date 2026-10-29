from pathlib import Path
import csv
import unicodedata
import gc

import editdistance
import librosa
import numpy as np
import soundfile as sf
import torch
from opencc import OpenCC
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_samples_5pairs"
ADAPTER = ROOT / "outputs/whisper_lora_aishell6_2k"
MANIFEST = DEMO / "manifest.csv"

if not ADAPTER.is_dir() or not MANIFEST.is_file():
    raise SystemExit("Missing adapter or demo manifest.")

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

rows = [
    r for r in rows
    if r.get("sample") in {"1", "2", "3", "4", "5"}
    and r.get("reference_transcript", "").strip()
]

if len(rows) != 5:
    raise SystemExit(f"Expected 5 samples with references; found {len(rows)}.")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
cc = OpenCC("t2s")

def normalize(text):
    text = cc.convert(unicodedata.normalize("NFKC", text).lower())
    return "".join(ch for ch in text if ch.isalnum())

print("Device:", device)
processor = WhisperProcessor.from_pretrained(str(ADAPTER))
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, str(ADAPTER)).to(device).eval()

results = []

for row in rows:
    audio_path = DEMO / row["audio_file"]
    audio, sr = sf.read(str(audio_path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    ref = row["reference_transcript"]
    record = {"sample": row["sample"], "reference": ref}

    for beams in (1, 5):
        with torch.inference_mode():
            tokens = model.generate(
                input_features=features,
                language="chinese",
                task="transcribe",
                max_new_tokens=128,
                num_beams=beams,
            )

        prediction = processor.batch_decode(
            tokens, skip_special_tokens=True
        )[0].strip()

        cer = editdistance.eval(
            normalize(ref), normalize(prediction)
        ) / max(1, len(normalize(ref)))

        record[f"prediction_beams_{beams}"] = prediction
        record[f"cer_beams_{beams}"] = cer

        print(f"\nSample {row['sample']} | beams={beams}")
        print("Prediction:", prediction)
        print(f"CER: {cer * 100:.2f}%")

    results.append(record)
    del features, tokens

print("\n===== SUMMARY =====")
for beams in (1, 5):
    edits = 0
    chars = 0
    exact = 0

    for r in results:
        ref = normalize(r["reference"])
        pred = normalize(r[f"prediction_beams_{beams}"])
        edits += editdistance.eval(ref, pred)
        chars += len(ref)
        exact += int(ref == pred)

    print(
        f"Beams {beams}: corpus CER={100 * edits / max(1, chars):.2f}% "
        f"| exact matches={exact}/5"
    )

out = ROOT / "outputs/evaluation/beam_search_demo_comparison.csv"
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("Saved:", out)

del model, base, processor
gc.collect()
if device == "mps":
    torch.mps.empty_cache()
