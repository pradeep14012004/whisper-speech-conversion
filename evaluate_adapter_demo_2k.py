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
from transformers import WhisperProcessor, WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_samples_5pairs"
MANIFEST = DEMO / "manifest.csv"

ADAPTERS = {
    "original": ROOT / "outputs/whisper_lora_small",
    "fine_tuned": ROOT / "outputs/whisper_lora_aishell6_2k",
}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
converter = OpenCC("t2s")


def normalize(text):
    text = converter.convert(unicodedata.normalize("NFKC", text).lower())
    return "".join(ch for ch in text if ch.isalnum())


def cer(reference, prediction):
    ref = normalize(reference)
    pred = normalize(prediction)
    if not ref:
        return float("nan")
    return editdistance.eval(ref, pred) / len(ref)


if not MANIFEST.is_file():
    raise SystemExit(f"Missing manifest: {MANIFEST}")

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

rows = [
    r for r in rows
    if (r.get("reference_transcript") or "").strip()
    and (DEMO / r.get("audio_file", "")).is_file()
]

if not rows:
    raise SystemExit("No usable demo audio and reference transcripts found.")

for name, path in ADAPTERS.items():
    if not path.is_dir():
        raise SystemExit(f"Missing {name} adapter: {path}")

print("Device:", DEVICE)
print("Evaluation samples:", len(rows))

all_results = []

def evaluate_adapter(label, adapter_dir):
    print(f"\nEvaluating {label} adapter...")
    processor = WhisperProcessor.from_pretrained(str(adapter_dir))
    base = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.to(DEVICE).eval()

    results = []
    for i, row in enumerate(rows, start=1):
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
        ).input_features.to(DEVICE)

        with torch.inference_mode():
            tokens = model.generate(
                input_features=features,
                language="chinese",
                task="transcribe",
                max_new_tokens=128,
                num_beams=1,
            )

        prediction = processor.batch_decode(
            tokens, skip_special_tokens=True
        )[0].strip()

        reference = row["reference_transcript"]
        score = cer(reference, prediction)
        results.append({
            "sample": row.get("sample", str(i)),
            "reference": reference,
            "prediction": prediction,
            "cer": score,
        })
        print(
            f"  Sample {row.get('sample', i)}: "
            f"CER {score * 100:.2f}% | {prediction}"
        )

    del model, base, processor
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return results


for label, adapter_dir in ADAPTERS.items():
    results = evaluate_adapter(label, adapter_dir)
    all_results.extend({**r, "adapter": label} for r in results)

print("\n=== SUMMARY ===")
for label in ADAPTERS:
    subset = [r for r in all_results if r["adapter"] == label]
    total_edits = 0
    total_chars = 0
    exact = 0

    for r in subset:
        ref = normalize(r["reference"])
        pred = normalize(r["prediction"])
        total_edits += editdistance.eval(ref, pred)
        total_chars += len(ref)
        exact += int(ref == pred)

    score = 100 * total_edits / max(1, total_chars)
    print(
        f"{label}: corpus CER {score:.2f}% | "
        f"exact matches {exact}/{len(subset)}"
    )

out = ROOT / "outputs/evaluation"
out.mkdir(parents=True, exist_ok=True)
report = out / "aishell6_demo_adapter_2k_comparison.csv"

with report.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=["adapter", "sample", "reference", "prediction", "cer"]
    )
    writer.writeheader()
    writer.writerows(all_results)

print("\nDetailed comparison saved to:", report)
print("No model files were changed.")
