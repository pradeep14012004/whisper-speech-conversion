import csv, os
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import cer

MODEL = "openai/whisper-large-v3"
N = 100

with open("outputs/valid_pairs.csv", encoding="utf-8-sig") as f:
    rows = [
        r for r in csv.DictReader(f)
        if r["transcript"] and r["whisper_exists"] == "True"
        and r["normal_exists"] == "True"
    ][:N]

processor = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()

def transcribe(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    inputs = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to("cuda", dtype=torch.float16)
    with torch.inference_mode():
        ids = model.generate(
            inputs, language="chinese", task="transcribe",
            max_new_tokens=128
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

results = []
for i, row in enumerate(rows, 1):
    try:
        wp = transcribe(row["whisper_audio"])
        np = transcribe(row["normal_audio"])
        results.append({
            "reference": row["transcript"],
            "whisper_prediction": wp,
            "normal_prediction": np,
            "whisper_CER": cer(row["transcript"], wp),
            "normal_CER": cer(row["transcript"], np)
        })
        if i % 10 == 0:
            print(f"Processed {i}/{len(rows)}", flush=True)
    except Exception as e:
        print(f"Skipped sample {i}: {e}", flush=True)

os.makedirs("outputs", exist_ok=True)
with open("outputs/baseline_100.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Successful pairs:", len(results))
print("Whispered-audio mean CER:",
      sum(r["whisper_CER"] for r in results)/len(results))
print("Normal-audio mean CER:",
      sum(r["normal_CER"] for r in results)/len(results))
print("Saved outputs/baseline_100.csv")
