import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MANIFEST = "outputs/manifest_all_16k.csv"
OUT = "outputs/evaluation/validation_beam_comparison.csv"
MODEL = "openai/whisper-large-v3"
N = 16

df = pd.read_csv(MANIFEST)
df = df[(df["split"] == "valid") & df["usable_audio"]]
df = df.head(N).reset_index(drop=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Device={device}; clips={len(df)}", flush=True)

processor = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)
model.eval()

rows = []
for i, r in df.iterrows():
    audio, sr = sf.read(r["whisper_16k"], dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}: {r['whisper_16k']}")

    feats = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype=dtype)

    with torch.inference_mode():
        base_ids = model.generate(
            feats, language="chinese", task="transcribe",
            max_new_tokens=128, num_beams=1
        )
        repeat_ids = model.generate(
            feats, language="chinese", task="transcribe",
            max_new_tokens=64, num_beams=2
        )

    base = processor.batch_decode(base_ids, skip_special_tokens=True)[0]
    repeat = processor.batch_decode(repeat_ids, skip_special_tokens=True)[0]
    del feats, base_ids, repeat_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows.append({
        "audio": r["whisper_16k"],
        "reference": str(r["transcript"]),
        "greedy": base,
        "beam_4": repeat,
    })

    if (i + 1) % 8 == 0 or i + 1 == len(df):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{i+1}/{len(df)} clips", flush=True)

def norm(s):
    return "".join(c for c in str(s) if c.isalnum())

for col in ["greedy", "beam_4"]:
    refs = [norm(x) for x in pd.DataFrame(rows)["reference"]]
    hyps = [norm(x) for x in pd.DataFrame(rows)[col]]
    print(f"{col} CER: {100 * jiwer.cer(refs, hyps):.3f}%", flush=True)

print("Saved:", OUT, flush=True)
