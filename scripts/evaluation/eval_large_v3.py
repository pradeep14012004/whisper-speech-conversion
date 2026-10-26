import os, sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL = "openai/whisper-large-v3"
MANIFEST = "outputs/valid_pairs.csv"
OUT = "outputs/evaluation/large_v3_predictions.csv"
BATCH = 16
SAVE_EVERY = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

# Same clip selection as evaluate_comparison.py
df = pd.read_csv(MANIFEST)
df = df[
    df["whisper_audio"].notna()
    & df["transcript"].notna()
    & df["whisper_exists"].astype(bool)
    & df["transcript_exists"].astype(bool)
].copy()
train_df = pd.read_csv("outputs/training_manifest.csv")
pilot = train_df.sample(n=min(32, len(train_df)), random_state=42)
pilot_paths = set(pilot["whisper_audio"].dropna())
df = df[~df["whisper_audio"].isin(pilot_paths)].reset_index(drop=True)
print("Clips selected:", len(df), flush=True)
if limit:
    df = df.iloc[:limit]
    print("Limited to first", limit, flush=True)

os.makedirs("outputs/evaluation", exist_ok=True)
rows = []
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, keep_default_na=False)
    rows = prev.to_dict("records")
    for k, r in enumerate(rows):
        assert r["whisper_audio"] == df.iloc[k]["whisper_audio"], "saved file does not match clip order"
    print("Resuming from clip", len(rows), flush=True)

processor = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True
).to(DEVICE)
model.eval()

def load(path):
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio.astype(np.float32))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio

def save():
    pd.DataFrame(rows).to_csv(OUT, index=False)

n_batches = 0
for b in range(len(rows), len(df), BATCH):
    chunk = df.iloc[b:b + BATCH]
    audios = [load(p) for p in chunk["whisper_audio"]]
    feats = processor(
        audios, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE, dtype=DTYPE)
    with torch.inference_mode():
        ids = model.generate(
            feats, language="chinese", task="transcribe", max_new_tokens=225
        )
    texts = processor.batch_decode(ids, skip_special_tokens=True)
    for (_, r), t in zip(chunk.iterrows(), texts):
        rows.append({
            "whisper_audio": r["whisper_audio"],
            "reference": str(r["transcript"]),
            "prediction": t,
        })
    n_batches += 1
    print(f"{len(rows)}/{len(df)}", flush=True)
    if n_batches % SAVE_EVERY == 0:
        save()

save()
print("Done. Saved to", OUT, flush=True)
