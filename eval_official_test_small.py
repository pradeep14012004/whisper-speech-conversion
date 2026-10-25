import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MANIFEST = "outputs/evaluation/official_test_manifest.csv"
OUT = "outputs/evaluation/official_test_small_predictions.csv"
MODEL_NAME = "openai/whisper-small"
BATCH = 1
SAVE_EVERY = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

df = pd.read_csv(MANIFEST).dropna(subset=["whisper_audio", "transcript"]).reset_index(drop=True)
for p in df["whisper_audio"]:
    if not os.path.isfile(p):
        raise FileNotFoundError(p)

rows = []
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, keep_default_na=False)
    if len(prev) > len(df) or (
        len(prev) and not prev["whisper_audio"].equals(df.iloc[:len(prev)]["whisper_audio"])
    ):
        raise RuntimeError("Existing predictions don't match this manifest. Rename the output file.")
    rows = prev.to_dict("records")

print(f"Device={device}; clips={len(df)}; resuming at {len(rows)}", flush=True)

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)
model.eval()

def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio.astype(np.float32))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio

def save():
    pd.DataFrame(rows).to_csv(OUT, index=False)

try:
    for i in range(len(rows), len(df)):
        r = df.iloc[i]
        audio = load_audio(r["whisper_audio"])
        feats = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)

        with torch.inference_mode():
            ids = model.generate(
                feats, language="chinese", task="transcribe", max_new_tokens=225
            )
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
        rows.append({
            "whisper_audio": r["whisper_audio"],
            "reference": str(r["transcript"]),
            "prediction": pred
        })

        if len(rows) % SAVE_EVERY == 0 or len(rows) == len(df):
            save()
        if len(rows) % 25 == 0 or len(rows) == len(df):
            print(f"{len(rows)}/{len(df)}", flush=True)

except KeyboardInterrupt:
    save()
    print("Interrupted; saved progress.", flush=True)
    raise
except Exception:
    save()
    raise

save()
print("Done:", OUT, flush=True)
