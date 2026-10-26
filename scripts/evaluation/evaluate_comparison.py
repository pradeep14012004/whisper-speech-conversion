import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import cer, wer

BASE = "openai/whisper-small"
ADAPTER = "outputs/whisper_lora_small"
MANIFEST = "outputs/valid_pairs.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 10000

df = pd.read_csv(MANIFEST)
df = df[
    df["whisper_audio"].notna()
    & df["transcript"].notna()
    & df["whisper_exists"].astype(bool)
    & df["transcript_exists"].astype(bool)
].copy()

# Deterministic sample for a repeatable comparison
train_df = pd.read_csv("outputs/training_manifest.csv")
pilot = train_df.sample(n=min(32, len(train_df)), random_state=42)
pilot_paths = set(pilot["whisper_audio"].dropna())

df = df[~df["whisper_audio"].isin(pilot_paths)].reset_index(drop=True)
print("Samples selected for evaluation:", len(df))

processor = WhisperProcessor.from_pretrained(ADAPTER)

base = WhisperForConditionalGeneration.from_pretrained(BASE).to(DEVICE)
base.eval()

adapted_base = WhisperForConditionalGeneration.from_pretrained(BASE)
adapted = PeftModel.from_pretrained(adapted_base, ADAPTER).to(DEVICE)
adapted.eval()

refs, base_preds, lora_preds = [], [], []

for i, row in df.iterrows():
    path = row["whisper_audio"]
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio.astype(np.float32))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    preds = []
    for model in (base, adapted):
        with torch.inference_mode():
            ids = model.generate(
                features,
                language="chinese",
                task="transcribe",
                max_new_tokens=128
            )
        preds.append(processor.batch_decode(
            ids, skip_special_tokens=True
        )[0])

    refs.append(str(row["transcript"]))
    base_preds.append(preds[0])
    lora_preds.append(preds[1])

    print(f"{i+1}/{len(df)} evaluated")

print("\n=== RESULTS ===")
print("Samples:", len(refs))
print(f"Base CER: {cer(refs, base_preds):.4f}")
print(f"LoRA CER: {cer(refs, lora_preds):.4f}")
print(f"Base WER: {wer(refs, base_preds):.4f}")
print(f"LoRA WER: {wer(refs, lora_preds):.4f}")

results = pd.DataFrame({
    "reference": refs,
    "base_prediction": base_preds,
    "lora_prediction": lora_preds
})
os.makedirs("outputs/evaluation", exist_ok=True)
results.to_csv("outputs/evaluation/whisper_comparison_full.csv", index=False)
print("Predictions saved to outputs/evaluation/whisper_comparison_full.csv")
