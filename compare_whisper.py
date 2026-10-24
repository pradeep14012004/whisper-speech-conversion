import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import cer

BASE = "openai/whisper-small"
ADAPTER = "outputs/whisper_lora_small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("outputs/training_manifest.csv")

# Exclude the 32 pilot-training samples
train_df = df.sample(n=min(32, len(df)), random_state=42)
df = df.drop(train_df.index).sample(
    n=min(20, len(df) - 32), random_state=123
).reset_index(drop=True)

processor = WhisperProcessor.from_pretrained(ADAPTER)

base = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float32
).to(DEVICE)
base.eval()

adapted_base = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float32
)
adapted = PeftModel.from_pretrained(
    adapted_base, ADAPTER
).to(DEVICE)
adapted.eval()

base_scores, lora_scores = [], []

for i, row in df.iterrows():
    audio, sr = sf.read(row["whisper_audio"])
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio.astype(np.float32))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    results = []
    for model in (base, adapted):
        with torch.no_grad():
            ids = model.generate(
                features, language="chinese",
                task="transcribe", max_new_tokens=128
            )
        results.append(processor.batch_decode(
            ids, skip_special_tokens=True
        )[0])

    ref = str(row["transcript"])
    b, l = cer(ref, results[0]), cer(ref, results[1])
    base_scores.append(b)
    lora_scores.append(l)
    print("{}/{} | Base CER {:.4f} | LoRA CER {:.4f}".format(
        i + 1, len(df), b, l
    ))

print("\nBase Whisper-small mean CER:", float(np.mean(base_scores)))
print("LoRA Whisper-small mean CER:", float(np.mean(lora_scores)))
print("Lower CER is better.")
