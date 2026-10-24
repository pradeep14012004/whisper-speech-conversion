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

processor = WhisperProcessor.from_pretrained(ADAPTER)
base = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE)
model.eval()

df = pd.read_csv("outputs/training_manifest.csv")
df = df.sample(n=min(20, len(df)), random_state=123).reset_index(drop=True)

scores = []
for i, row in df.iterrows():
    audio, sr = sf.read(row["whisper_audio"])
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio.astype(np.float32))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    inputs = processor(
        audio.astype(np.float32), sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        ids = model.generate(
            inputs,
            language="chinese",
            task="transcribe",
            max_new_tokens=128
        )

    pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
    ref = str(row["transcript"])
    score = cer(ref, pred)
    scores.append(score)

    print("{}/{} | CER {:.4f}".format(i + 1, len(df), score))
    print("Reference:", ref)
    print("Prediction:", pred)

print("\nMean CER:", float(np.mean(scores)))
print("Lower CER is better.")
