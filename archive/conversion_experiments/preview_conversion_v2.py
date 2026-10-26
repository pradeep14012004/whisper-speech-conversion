import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn

SR, MELS, HOP, NFFT = 16000, 80, 160, 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "outputs/acoustic_conversion_v2.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(MELS, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, MELS, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x)

ckpt = torch.load(CHECKPOINT, map_location="cpu")
model = Net()
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()

def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=NFFT, hop_length=HOP,
        n_mels=MELS, fmin=0, fmax=SR // 2
    )
    return np.log(np.maximum(mel, 1e-5)).astype(np.float32)

with open("outputs/conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "valid"]

if not rows:
    raise SystemExit("No validation rows found.")

r = rows[0]
audio = load_audio(r["whisper_audio"])
x = to_logmel(audio)

with torch.no_grad():
    pred = model(torch.from_numpy(x).unsqueeze(0).to(DEVICE))
pred = pred.squeeze(0).cpu().numpy()

# Experimental Griffin-Lim reconstruction; not a trained vocoder.
mel_power = np.exp(np.clip(pred, -12, 8))
mel_power = np.maximum(mel_power, 1e-8)
preview = librosa.feature.inverse.mel_to_audio(
    mel_power, sr=SR, n_fft=NFFT, hop_length=HOP,
    fmin=0, fmax=SR // 2, n_iter=32
)

# Match the source duration, since the model predicts one frame per input frame.
preview = librosa.util.fix_length(preview, size=len(audio))
out = Path("outputs/conversion_v2_preview.wav")
sf.write(out, preview, SR)

print("Device:", DEVICE)
print("Input:", r["whisper_audio"])
print("Transcript:", r["transcript"])
print("Input duration:", round(len(audio) / SR, 2), "seconds")
print("Preview:", out)
print("Preview duration:", round(len(preview) / SR, 2), "seconds")
print("WARNING: experimental output; listen and evaluate intelligibility.")
