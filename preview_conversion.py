import csv
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from pathlib import Path

SR, N_MELS, FRAMES = 16000, 80, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = torch.load("outputs/acoustic_baseline.pt", map_location="cpu")

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(80, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 80, 5, padding=2)
        )
    def forward(self, x):
        return self.net(x)

model = ConvModel()
model.load_state_dict(CKPT["model_state_dict"])
model.to(DEVICE).eval()

with open("outputs/conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "valid"]

if not rows:
    raise SystemExit("No validation rows found.")

r = rows[0]
def load_mel(path, fixed_length=True):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=400, hop_length=160,
        n_mels=N_MELS, fmin=0, fmax=SR // 2
    )
    logmel = np.log(np.maximum(mel, 1e-5))
    mean, std = logmel.mean(), logmel.std() + 1e-5
    if fixed_length:
        logmel = librosa.util.fix_length(logmel, size=FRAMES, axis=1)
    return logmel.astype(np.float32), mean, std

x, _, _ = load_mel(r["whisper_audio"], fixed_length=False)
_, target_mean, target_std = load_mel(r["normal_audio"], fixed_length=False)

with torch.no_grad():
    pred = model(torch.from_numpy(x).unsqueeze(0).to(DEVICE))
pred = pred.squeeze(0).cpu().numpy()

# Undo approximate normalization using the normal target's statistics.
pred = pred * target_std + target_mean
mel = np.exp(np.clip(pred, -12, 8))
audio = librosa.feature.inverse.mel_to_audio(
    mel, sr=SR, n_fft=400, hop_length=160,
    fmin=0, fmax=SR // 2, n_iter=32
)

out = Path("outputs/conversion_preview.wav")
sf.write(out, audio, SR)
print("Input whispered audio:", r["whisper_audio"])
print("Reference normal audio:", r["normal_audio"])
print("Transcript:", r["transcript"])
print("Preview:", out, "| duration:", round(len(audio)/SR, 2), "seconds")
print("Note: rough experimental preview, not a validated speech conversion.")
