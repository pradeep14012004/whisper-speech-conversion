import csv
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from pathlib import Path

SR, N_MELS, FRAMES = 16000, 80, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_mel(path):
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
    mean, std = float(logmel.mean()), float(logmel.std() + 1e-5)
    return ((logmel - mean) / std).astype(np.float32), mean, std

ckpt = torch.load("outputs/acoustic_baseline.pt", map_location="cpu")
model = ConvModel()
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()

with open("outputs/conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "valid"]

if not rows:
    raise SystemExit("ERROR: no validation pairs found.")

results = []
errors = []
for r in rows[:100]:
    try:
        x, _, _ = load_mel(r["whisper_audio"])
        y, _, _ = load_mel(r["normal_audio"])
        # Compare only equal-length time regions; report length mismatch separately.
        n = min(x.shape[1], y.shape[1], FRAMES)
        xt = torch.from_numpy(x[:, :n]).unsqueeze(0).to(DEVICE)
        yt = torch.from_numpy(y[:, :n]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(xt)
        pred = pred.squeeze(0).cpu().numpy()
        pred_l1 = float(np.mean(np.abs(pred - y[:, :n])))
        identity_l1 = float(np.mean(np.abs(x[:, :n] - y[:, :n])))
        results.append((pred_l1, identity_l1, x.shape[1], y.shape[1]))
    except Exception as e:
        errors.append(str(e))

if not results:
    raise SystemExit("ERROR: no samples evaluated: " + "; ".join(errors[:3]))

a = np.array(results)
print("Device:", DEVICE)
print("Validation examples evaluated:", len(results))
print("Read/evaluation errors:", len(errors))
print("Mean model L1:", round(float(a[:, 0].mean()), 4))
print("Mean identity-baseline L1:", round(float(a[:, 1].mean()), 4))
print("Model better than identity:", int(np.sum(a[:, 0] < a[:, 1])), "/", len(results))
print("Mean input frames:", round(float(a[:, 2].mean()), 1))
print("Mean target frames:", round(float(a[:, 3].mean()), 1))
print("Note: mel L1 is diagnostic, not a perceptual speech-quality metric.")
