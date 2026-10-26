import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

ROOT = Path(".")
SR, HOP, NFFT, MELS = 16000, 256, 1024, 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(path):
    y, sr = sf.read(str(ROOT / path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    return np.nan_to_num(y).astype(np.float32)

def mel(y):
    m, _ = mel_spectogram(
        audio=torch.from_numpy(y), sample_rate=SR,
        hop_length=HOP, win_length=NFFT, n_fft=NFFT,
        n_mels=MELS, f_min=0, f_max=SR / 2, power=1,
        normalized=False, min_max_energy_norm=True,
        norm="slaney", mel_scale="slaney", compression=True,
    )
    return torch.nan_to_num(m.float())

class ResidualConversionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(MELS, 192, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(192, 192, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(192, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, MELS, 5, padding=2),
        )
    def forward(self, x):
        return x + self.net(x)

ckpt = torch.load("outputs/acoustic_conversion_v2.pt",
                  map_location=DEVICE, weights_only=False)
model = ResidualConversionNet().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open("outputs/conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
row = next(r for r in rows if r["whisper_audio"] ==
           "train/audio/hfm/whisper/S0209/S0209_M-0209-2_030936-031476.wav")

x = mel(load_audio(row["whisper_audio"]))
y = mel(load_audio(row["normal_audio"]))
y = F.interpolate(y.unsqueeze(0), size=x.shape[-1],
                  mode="linear", align_corners=False).squeeze(0)

with torch.no_grad():
    pred = model(x.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

for name, z in [("source", x), ("target", y), ("prediction", pred)]:
    print(f"{name:10s} shape={tuple(z.shape)} "
          f"min={z.min().item():.4f} max={z.max().item():.4f} "
          f"mean={z.mean().item():.4f} std={z.std().item():.4f}")

print("Source-target MAE:", F.l1_loss(x, y).item())
print("Prediction-target MAE:", F.l1_loss(pred, y).item())
print("Prediction-source MAE:", F.l1_loss(pred, x).item())
print("Checkpoint best validation L1:", ckpt["best_val_loss"])
