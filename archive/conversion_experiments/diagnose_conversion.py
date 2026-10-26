import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
from pathlib import Path
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

ROOT = Path.cwd()
OUT = ROOT / "outputs"
SR, HOP, NFFT, MELS = 16000, 256, 1024, 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ConversionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(80, 192, 7, padding=3), nn.GELU(),
            nn.Conv1d(192, 192, 5, padding=2), nn.GELU(),
            nn.Conv1d(192, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 80, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x)

def load_audio(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.nan_to_num(y).astype(np.float32)

def to_mel(y):
    mel, _ = mel_spectogram(
        audio=torch.from_numpy(y),
        sample_rate=SR,
        hop_length=HOP,
        win_length=NFFT,
        n_mels=MELS,
        n_fft=NFFT,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )
    mel = mel.detach().float().cpu()
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel.squeeze(0)
    if mel.shape[0] != MELS and mel.shape[-1] == MELS:
        mel = mel.transpose(0, 1)
    return mel

def stats(name, x):
    x = x.detach().float().cpu()
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={x.min().item():.5f}, max={x.max().item():.5f}, "
        f"mean={x.mean().item():.5f}, std={x.std().item():.5f}, "
        f"finite={torch.isfinite(x).all().item()}"
    )

ckpt = torch.load(OUT / "acoustic_conversion_hifigan_compatible.pt",
                  map_location="cpu")
model = ConversionNet().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open(OUT / "conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
row = next(r for r in rows if r["split"] == "valid")

source = to_mel(load_audio(ROOT / row["whisper_audio"]))
target = to_mel(load_audio(ROOT / row["normal_audio"]))
target = F.interpolate(target.unsqueeze(0), size=source.shape[-1],
                       mode="linear", align_corners=False).squeeze(0)

with torch.no_grad():
    pred = model(source.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

print("Device:", DEVICE)
print("Transcript:", row.get("transcript", "(not in manifest)"))
stats("Whisper/source mel", source)
stats("Normal/target mel", target)
stats("Predicted mel", pred)
print("L1 identity source-to-target:", F.l1_loss(source, target).item())
print("L1 prediction-to-target:", F.l1_loss(pred, target).item())
print("L1 prediction-to-source:", F.l1_loss(pred, source).item())
