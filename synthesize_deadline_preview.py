import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn

ROOT = Path(".")
OUT = ROOT / "outputs" / "deadline_preview"
OUT.mkdir(parents=True, exist_ok=True)

SR, MELS, N_FFT, HOP = 16000, 80, 400, 160
device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(80, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 80, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x)

ckpt = torch.load(
    ROOT / "outputs" / "acoustic_conversion_v2.pt",
    map_location=device,
    weights_only=False,
)
model = Net().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def get_logmel(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=MELS, fmin=0, fmax=SR // 2,
    )
    return np.log(np.maximum(mel, 1e-5)).astype(np.float32)

def reconstruct(logmel):
    # Undo the natural-log power-mel representation.
    power_mel = np.exp(np.clip(logmel, -11.5, 5.0))
    power_mel = np.maximum(power_mel, 1e-10)

    wav = librosa.feature.inverse.mel_to_audio(
        power_mel,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_iter=48,
        power=2.0,
        fmin=0,
        fmax=SR // 2,
    )
    wav = np.nan_to_num(wav)
    peak = np.max(np.abs(wav))
    if peak > 1e-8:
        wav = 0.95 * wav / peak
    return wav.astype(np.float32)

with open(ROOT / "outputs" / "conversion_pairs.csv",
          encoding="utf-8-sig") as f:
    row = next(
        r for r in csv.DictReader(f)
        if r["split"] == "valid"
    )

x = get_logmel(ROOT / row["whisper_audio"])
target = get_logmel(ROOT / row["normal_audio"])

with torch.no_grad():
    pred = model(
        torch.from_numpy(x).unsqueeze(0).to(device)
    ).squeeze(0).cpu().numpy()

# Same time alignment used by the current training objective.
target_t = torch.nn.functional.interpolate(
    torch.from_numpy(target).unsqueeze(0),
    size=x.shape[1],
    mode="linear",
    align_corners=False,
).squeeze(0).numpy()

items = {
    "whisper_input_reconstructed.wav": x,
    "normal_reference_reconstructed.wav": target_t,
    "converted_prediction_reconstructed.wav": pred,
}

for name, mel in items.items():
    wav = reconstruct(mel)
    path = OUT / name
    sf.write(path, wav, SR)
    print(f"Saved {path} | {len(wav)/SR:.2f}s")

print("\nTranscript:", row["transcript"])
print("These are Griffin-Lim baseline reconstructions, not HiFi-GAN output.")
