import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(".")
OUT = ROOT / "outputs"
SR, MELS = 16000, 80
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
    OUT / "acoustic_conversion_v2.pt",
    map_location=device,
    weights_only=False,
)
model = Net().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def mel(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=400, hop_length=160,
        n_mels=MELS, fmin=0, fmax=SR // 2,
    )
    return np.log(np.maximum(m, 1e-5)).astype(np.float32)

with open(OUT / "conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "valid"]

results = []
skipped = 0

for i, r in enumerate(rows):
    try:
        x = mel(r["whisper_audio"])
        y = mel(r["normal_audio"])

        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        yt = torch.from_numpy(y).unsqueeze(0).to(device)
        yt = F.interpolate(
            yt, size=xt.shape[-1],
            mode="linear", align_corners=False,
        )

        with torch.no_grad():
            pred = model(xt)

        model_mae = F.l1_loss(pred, yt).item()
        identity_mae = F.l1_loss(xt, yt).item()

        results.append({
            "whisper_audio": r["whisper_audio"],
            "normal_audio": r["normal_audio"],
            "model_mae": model_mae,
            "identity_mae": identity_mae,
            "improvement": identity_mae - model_mae,
        })
    except Exception as e:
        skipped += 1
        print("SKIP:", r.get("whisper_audio", "?"), repr(e))

if not results:
    raise RuntimeError("No validation examples evaluated.")

model_avg = np.mean([r["model_mae"] for r in results])
identity_avg = np.mean([r["identity_mae"] for r in results])
improvement = identity_avg - model_avg
relative = 100 * improvement / max(identity_avg, 1e-12)

csv_path = OUT / "conversion_validation_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n===== VALIDATION SUMMARY =====")
print("Evaluated:", len(results))
print("Skipped:", skipped)
print(f"Model mean mel MAE:    {model_avg:.6f}")
print(f"Identity mean mel MAE: {identity_avg:.6f}")
print(f"Absolute improvement:  {improvement:.6f}")
print(f"Relative improvement:  {relative:.2f}%")
print("CSV saved:", csv_path)

if improvement > 0:
    print("RESULT: Model improves mean mel MAE over identity baseline.")
else:
    print("RESULT: Model does NOT improve mean mel MAE over identity baseline.")
print("Note: Mel MAE is not a perceptual speech-quality metric.")
