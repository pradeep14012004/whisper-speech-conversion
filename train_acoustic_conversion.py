import csv, random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
SR, MELS = 16000, 80
EPOCHS, MAX_TRAIN, MAX_VALID = 3, 1200, 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with open("outputs/conversion_pairs.csv", encoding="utf-8-sig") as f:
    all_rows = list(csv.DictReader(f))
train_rows = [r for r in all_rows if r["split"] == "train"]
valid_rows = [r for r in all_rows if r["split"] == "valid"]
random.shuffle(train_rows)
random.shuffle(valid_rows)
train_rows, valid_rows = train_rows[:MAX_TRAIN], valid_rows[:MAX_VALID]

def mel(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=400, hop_length=160,
        n_mels=MELS, fmin=0, fmax=SR // 2
    )
    return np.log(np.maximum(m, 1e-5)).astype(np.float32)

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

model = Net().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

def evaluate(rows, training):
    model.train(training)
    total = 0.0
    used = 0
    for r in rows:
        try:
            x = mel(r["whisper_audio"])
            y = mel(r["normal_audio"])
            # Align frame counts by time-normalizing the target to input length.
            # This is an approximate baseline, not forced phoneme alignment.
            xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
            yt = torch.from_numpy(y).unsqueeze(0).to(DEVICE)
            yt = F.interpolate(yt, size=xt.shape[-1], mode="linear", align_corners=False)
            if training:
                opt.zero_grad(set_to_none=True)
                pred = model(xt)
                loss = F.l1_loss(pred, yt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                with torch.no_grad():
                    pred = model(xt)
                    loss = F.l1_loss(pred, yt)
            total += float(loss.detach())
            used += 1
        except Exception as e:
            print("SKIP:", e)
    return total / max(used, 1), used

print("Device:", DEVICE)
print("Training rows:", len(train_rows), "Validation rows:", len(valid_rows))
for epoch in range(1, EPOCHS + 1):
    random.shuffle(train_rows)
    tr, nt = evaluate(train_rows, True)
    va, nv = evaluate(valid_rows, False)
    print(f"Epoch {epoch}/{EPOCHS} train_L1={tr:.4f} valid_L1={va:.4f} "
          f"used_train={nt} used_valid={nv}", flush=True)

out = Path("outputs/acoustic_conversion_v2.pt")
torch.save({
    "model_state_dict": model.cpu().state_dict(),
    "n_mels": MELS, "sample_rate": SR, "hop_length": 160,
    "n_fft": 400, "normalization": "none",
    "architecture": "3-layer Conv1d; target time-resized to input length",
    "train_examples": len(train_rows), "valid_examples": len(valid_rows),
    "seed": SEED,
}, out)
print("Saved:", out)
print("WARNING: experimental mel predictor; vocoder and perceptual evaluation still required.")
