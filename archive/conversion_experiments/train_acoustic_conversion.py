import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

ROOT = Path("/home/jovyan/whisper_speech_conversion")
MANIFEST = ROOT / "outputs/conversion_pairs.csv"
MODEL_PATH = ROOT / "outputs/acoustic_conversion_v2.pt"

SR = 16000
N_MELS = 80
HOP = 256
NFFT = 1024
EPOCHS = 10
MAX_TRAIN = 3000
MAX_VALID = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)


def load_audio(relative_path):
    path = ROOT / relative_path
    audio, sr = sf.read(str(path), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SR
        )

    return np.nan_to_num(audio).astype(np.float32)


def mel(y):
    features, _ = mel_spectogram(
        audio=torch.from_numpy(y),
        sample_rate=SR,
        hop_length=HOP,
        win_length=NFFT,
        n_fft=NFFT,
        n_mels=N_MELS,
        f_min=0,
        f_max=SR / 2,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )
    return torch.nan_to_num(features.float())


with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


def duration(path):
    return sf.info(str(ROOT / path)).duration


# Exclude very short clips and extreme duration mismatches.
filtered = []
for r in rows:
    try:
        ds = duration(r["whisper_audio"])
        dt = duration(r["normal_audio"])
        ratio = dt / ds if ds > 0 else 0

        if ds >= 1.0 and dt >= 1.0 and 0.7 <= ratio <= 1.4:
            filtered.append(r)
    except Exception as e:
        print("Skipping unreadable pair:", e)

train_rows = [r for r in filtered if r["split"] == "train"]
valid_rows = [r for r in filtered if r["split"] == "valid"]

random.shuffle(train_rows)
random.shuffle(valid_rows)
train_rows = train_rows[:MAX_TRAIN]
valid_rows = valid_rows[:MAX_VALID]

print("Device:", DEVICE)
print("Filtered pairs:", len(filtered))
print("Train selected:", len(train_rows))
print("Validation selected:", len(valid_rows))


def prepare(rows):
    data = []

    for i, r in enumerate(rows, 1):
        try:
            x = mel(load_audio(r["whisper_audio"]))
            y = mel(load_audio(r["normal_audio"]))

            # Approximate duration alignment only; not phoneme alignment.
            y = F.interpolate(
                y.unsqueeze(0),
                size=x.shape[-1],
                mode="linear",
                align_corners=False,
            ).squeeze(0)

            data.append((x, y))
        except Exception as e:
            print(f"Skipping pair {i}: {e}")

        if i % 100 == 0:
            print(f"Prepared {i}/{len(rows)}")

    return data


train_data = prepare(train_rows)
valid_data = prepare(valid_rows)

if not train_data or not valid_data:
    raise RuntimeError("Training or validation data is empty.")


class ResidualConversionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_MELS, 192, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(192, 192, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(192, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, N_MELS, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return x + self.net(x)


model = ResidualConversionNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    random.shuffle(train_data)
    losses = []

    for x, y in train_data:
        x = x.unsqueeze(0).to(DEVICE)
        y = y.unsqueeze(0).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.l1_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    vals = []

    with torch.no_grad():
        for x, y in valid_data:
            pred = model(x.unsqueeze(0).to(DEVICE))
            vals.append(
                F.l1_loss(pred, y.unsqueeze(0).to(DEVICE)).item()
            )

    train_loss = float(np.mean(losses))
    val_loss = float(np.mean(vals))

    print(
        f"Epoch {epoch}/{EPOCHS}: "
        f"train_L1={train_loss:.4f}, valid_L1={val_loss:.4f}",
        flush=True,
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "n_mels": N_MELS,
                "sample_rate": SR,
                "hop_length": HOP,
                "n_fft": NFFT,
                "architecture": "ResidualConversionNet",
                "best_val_loss": best_val,
            },
            MODEL_PATH,
        )
        print("Saved best checkpoint:", MODEL_PATH)

print("Training complete. Best validation L1:", best_val)
