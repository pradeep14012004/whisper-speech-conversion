import csv
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

ROOT = Path("/home/jovyan/whisper_speech_conversion")
MANIFEST = ROOT / "outputs/conversion_pairs.csv"
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "acoustic_conversion_hifigan_compatible.pt"
DEVICE = "cpu"  # Current NVIDIA driver is incompatible with installed CUDA.
SR = 16000
N_MELS = 80
HOP = 256
NFFT = 1024
EPOCHS = 3
MAX_TRAIN = 1200
MAX_VALID = 200
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def load_audio(path):
    y, sr = sf.read(str(resolve(path)), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.nan_to_num(y).astype(np.float32)

def compatible_mel(y):
    # Keep these settings consistent with the HiFi-GAN test.
    audio = torch.from_numpy(y)
    mel, _ = mel_spectogram(
        audio=audio,
        sample_rate=SR,
        hop_length=HOP,
        win_length=NFFT,
        n_mels=N_MELS,
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
    if mel.ndim != 2:
        raise ValueError(f"Unexpected mel shape: {tuple(mel.shape)}")
    if mel.shape[0] != N_MELS and mel.shape[1] == N_MELS:
        mel = mel.transpose(0, 1)
    if mel.shape[0] != N_MELS:
        raise ValueError(f"Expected {N_MELS} mel bins, got {tuple(mel.shape)}")
    return mel

class ConversionNet(nn.Module):
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
        return self.net(x)

with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]

random.shuffle(train_rows)
random.shuffle(valid_rows)
train_rows = train_rows[:MAX_TRAIN]
valid_rows = valid_rows[:MAX_VALID]

if not train_rows or not valid_rows:
    raise RuntimeError("Training or validation split is empty.")

print("Device:", DEVICE)
print("Training pairs:", len(train_rows))
print("Validation pairs:", len(valid_rows))
print("Precomputing compatible mel features...")

def prepare(rows):
    result = []
    for i, r in enumerate(rows, 1):
        try:
            source = compatible_mel(load_audio(r["whisper_audio"]))
            target = compatible_mel(load_audio(r["normal_audio"]))

            # Approximate duration alignment; not phoneme alignment.
            target = F.interpolate(
                target.unsqueeze(0),
                size=source.shape[-1],
                mode="linear",
                align_corners=False,
            ).squeeze(0)

            result.append((source, target, r))
        except Exception as e:
            print(f"Skipping pair {i}: {e}")

        if i % 100 == 0:
            print(f"Prepared {i}/{len(rows)}")

    return result

train_data = prepare(train_rows)
valid_data = prepare(valid_rows)

if not train_data or not valid_data:
    raise RuntimeError("No usable feature pairs were prepared.")

model = ConversionNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    random.shuffle(train_data)
    train_losses = []

    for source, target, _ in train_data:
        x = source.unsqueeze(0).to(DEVICE)
        y = target.unsqueeze(0).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.l1_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for source, target, _ in valid_data:
            pred = model(source.unsqueeze(0).to(DEVICE))
            val_losses.append(F.l1_loss(pred, target.unsqueeze(0).to(DEVICE)).item())

    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))
    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"train_L1={train_loss:.4f} valid_L1={val_loss:.4f}",
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
                "feature_extraction": "SpeechBrain FastSpeech2 mel_spectogram",
                "train_examples": len(train_data),
                "valid_examples": len(valid_data),
                "seed": SEED,
            },
            MODEL_PATH,
        )
        print("Saved best checkpoint:", MODEL_PATH)

print("Training complete. Best validation L1:", best_val)

# Reload the best model and synthesize a validation prediction.
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir=str(ROOT / "pretrained_models" / "speechbrain_hifigan"),
    run_opts={"device": DEVICE},
)

source, target, row = valid_data[0]
with torch.no_grad():
    pred = model(source.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

def synthesize(mel, filename):
    with torch.no_grad():
        wav = vocoder.decode_batch(mel.unsqueeze(0).to(DEVICE))
    wav = wav.squeeze().cpu().numpy()
    wav = np.nan_to_num(wav).astype(np.float32)
    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if peak > 1e-8:
        wav = 0.95 * wav / peak
    path = OUT / filename
    sf.write(path, wav, SR)
    print("Saved:", path, "| duration:", round(len(wav) / SR, 2), "seconds")

synthesize(source, "hifigan_compatible_input.wav")
synthesize(target, "hifigan_compatible_normal_reference.wav")
synthesize(pred, "hifigan_compatible_converted.wav")

print("Transcript:", row.get("transcript", ""))
print("Listen to all three files before deciding whether to scale up.")
