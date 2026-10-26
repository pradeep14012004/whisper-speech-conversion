
import os
import random
from pathlib import Path
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

from av_fusion import AudioVisualFusion

ROOT = Path(".")
MANIFEST = ROOT / "outputs/verified_av_manifest.csv"
BASE = "openai/whisper-small"
ADAPTER = ROOT / "outputs/whisper_lora_small"
OUT = ROOT / "outputs/av_to_mel_bridge.pt"

SR, HOP, NFFT, N_MELS = 16000, 256, 1024, 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("AV_MEL_EPOCHS", "2"))
LR = 1e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")
if not MANIFEST.is_file():
    raise FileNotFoundError(MANIFEST)
if not ADAPTER.exists():
    raise FileNotFoundError(ADAPTER)


def load_audio(relative_path):
    path = ROOT / str(relative_path)
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio).astype(np.float32)
    if sr != SR:
        g = gcd(int(sr), SR)
        audio = resample_poly(
            audio, SR // g, int(sr) // g
        ).astype(np.float32)
    return audio


def make_mel(audio):
    features, _ = mel_spectogram(
        audio=torch.from_numpy(audio),
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


def resolve_feature(value):
    path = ROOT / str(value)
    if path.is_file():
        return path

    name = Path(str(value)).name
    candidates = [
        ROOT / "outputs/train_video_features" / (name + ".npy"),
        ROOT / "outputs/train_video_features" /
        (Path(name).stem + ".npy"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Video feature not found: {value}"
    )


df = pd.read_csv(MANIFEST)

required = [
    "split", "speaker_id", "whisper_audio",
    "normal_audio", "feature_path"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[df["split"].isin(["train", "valid"])].copy()
df["feature_resolved"] = df["feature_path"].map(resolve_feature)

for col in ["whisper_audio", "normal_audio"]:
    for path in df[col].astype(str):
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)

train_speakers = set(
    df.loc[df.split == "train", "speaker_id"].astype(str)
)
valid_speakers = set(
    df.loc[df.split == "valid", "speaker_id"].astype(str)
)
if train_speakers & valid_speakers:
    raise RuntimeError("Speaker leakage between train and validation.")

train_df = df[df.split == "train"].sample(
    frac=1, random_state=SEED
).reset_index(drop=True)
valid_df = df[df.split == "valid"].reset_index(drop=True)

print("Device:", DEVICE, flush=True)
print("Train rows:", len(train_df), flush=True)
print("Validation rows:", len(valid_df), flush=True)
print("Loading Whisper and LoRA adapter...", flush=True)

processor = WhisperProcessor.from_pretrained(str(ADAPTER))
base = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float32
)
whisper = PeftModel.from_pretrained(base, str(ADAPTER)).to(DEVICE)
whisper.eval()
for param in whisper.parameters():
    param.requires_grad = False

fusion = AudioVisualFusion(
    audio_dim=whisper.config.d_model,
    video_dim=128,
    hidden_dim=256,
    heads=4,
).to(DEVICE)

decoder = nn.Sequential(
    nn.Linear(256, 512),
    nn.GELU(),
    nn.Linear(512, N_MELS),
).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(fusion.parameters()) + list(decoder.parameters()),
    lr=LR,
    weight_decay=1e-4,
)

def encode_whisper(audio):
    features = processor(
        audio,
        sampling_rate=SR,
        return_tensors="pt",
    ).input_features.to(DEVICE)
    with torch.no_grad():
        hidden = whisper.get_encoder()(
            input_features=features
        ).last_hidden_state
    return hidden

def forward_one(row):
    source = load_audio(row["whisper_audio"])
    target = load_audio(row["normal_audio"])

    if len(source) < SR // 2 or len(target) < SR // 2:
        raise ValueError("Audio shorter than 0.5 seconds")

    duration = len(source) / SR
    hidden = encode_whisper(source)

    # Whisper encoder runs at approximately 50 steps per second.
    steps = max(1, min(hidden.shape[1], round(duration * 50)))
    hidden = hidden[:, :steps, :]

    video_np = np.load(row["feature_resolved"]).astype(np.float32)
    if video_np.shape != (16, 48, 48):
        raise ValueError(f"Unexpected video shape: {video_np.shape}")

    video = torch.from_numpy(
        np.nan_to_num(video_np)
    ).unsqueeze(0).to(DEVICE)

    fused = fusion(hidden, video)
    predicted = decoder(fused).transpose(1, 2)

    target_mel = make_mel(target).to(DEVICE)
    target_aligned = F.interpolate(
        target_mel.unsqueeze(0),
        size=steps,
        mode="linear",
        align_corners=False,
    )

    loss = F.l1_loss(predicted, target_aligned)
    return loss


@torch.no_grad()
def evaluate():
    fusion.eval()
    decoder.eval()
    losses = []

    for i, row in valid_df.iterrows():
        try:
            loss = forward_one(row)
            if torch.isfinite(loss):
                losses.append(loss.item())
        except Exception as exc:
            print(
                f"[validation skip {i}] {type(exc).__name__}: {exc}",
                flush=True,
            )

    if not losses:
        raise RuntimeError("No successful validation examples.")

    return float(np.mean(losses)), len(losses)


best_val = float("inf")
OUT.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    fusion.train()
    decoder.train()
    train_losses = []
    skipped = 0

    for i, row in train_df.iterrows():
        optimizer.zero_grad(set_to_none=True)

        try:
            loss = forward_one(row)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) +
                list(decoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            train_losses.append(loss.item())

        except Exception as exc:
            skipped += 1
            print(
                f"[training skip {i}] {type(exc).__name__}: {exc}",
                flush=True,
            )

        if (i + 1) % 100 == 0:
            recent = train_losses[-100:]
            average = float(np.mean(recent)) if recent else float("nan")
            print(
                f"Epoch {epoch}/{EPOCHS}, "
                f"row {i+1}/{len(train_df)}, "
                f"recent L1={average:.5f}, skipped={skipped}",
                flush=True,
            )

    if not train_losses:
        raise RuntimeError("No successful training examples.")

    train_loss = float(np.mean(train_losses))
    val_loss, val_count = evaluate()

    print(
        f"Epoch {epoch}: train L1={train_loss:.6f}, "
        f"validation L1={val_loss:.6f} "
        f"({val_count}/{len(valid_df)} examples)",
        flush=True,
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "fusion_state_dict": fusion.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "audio_dim": whisper.config.d_model,
                "video_dim": 128,
                "hidden_dim": 256,
                "n_mels": N_MELS,
                "sample_rate": SR,
                "hop_length": HOP,
                "n_fft": NFFT,
                "epoch": epoch,
                "train_loss": train_loss,
                "best_val_loss": best_val,
                "architecture": "AudioVisualFusion+MelDecoder",
            },
            OUT,
        )
        print("Saved best checkpoint:", OUT, flush=True)

print("Training complete. Best validation L1:", best_val, flush=True)
