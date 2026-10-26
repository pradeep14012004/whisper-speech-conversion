import csv
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
OUT = ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR, MELS, HOP, NFFT = 16000, 80, 256, 1024

class ConversionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(80, 192, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(192, 192, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(192, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 80, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x)

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
    if mel.shape[0] != MELS:
        raise ValueError(f"Unexpected mel shape: {tuple(mel.shape)}")
    return mel

ckpt_path = OUT / "acoustic_conversion_hifigan_compatible.pt"
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model = ConversionNet().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir=str(ROOT / "pretrained_models" / "speechbrain_hifigan"),
    run_opts={"device": DEVICE},
)

with open(OUT / "conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

row = next(r for r in rows if r["split"] == "valid")
source_audio = load_audio(row["whisper_audio"])
normal_audio = load_audio(row["normal_audio"])

source_mel = to_mel(source_audio)
target_mel = to_mel(normal_audio)

# Model expects [batch, mel_bins, time].
with torch.no_grad():
    predicted_mel = model(source_mel.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

target_mel = F.interpolate(
    target_mel.unsqueeze(0),
    size=source_mel.shape[-1],
    mode="linear",
    align_corners=False,
).squeeze(0)

def synthesize(mel, filename):
    # IMPORTANT: HiFi-GAN decode_batch expects [batch, time, mel_bins].
    vocoder_input = mel.unsqueeze(0).to(DEVICE)
    print("Vocoder input shape:", tuple(vocoder_input.shape))
    with torch.no_grad():
        wav = vocoder.decode_batch(vocoder_input)

    wav = np.nan_to_num(wav.squeeze().cpu().numpy()).astype(np.float32)
    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if peak > 1e-8:
        wav = 0.95 * wav / peak

    path = OUT / filename
    sf.write(path, wav, SR)
    print("Saved:", path, "| duration:", round(len(wav) / SR, 2), "seconds")

synthesize(source_mel, "hifigan_fixed_input.wav")
synthesize(target_mel, "hifigan_fixed_reference.wav")
synthesize(predicted_mel, "hifigan_fixed_converted.wav")

print("Transcript:", row.get("transcript", ""))
