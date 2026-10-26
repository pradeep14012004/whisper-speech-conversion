import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from math import gcd
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

ROOT = Path(".")
OUT = ROOT / "outputs"
SR, HOP, NFFT, N_MELS = 16000, 256, 1024, 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = OUT / "acoustic_conversion_v2.pt"


def load_audio(relative_path):
    path = ROOT / relative_path
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        audio = resample_poly(audio, SR // g, int(sr) // g)
    return np.nan_to_num(audio).astype(np.float32)


def mel(audio):
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


ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
if ckpt.get("architecture") != "ResidualConversionNet":
    raise RuntimeError(f"Unexpected checkpoint architecture: {ckpt.get('architecture')}")

model = ResidualConversionNet().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir=str(ROOT / "pretrained_models" / "speechbrain_hifigan"),
    run_opts={"device": DEVICE},
)

with open(OUT / "conversion_pairs.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

# Choose a validation pair whose two audio files are readable.
candidates = [r for r in rows if r["split"] == "valid"]
random.seed(42)
random.shuffle(candidates)

selected = None
for r in candidates:
    try:
        source = load_audio(r["whisper_audio"])
        target = load_audio(r["normal_audio"])
        if len(source) >= SR and len(target) >= SR:
            selected = r
            break
    except Exception as e:
        print("Skipping pair:", repr(e))

if selected is None:
    raise RuntimeError("Could not find a usable validation pair.")

source_mel = mel(source)
target_mel = mel(target)

with torch.no_grad():
    predicted_mel = model(source_mel.unsqueeze(0).to(DEVICE)).squeeze(0)

target_aligned = F.interpolate(
    target_mel.unsqueeze(0),
    size=source_mel.shape[-1],
    mode="linear",
    align_corners=False,
).squeeze(0)

def synthesize(mel_tensor, path):
    # This SpeechBrain HiFi-GAN expects [batch, mel_channels, time].
    if mel_tensor.ndim != 2 or mel_tensor.shape[0] != N_MELS:
        raise ValueError(f"Expected mel shape ({N_MELS}, time), got {tuple(mel_tensor.shape)}")
    vocoder_input = mel_tensor.unsqueeze(0).to(DEVICE)
    print("Vocoder input shape:", tuple(vocoder_input.shape))
    with torch.no_grad():
        waveform = vocoder.decode_batch(vocoder_input)
    waveform = waveform.squeeze().detach().cpu().numpy()
    waveform = np.nan_to_num(waveform).astype(np.float32)
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    sf.write(str(path), waveform, SR)
    print("Saved:", path, "| duration:", round(len(waveform) / SR, 2), "sec")

# Save source, target, and conversion through the same vocoder for comparison.
synthesize(source_mel, OUT / "v2_whisper_reconstructed.wav")
synthesize(target_aligned, OUT / "v2_normal_reference_reconstructed.wav")
synthesize(predicted_mel.squeeze(0) if predicted_mel.ndim == 3 else predicted_mel,
           OUT / "v2_converted_hifigan.wav")

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)
print("Architecture:", ckpt["architecture"])
print("Best validation L1:", ckpt.get("best_val_loss"))
print("Whisper input:", selected["whisper_audio"])
print("Normal target:", selected["normal_audio"])
print("Transcript:", selected.get("transcript", ""))
print("Mel shapes:", tuple(source_mel.shape), tuple(predicted_mel.squeeze(0).shape))
