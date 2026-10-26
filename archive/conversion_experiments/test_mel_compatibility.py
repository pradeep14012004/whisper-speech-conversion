from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram as fastspeech_mel
from speechbrain.lobes.models.HifiGAN import mel_spectogram as hifigan_mel

ROOT = Path("/home/jovyan/whisper_speech_conversion")
OUT = ROOT / "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use the already-created clear reference WAV.
wav_path = OUT / "hifigan_fixed_reference.wav"
audio, sr = sf.read(str(wav_path), dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
assert sr == 16000, f"Expected 16000 Hz, got {sr}"

audio_t = torch.from_numpy(audio).to(DEVICE)

vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir=str(ROOT / "pretrained_models" / "speechbrain_hifigan"),
    run_opts={"device": DEVICE},
)

def report(name, mel):
    mel = mel.detach().float()
    print(
        f"{name}: shape={tuple(mel.shape)}, "
        f"min={mel.min().item():.4f}, max={mel.max().item():.4f}, "
        f"mean={mel.mean().item():.4f}, std={mel.std().item():.4f}"
    )
    if mel.ndim == 2 and mel.shape[0] != 80 and mel.shape[1] == 80:
        mel = mel.T
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel.squeeze(0)
    if mel.ndim != 2 or mel.shape[0] != 80:
        raise ValueError(f"Unexpected mel shape for {name}: {tuple(mel.shape)}")
    return mel

with torch.no_grad():
    mel_fs, _ = fastspeech_mel(
        audio=audio_t,
        sample_rate=16000,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )

    # Try HiFi-GAN's own mel function with explicit parameters.
    mel_hg = hifigan_mel(
        audio=audio_t,
        sample_rate=16000,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )

mel_fs = report("FastSpeech2 mel", mel_fs).cpu()
mel_hg = report("HiFi-GAN mel", mel_hg).cpu()

print("Mean absolute difference between mels:",
      (mel_fs - mel_hg).abs().mean().item())

for name, mel in [
    ("reference_resynth_fastspeech_mel.wav", mel_fs),
    ("reference_resynth_hifigan_mel.wav", mel_hg),
]:
    with torch.no_grad():
        generated = vocoder.decode_batch(mel.unsqueeze(0).to(DEVICE))
    generated = generated.squeeze().cpu().numpy()
    generated = np.nan_to_num(generated).astype(np.float32)
    sf.write(str(OUT / name), generated, 16000)
    print("Saved:", OUT / name)

print("Done. Play both new WAV files and compare them with the original reference.")
