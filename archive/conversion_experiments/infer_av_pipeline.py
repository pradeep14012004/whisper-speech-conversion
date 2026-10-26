import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import librosa

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from speechbrain.inference.vocoders import HIFIGAN

from av_fusion import AudioVisualFusion


ROOT = Path(".")
BASE = "openai/whisper-small"
ADAPTER = ROOT / "outputs/whisper_lora_small"
CHECKPOINT = ROOT / "outputs/av_to_mel_bridge.pt"
SR = 16000


def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio).astype(np.float32)

    if sr != SR:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=SR
        ).astype(np.float32)

    if len(audio) < int(0.5 * SR):
        raise ValueError("Input audio must be at least 0.5 seconds.")

    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Whispered Chinese WAV")
    parser.add_argument("--video-features", required=True, help="Matching .npy feature array")
    parser.add_argument("--output", default="outputs/av_inference")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not ADAPTER.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"AV checkpoint not found: {CHECKPOINT}")

    audio = load_audio(args.audio)

    video = np.load(args.video_features).astype(np.float32)
    if video.shape != (16, 48, 48):
        raise ValueError(
            f"Expected video features shape (16, 48, 48), got {video.shape}"
        )
    if not np.isfinite(video).all():
        raise ValueError("Video features contain NaN or infinity.")

    print("Device:", device)
    print("Audio duration:", round(len(audio) / SR, 2), "seconds")
    print("Video feature shape:", video.shape)
    print("Loading Whisper + LoRA...")

    processor = WhisperProcessor.from_pretrained(str(ADAPTER))
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32
    )
    whisper = PeftModel.from_pretrained(base, str(ADAPTER)).to(device)
    whisper.eval()

    ckpt = torch.load(CHECKPOINT, map_location=device)

    fusion = AudioVisualFusion(
        audio_dim=ckpt["audio_dim"],
        video_dim=ckpt["video_dim"],
        hidden_dim=ckpt["hidden_dim"],
        heads=4,
    ).to(device)
    decoder = nn.Sequential(
        nn.Linear(ckpt["hidden_dim"], 512),
        nn.GELU(),
        nn.Linear(512, ckpt["n_mels"]),
    ).to(device)

    fusion.load_state_dict(ckpt["fusion_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    fusion.eval()
    decoder.eval()

    inputs = processor(
        audio,
        sampling_rate=SR,
        return_tensors="pt",
    ).input_features.to(device)

    with torch.no_grad():
        hidden = whisper.get_encoder()(
            input_features=inputs
        ).last_hidden_state

        # Match the time-step convention used during training.
        steps = min(hidden.shape[1], round(len(audio) / SR * 50))
        if steps < 1:
            raise ValueError("No Whisper encoder steps available.")
        hidden = hidden[:, :steps, :]

        video_tensor = torch.from_numpy(video).unsqueeze(0).to(device)
        fused = fusion(hidden, video_tensor)
        predicted = decoder(fused).transpose(1, 2).squeeze(0)

    mel = predicted.float().cpu().numpy()
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)

    mel_path = output.with_suffix(".npy")
    np.save(mel_path, mel)

    print("Predicted mel shape:", mel.shape)
    print("Mel range:", float(mel.min()), "to", float(mel.max()))
    print("Saved predicted mel:", mel_path)

    # Exploratory synthesis only: predicted mel values may not yet match
    # the exact input representation expected by the pretrained vocoder.
    print("Loading HiFi-GAN...")
    vocoder = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz",
        savedir=str(ROOT / "pretrained_models" / "speechbrain_hifigan"),
        run_opts={"device": device},
    )

    vocoder_input = torch.from_numpy(mel.copy()).unsqueeze(0).to(device)
    print("HiFi-GAN input shape:", tuple(vocoder_input.shape))

    with torch.no_grad():
        wav = vocoder.decode_batch(vocoder_input)

    wav = wav.squeeze().float().cpu().numpy()
    wav = np.nan_to_num(wav).astype(np.float32)

    if wav.size == 0:
        raise RuntimeError("HiFi-GAN returned an empty waveform.")

    peak = float(np.max(np.abs(wav)))
    if peak > 1e-8:
        wav = 0.95 * wav / peak

    wav_path = output.with_name(output.name + "_converted.wav")
    sf.write(wav_path, wav, SR)

    print("Saved exploratory WAV:", wav_path)
    print("Output duration:", round(len(wav) / SR, 2), "seconds")
    print("Note: listen to the output; successful synthesis does not guarantee intelligible speech.")


if __name__ == "__main__":
    main()
