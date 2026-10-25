import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput
from peft import PeftModel

BASE = "openai/whisper-small"
ADAPTER = "outputs/whisper_lora_small"
MANIFEST = "outputs/train_av_manifest.csv"
OUT = "outputs/av_transcription_full.pt"

MAX_TRAIN_ROWS = 10000
MAX_TARGET_TOKENS = 128
LR = 1e-4
LOG_EVERY = 25

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

DEVICE = "cuda"
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)


class AVResidualAdapter(nn.Module):
    """Learn a small visual correction to Whisper encoder features."""

    def __init__(self, audio_dim=768, hidden_dim=256, video_dim=128):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.video_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 4 * 4, video_dim),
            nn.ReLU(),
        )
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 4, batch_first=True
        )
        self.out_proj = nn.Linear(hidden_dim, audio_dim)

        # Start near the audio-only model, then learn the visual correction.
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, audio_hidden, video):
        # audio_hidden: [1, T, 768]
        # video: [1, 16, 48, 48]
        audio_q = self.audio_proj(audio_hidden)

        b, t, h, w = video.shape
        vf = self.video_encoder(video.reshape(b * t, 1, h, w))
        vf = self.video_proj(vf.reshape(b, t, -1))

        attended, _ = self.cross_attention(
            query=audio_q, key=vf, value=vf, need_weights=False
        )

        correction = self.out_proj(attended)
        return audio_hidden + self.gate * correction


df = pd.read_csv(MANIFEST)

df["feature"] = df["whisper_video"].map(
    lambda p: os.path.join(
        "outputs/train_video_features",
        os.path.basename(str(p)) + ".npy",
    )
)

required = ["whisper_audio", "feature", "transcript"]
df = df.dropna(subset=required).copy()
df = df[df["transcript"].astype(str).str.strip().ne("")]

df = df[
    df["whisper_audio"].map(os.path.isfile)
    & df["feature"].map(os.path.isfile)
].reset_index(drop=True)

if len(df) == 0:
    raise RuntimeError("No complete training audio/video pairs found.")

df = df.sample(
    n=min(MAX_TRAIN_ROWS, len(df)), random_state=42
).reset_index(drop=True)

print("Device:", DEVICE)
print("Expanded training rows:", len(df), flush=True)

processor = WhisperProcessor.from_pretrained(ADAPTER)
base = WhisperForConditionalGeneration.from_pretrained(
    BASE, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE)
model.eval()

# Freeze Whisper and LoRA; train only the new audiovisual adapter.
for p in model.parameters():
    p.requires_grad = False

av = AVResidualAdapter(
    audio_dim=model.config.d_model
).to(DEVICE)

optimizer = torch.optim.AdamW(av.parameters(), lr=LR)
losses = []
skipped = 0

for i, row in df.iterrows():
    try:
        audio, sr = sf.read(row["whisper_audio"])
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.nan_to_num(audio.astype(np.float32))

        if sr != 16000:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=16000
            ).astype(np.float32)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(DEVICE)

        video_np = np.load(row["feature"]).astype(np.float32)
        if video_np.shape != (16, 48, 48):
            raise ValueError(
                "Unexpected video feature shape: {}".format(video_np.shape)
            )
        if not np.isfinite(video_np).all():
            raise ValueError("Non-finite video feature")

        video = torch.from_numpy(video_np).unsqueeze(0).to(DEVICE)

        text = str(row["transcript"]).strip()
        labels = processor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TARGET_TOKENS,
        ).input_ids.to(DEVICE)

        # Whisper shifts labels internally; avoid duplicating its start token.
        if labels.shape[1] > 0 and (
            labels[0, 0].item() == model.config.decoder_start_token_id
        ):
            labels = labels[:, 1:]

        if labels.shape[1] == 0:
            skipped += 1
            continue

        with torch.no_grad():
            audio_hidden = model.get_encoder()(
                input_features=inputs
            ).last_hidden_state

        fused_hidden = av(audio_hidden, video)

        # Frozen Whisper decoder computes transcript cross-entropy.
        # Gradients still flow through its decoder into the AV adapter.
        output = model(
            input_features=inputs,
            encoder_outputs=BaseModelOutput(
                last_hidden_state=fused_hidden
            ),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        loss = output.loss
        if not torch.isfinite(loss):
            skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.item()))

        if (i + 1) % LOG_EVERY == 0:
            print(
                "Rows {}/{} | successful {} | loss {:.4f} | gate {:.4f}".format(
                    i + 1, len(df), len(losses),
                    losses[-1], av.gate.item()
                ),
                flush=True,
            )

    except Exception as e:
        skipped += 1
        print("Skipping row {}: {}".format(i, str(e)[:250]), flush=True)

    finally:
        if "inputs" in locals():
            del inputs
        if "audio_hidden" in locals():
            del audio_hidden
        if "fused_hidden" in locals():
            del fused_hidden
        if "output" in locals():
            del output
        if "loss" in locals():
            del loss
        if "labels" in locals():
            del labels
        if "video" in locals():
            del video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if not losses:
    raise RuntimeError("No successful training rows; no checkpoint saved.")

os.makedirs("outputs", exist_ok=True)
torch.save(
    {
        "av_adapter": av.state_dict(),
        "training_rows": len(df),
        "successful_rows": len(losses),
        "skipped_rows": skipped,
        "mean_loss": float(np.mean(losses)),
        "final_loss": losses[-1],
        "gate": av.gate.item(),
        "base_model": BASE,
        "lora_adapter": ADAPTER,
    },
    OUT,
)

print("\nPilot finished.")
print("Successful:", len(losses), "| skipped:", skipped)
print("Mean transcript loss:", round(float(np.mean(losses)), 4))
print("Final transcript loss:", round(losses[-1], 4))
print("Learned gate:", round(av.gate.item(), 4))
print("Saved:", OUT, flush=True)
