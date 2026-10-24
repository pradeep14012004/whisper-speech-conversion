import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from av_fusion import AudioVisualFusion

BASE = "openai/whisper-small"
ADAPTER = "outputs/whisper_lora_small"
MANIFEST = "outputs/training_manifest.csv"
OUT = "outputs/av_fusion.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1
LR = 1e-4

torch.manual_seed(42)
np.random.seed(42)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check DGX GPU allocation.")

df = pd.read_csv(MANIFEST)

required = [
    "whisper_audio", "normal_audio", "feature", "transcript"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Missing manifest columns: {}".format(missing))

df = df.dropna(subset=required).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep only complete pairs; use the entire remaining manifest.
valid_rows = []
for _, row in df.iterrows():
    if all(os.path.isfile(str(row[c])) for c in
           ["whisper_audio", "normal_audio", "feature"]):
        valid_rows.append(row)

df = pd.DataFrame(valid_rows).reset_index(drop=True)

if len(df) == 0:
    raise RuntimeError("No complete audio-video pairs found.")

print("Device:", DEVICE)
print("Full usable dataset:", len(df), "pairs", flush=True)

processor = WhisperProcessor.from_pretrained(ADAPTER)

base = WhisperForConditionalGeneration.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

whisper = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE)
whisper.eval()

for p in whisper.parameters():
    p.requires_grad = False

fusion = AudioVisualFusion(
    audio_dim=whisper.config.d_model,
    video_dim=128,
    hidden_dim=256,
    heads=4
).to(DEVICE)

# Map fused representations back to the frozen teacher's
# normal-speech encoder representation space.
decoder = nn.Sequential(
    nn.Linear(256, 512),
    nn.GELU(),
    nn.Linear(512, whisper.config.d_model)
).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(fusion.parameters()) + list(decoder.parameters()),
    lr=LR
)

def load_audio(path):
    x, sr = sf.read(path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.nan_to_num(x.astype(np.float32))
    duration = len(x) / float(sr)

    if sr != 16000:
        x = librosa.resample(
            x, orig_sr=sr, target_sr=16000
        ).astype(np.float32)

    return x, duration

def encode_audio(x):
    features = processor(
        x,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        hidden = whisper.get_encoder()(
            input_features=features
        ).last_hidden_state

    return hidden

os.makedirs("outputs", exist_ok=True)
best_loss = float("inf")

for epoch in range(EPOCHS):
    fusion.train()
    decoder.train()
    losses = []

    for i, row in df.iterrows():
        try:
            x, x_duration = load_audio(row["whisper_audio"])
            y, y_duration = load_audio(row["normal_audio"])

            # Frozen Whisper encoder representations.
            audio_emb = encode_audio(x)
            with torch.no_grad():
                target_emb = encode_audio(y)

            vf = np.load(row["feature"]).astype(np.float32)
            if vf.ndim == 3:
                vf = torch.from_numpy(vf).unsqueeze(0).to(DEVICE)
            elif vf.ndim == 4:
                vf = torch.from_numpy(vf).to(DEVICE)
            else:
                raise ValueError(
                    "Unexpected video feature shape: {}".format(vf.shape)
                )

            # Predict normal-speech representations at matching
            # encoder time positions. Ignore padded audio regions.
            fused = fusion(audio_emb, vf)
            pred = decoder(fused)

            T = min(pred.shape[1], target_emb.shape[1])
            pred = pred[:, :T, :]
            target = target_emb[:, :T, :]

            # Whisper encoder has approximately 50 time steps/second.
            valid_seconds = min(x_duration, y_duration, 30.0)
            valid_T = max(1, min(T, int(valid_seconds * 50)))

            loss = torch.nn.functional.smooth_l1_loss(
                pred[:, :valid_T, :],
                target[:, :valid_T, :]
            )

            if not torch.isfinite(loss):
                print("Skipping non-finite loss at row", i, flush=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fusion.parameters()) +
                list(decoder.parameters()), 1.0
            )
            optimizer.step()

            losses.append(float(loss.item()))

            if (i + 1) % 25 == 0:
                print(
                    "Epoch {}/{} | {}/{} | loss {:.5f}".format(
                        epoch + 1, EPOCHS, i + 1, len(df),
                        losses[-1]
                    ),
                    flush=True
                )

        except Exception as e:
            print(
                "Row {} skipped: {}".format(i, repr(e)),
                flush=True
            )

    if not losses:
        raise RuntimeError("No successful training batches.")

    mean_loss = float(np.mean(losses))
    print(
        "Epoch {} | successful {} / {} | mean loss {:.6f}".format(
            epoch + 1, len(losses), len(df), mean_loss
        ),
        flush=True
    )

    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save({
            "fusion": fusion.state_dict(),
            "decoder": decoder.state_dict(),
            "audio_dim": whisper.config.d_model,
            "hidden_dim": 256,
            "video_dim": 128,
            "epoch": epoch + 1,
            "mean_loss": mean_loss,
            "dataset_rows": len(df),
            "successful_rows": len(losses)
        }, OUT)
        print("Checkpoint saved:", OUT, flush=True)

print("Training finished.")
print("Best training loss:", best_loss)
print("Checkpoint:", OUT)
