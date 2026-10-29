from pathlib import Path
import csv
import gc
import unicodedata

import cv2
import editdistance
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from opencc import OpenCC
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/aishell6_lora_2k/av_manifest.csv"
LORA_DIR = ROOT / "outputs/whisper_lora_aishell6_2k"
CHECKPOINT = ROOT / "outputs/av_transcription_aishell6_2k.pt"
OUT = ROOT / "outputs/evaluation/aishell6_2k_av_scale_sweep.csv"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
converter = OpenCC("t2s")

class AVResidualAdapter(nn.Module):
    def __init__(self, audio_dim=768, hidden_dim=256, video_dim=128):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(start_dim=1),
            nn.Linear(256, video_dim),
            nn.ReLU(),
        )
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 4, batch_first=True
        )
        self.out_proj = nn.Linear(hidden_dim, audio_dim)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, audio_hidden, video):
        audio_q = self.audio_proj(audio_hidden)
        b, frames, h, w = video.shape
        features = self.video_encoder(
            video.reshape(b * frames, 1, h, w)
        )
        features = self.video_proj(features.reshape(b, frames, -1))
        attended, _ = self.cross_attention(
            audio_q, features, features, need_weights=False
        )
        return audio_hidden + self.gate * self.out_proj(attended)

def normalize(text):
    text = converter.convert(unicodedata.normalize("NFKC", text).lower())
    return "".join(ch for ch in text if ch.isalnum())

def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio):
        raise ValueError(f"Empty audio: {path}")
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)
    return audio

def load_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            for idx in np.linspace(0, total - 1, 16).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, (48, 48))
                    frames.append(frame.astype(np.float32) / 255.0)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames found: {path}")
    while len(frames) < 16:
        frames.append(frames[-1].copy())
    return torch.from_numpy(np.stack(frames[:16])).unsqueeze(0).to(DEVICE)

if not MANIFEST.is_file() or not LORA_DIR.is_dir() or not CHECKPOINT.is_file():
    raise FileNotFoundError("A required manifest, LoRA adapter, or AV checkpoint is missing.")

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
rows = [
    r for r in rows
    if r["split"] == "valid"
    and r["transcript"].strip()
    and Path(r["audio_path"]).is_file()
    and Path(r["video_path"]).is_file()
]
if not rows:
    raise RuntimeError("No valid matched validation rows found.")

print("Device:", DEVICE)
print("Validation pairs:", len(rows))
print("Testing residual scales: 0.25 and 0.50")

processor = WhisperProcessor.from_pretrained(str(LORA_DIR))
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", low_cpu_mem_usage=True
)
whisper = PeftModel.from_pretrained(base, str(LORA_DIR)).to(DEVICE).eval()

av = AVResidualAdapter(audio_dim=whisper.config.d_model)
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
av.load_state_dict(checkpoint["av_adapter"])
av = av.to(DEVICE).eval()

results = []

for i, row in enumerate(rows, start=1):
    audio = load_audio(row["audio_path"])
    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)
    video = load_video(row["video_path"])

    with torch.inference_mode():
        hidden = whisper.get_encoder()(
            input_features=features
        ).last_hidden_state

        full_fused = av(hidden, video)
        residual = full_fused - hidden

        for scale in (0.25, 0.50):
            scaled_hidden = hidden + scale * residual
            tokens = whisper.generate(
                input_features=features,
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=scaled_hidden
                ),
                language="chinese",
                task="transcribe",
                max_new_tokens=128,
                num_beams=1,
            )
            prediction = processor.batch_decode(
                tokens, skip_special_tokens=True
            )[0].strip()

            reference_norm = normalize(row["transcript"])
            prediction_norm = normalize(prediction)
            edits = editdistance.eval(reference_norm, prediction_norm)

            results.append({
                "audio_path": row["audio_path"],
                "scale": scale,
                "reference": row["transcript"],
                "prediction": prediction,
                "edits": edits,
                "reference_chars": len(reference_norm),
                "exact_match": int(reference_norm == prediction_norm),
            })

            del tokens, scaled_hidden

    if i % 25 == 0 or i == len(rows):
        print(f"Processed {i}/{len(rows)} validation pairs", flush=True)

    del audio, features, video, hidden, full_fused, residual
    if i % 10 == 0:
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\n=== RESIDUAL SCALE RESULTS ===")
for scale in (0.25, 0.50):
    subset = [r for r in results if r["scale"] == scale]
    edits = sum(r["edits"] for r in subset)
    chars = sum(r["reference_chars"] for r in subset)
    exact = sum(r["exact_match"] for r in subset)
    print(
        f"Scale {scale:.2f}: CER {100 * edits / max(1, chars):.2f}% | "
        f"exact matches {exact}/{len(subset)}"
    )

print("Existing audio-only CER baseline: 24.03%")
print("Existing full AV CER baseline: 25.61%")
print("Saved:", OUT)
