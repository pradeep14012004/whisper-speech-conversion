import os
import unicodedata
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "openai/whisper-small"
ADAPTER = "outputs/whisper_lora_small"
CHECKPOINT = "outputs/av_transcription_full.pt"
MANIFEST = "outputs/verified_av_manifest.csv"
OUT = "outputs/evaluation/verified_av_validation_predictions.csv"

class AVResidualAdapter(nn.Module):
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
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, audio_hidden, video):
        audio_q = self.audio_proj(audio_hidden)
        b, t, h, w = video.shape
        vf = self.video_encoder(video.reshape(b * t, 1, h, w))
        vf = self.video_proj(vf.reshape(b, t, -1))
        attended, _ = self.cross_attention(
            query=audio_q, key=vf, value=vf, need_weights=False
        )
        return audio_hidden + self.gate * self.out_proj(attended)

def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(c for c in text if c.isalnum())

def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + 1, prev[j] + 1,
                prev[j - 1] + (ca != cb)
            ))
        prev = cur
    return prev[-1]

def corpus_cer(refs, hyps):
    errors = chars = 0
    for ref, hyp in zip(refs, hyps):
        r, h = normalize(ref), normalize(hyp)
        errors += edit_distance(r, h)
        chars += len(r)
    return errors / max(chars, 1)

def save_results(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(CHECKPOINT)

df = pd.read_csv(MANIFEST)
df = df[df["split"] == "valid"].copy()
df = df.dropna(subset=["whisper_audio", "feature_path", "transcript"])
df = df[
    df["whisper_audio"].map(os.path.isfile)
    & df["feature_path"].map(os.path.isfile)
].reset_index(drop=True)

print("Device:", DEVICE, "| validation clips:", len(df), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.isfile(OUT):
    old = pd.read_csv(OUT)
    rows = old.to_dict("records")
    done = set(old["whisper_audio"].astype(str))
    print("Resuming from", len(done), "saved predictions", flush=True)
else:
    rows, done = [], set()

processor = WhisperProcessor.from_pretrained(ADAPTER)
base = WhisperForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE).eval()

ckpt = torch.load(CHECKPOINT, map_location="cpu")
av = AVResidualAdapter(
    audio_dim=model.config.d_model,
    hidden_dim=256,
    video_dim=128,
).to(DEVICE)
av.load_state_dict(ckpt["av_adapter"])
av.eval()

for idx, row in df.iterrows():
    audio_path = str(row["whisper_audio"])
    if audio_path in done:
        continue

    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.nan_to_num(audio.astype(np.float32))
        if sr != 16000:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=16000
            ).astype(np.float32)

        video_np = np.load(row["feature_path"]).astype(np.float32)
        if video_np.shape != (16, 48, 48):
            raise ValueError("Unexpected video shape: {}".format(video_np.shape))
        if not np.isfinite(video_np).all():
            raise ValueError("Video features contain non-finite values")

        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)
        video = torch.from_numpy(video_np).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            audio_hidden = model.get_encoder()(
                input_features=inputs
            ).last_hidden_state
            fused_hidden = av(audio_hidden, video)

            baseline_tokens = model.generate(
                inputs, language="chinese", task="transcribe",
                max_new_tokens=128
            )
            av_tokens = model.generate(
                inputs,
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=fused_hidden
                ),
                language="chinese", task="transcribe",
                max_new_tokens=128
            )

        baseline_text = processor.batch_decode(
            baseline_tokens, skip_special_tokens=True
        )[0]
        av_text = processor.batch_decode(
            av_tokens, skip_special_tokens=True
        )[0]

        rows.append({
            "split": "valid",
            "speaker_id": row["speaker_id"],
            "whisper_audio": audio_path,
            "whisper_video": row["whisper_video"],
            "reference": str(row["transcript"]),
            "audio_only_prediction": baseline_text,
            "av_prediction": av_text,
        })
        done.add(audio_path)

        if len(rows) % 10 == 0:
            save_results(rows)
            print("Saved {}/{} predictions".format(
                len(done), len(df)
            ), flush=True)

        del inputs, video, audio_hidden, fused_hidden
        del baseline_tokens, av_tokens

    except Exception as e:
        print("Skipped row {}: {}".format(
            idx, str(e)[:250]
        ), flush=True)

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

save_results(rows)

if not rows:
    raise RuntimeError("No predictions were generated.")

results = pd.DataFrame(rows)
print("\nSuccessful predictions:", len(results))
print("Audio-only corpus CER: {:.4%}".format(
    corpus_cer(results["reference"], results["audio_only_prediction"])
))
print("AV corpus CER:         {:.4%}".format(
    corpus_cer(results["reference"], results["av_prediction"])
))
print("Saved predictions:", OUT)
