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
OUT = ROOT / "outputs/av_transcription_aishell6_2k.pt"
EVAL_DIR = ROOT / "outputs/evaluation"

LEARNING_RATE = 1e-4
MAX_TARGET_TOKENS = 128
SAVE_EVERY = 250

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)
print("LoRA adapter:", LORA_DIR)
print("AV manifest:", MANIFEST)
print("New AV checkpoint:", OUT)

if not MANIFEST.is_file():
    raise FileNotFoundError(MANIFEST)
if not (LORA_DIR / "adapter_config.json").is_file():
    raise FileNotFoundError(f"Missing fine-tuned LoRA adapter: {LORA_DIR}")
if OUT.exists():
    raise FileExistsError(
        f"{OUT} already exists. Existing checkpoint was not overwritten."
    )

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

def usable(row):
    return (
        row.get("split") in ("train", "valid")
        and row.get("transcript", "").strip()
        and Path(row.get("audio_path", "")).is_file()
        and Path(row.get("video_path", "")).is_file()
    )

rows = [r for r in rows if usable(r)]
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]

if not train_rows or not valid_rows:
    raise RuntimeError("No complete train or validation audio/video pairs.")

print("Matched training pairs:", len(train_rows))
print("Matched validation pairs:", len(valid_rows))

# Use exactly the AVResidualAdapter architecture expected by app.py.
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
        batch, frames, height, width = video.shape
        features = self.video_encoder(
            video.reshape(batch * frames, 1, height, width)
        )
        features = self.video_proj(
            features.reshape(batch, frames, -1)
        )
        attended, _ = self.cross_attention(
            audio_q, features, features, need_weights=False
        )
        return audio_hidden + self.gate * self.out_proj(attended)


processor = WhisperProcessor.from_pretrained(str(LORA_DIR))
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    low_cpu_mem_usage=True,
)
whisper = PeftModel.from_pretrained(base, str(LORA_DIR)).to(DEVICE).eval()

# Freeze Whisper and LoRA. Only the newly initialized AV adapter is trained.
for param in whisper.parameters():
    param.requires_grad = False

av = AVResidualAdapter(audio_dim=whisper.config.d_model).to(DEVICE)
optimizer = torch.optim.AdamW(av.parameters(), lr=LEARNING_RATE)
converter = OpenCC("t2s")

def normalize(text):
    text = converter.convert(unicodedata.normalize("NFKC", text).lower())
    return "".join(ch for ch in text if ch.isalnum())

def load_features(audio_path):
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio).astype(np.float32)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")
    if len(audio) / sr > 120:
        raise ValueError(f"Audio exceeds 120 seconds: {audio_path}")
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)

    return processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

def load_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            indices = np.linspace(0, total - 1, 16).astype(int)
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                frames.append(gray.astype(np.float32) / 255.0)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    while len(frames) < 16:
        frames.append(frames[-1].copy())

    array = np.stack(frames[:16])
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite video features: {video_path}")

    return torch.from_numpy(array).unsqueeze(0).to(DEVICE)

def make_labels(text):
    ids = processor.tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TARGET_TOKENS,
    ).input_ids

    # Whisper's decoder shifts the target internally.
    if ids.shape[1] and (
        ids[0, 0].item() == whisper.config.decoder_start_token_id
    ):
        ids = ids[:, 1:]

    if ids.shape[1] == 0:
        raise ValueError("Empty transcript token sequence")

    return ids.to(DEVICE)

def run_audio_only(row):
    features = load_features(row["audio_path"])
    with torch.inference_mode():
        tokens = whisper.generate(
            input_features=features,
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
            num_beams=1,
        )
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    del features, tokens
    return text

def run_audio_video(row):
    features = load_features(row["audio_path"])
    video = load_video(row["video_path"])

    with torch.inference_mode():
        hidden = whisper.get_encoder(
        )(input_features=features).last_hidden_state
        fused = av(hidden, video)
        tokens = whisper.generate(
            input_features=features,
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
            num_beams=1,
        )

    text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    del features, video, hidden, fused, tokens
    return text

def save_checkpoint(path, step, loss_values):
    torch.save(
        {
            "av_adapter": av.state_dict(),
            "training_rows": len(train_rows),
            "validation_rows": len(valid_rows),
            "completed_steps": step,
            "mean_training_loss": float(np.mean(loss_values)),
            "base_model": "openai/whisper-small",
            "lora_adapter": str(LORA_DIR.relative_to(ROOT)),
            "learning_rate": LEARNING_RATE,
        },
        path,
    )
    print("Saved recovery checkpoint:", path, flush=True)

def evaluate(rows_to_test, mode, output_csv):
    av.eval()
    predictions = []
    edit_sum = 0
    ref_sum = 0
    exact = 0

    print(f"\nEvaluating {mode} on {len(rows_to_test)} validation pairs...")
    for index, row in enumerate(rows_to_test, start=1):
        prediction = (
            run_audio_only(row)
            if mode == "audio-only"
            else run_audio_video(row)
        )
        reference = row["transcript"]
        ref = normalize(reference)
        pred = normalize(prediction)

        edit_sum += editdistance.eval(ref, pred)
        ref_sum += len(ref)
        exact += int(ref == pred)

        predictions.append({
            "audio_path": row["audio_path"],
            "video_path": row["video_path"],
            "reference": reference,
            "prediction": prediction,
            "cer": editdistance.eval(ref, pred) / max(1, len(ref)),
        })

        if index % 25 == 0 or index == len(rows_to_test):
            print(f"  {mode}: {index}/{len(rows_to_test)}", flush=True)

        if index % 10 == 0:
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    corpus_cer = edit_sum / max(1, ref_sum)
    exact_rate = exact / max(1, len(rows_to_test))
    print(f"{mode} corpus CER: {100 * corpus_cer:.2f}%")
    print(f"{mode} exact-match rate: {100 * exact_rate:.2f}%")

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    return corpus_cer, exact_rate

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Establish audio-only reference performance using the new LoRA model.
audio_cer, audio_exact = evaluate(
    valid_rows,
    "audio-only",
    EVAL_DIR / "aishell6_2k_audio_only.csv",
)

av.train()
losses = []
skipped = 0
print("\nStarting AV adapter training...", flush=True)

for step, row in enumerate(train_rows, start=1):
    features = video = labels = hidden = fused = result = loss = None
    try:
        features = load_features(row["audio_path"])
        video = load_video(row["video_path"])
        labels = make_labels(row["transcript"])

        # The Whisper encoder is frozen; compute its representation without
        # gradients, then train the AV correction through the frozen decoder.
        with torch.no_grad():
            hidden = whisper.get_encoder(
            )(input_features=features).last_hidden_state

        fused = av(hidden, video)
        result = whisper(
            input_features=features,
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        loss = result.loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        if step % 25 == 0 or step == len(train_rows):
            print(
                f"Training {step}/{len(train_rows)} | "
                f"loss={losses[-1]:.4f} | "
                f"mean_loss={np.mean(losses):.4f} | "
                f"gate={av.gate.item():.4f}",
                flush=True,
            )

        if step % SAVE_EVERY == 0:
            save_checkpoint(
                ROOT / "outputs" / f"av_aishell6_2k_step_{step}.pt",
                step,
                losses,
            )

    except Exception as exc:
        skipped += 1
        print(f"Skipping training row {step}: {type(exc).__name__}: {exc}",
              flush=True)

    finally:
        del features, video, labels, hidden, fused, result, loss
        if step % 10 == 0:
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

if not losses:
    raise RuntimeError("No successful AV training rows. No final checkpoint saved.")

print(f"\nTraining completed: {len(losses)} successful, {skipped} skipped.")
save_checkpoint(OUT, len(train_rows), losses)

# Evaluate the trained AV adapter on the same validation pairs.
av_cer, av_exact = evaluate(
    valid_rows,
    "audio-visual",
    EVAL_DIR / "aishell6_2k_av_validation.csv",
)

summary = EVAL_DIR / "aishell6_2k_av_summary.txt"
summary.write_text(
    f"Device: {DEVICE}\n"
    f"Training pairs: {len(train_rows)}\n"
    f"Validation pairs: {len(valid_rows)}\n"
    f"Successful AV training steps: {len(losses)}\n"
    f"Skipped training steps: {skipped}\n"
    f"Mean AV training loss: {np.mean(losses):.4f}\n"
    f"Audio-only validation CER: {audio_cer * 100:.2f}%\n"
    f"Audio-visual validation CER: {av_cer * 100:.2f}%\n"
    f"Audio-only exact-match rate: {audio_exact * 100:.2f}%\n"
    f"Audio-visual exact-match rate: {av_exact * 100:.2f}%\n"
    f"New checkpoint: {OUT}\n",
    encoding="utf-8",
)

print("\nAV TRAINING COMPLETE")
print(f"Audio-only CER: {audio_cer * 100:.2f}%")
print(f"Audio-visual CER: {av_cer * 100:.2f}%")
print("New AV checkpoint:", OUT)
print("Summary:", summary)
print("Original AV checkpoint and LoRA adapters were not modified.")
