from pathlib import Path
import csv
import gc
import random
import unicodedata

import editdistance
import librosa
import numpy as np
import soundfile as sf
import torch
from opencc import OpenCC
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/aishell6_lora_2k/manifest.csv"
INITIAL_ADAPTER = ROOT / "outputs/whisper_lora_aishell6_pilot"
OUTPUT_DIR = ROOT / "outputs/whisper_lora_aishell6_2k"

SEED = 42
LEARNING_RATE = 5e-6
MAX_TARGET_TOKENS = 128
SAVE_EVERY = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)
print("Training manifest:", MANIFEST)
print("Starting adapter:", INITIAL_ADAPTER)
print("New adapter destination:", OUTPUT_DIR)

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
if not INITIAL_ADAPTER.is_dir():
    raise FileNotFoundError(f"Missing existing adapter: {INITIAL_ADAPTER}")
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. Rename or back up that directory "
        "before starting another run; the existing output will not be overwritten."
    )

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]

for split_name, split_rows in [("train", train_rows), ("valid", valid_rows)]:
    for row in split_rows:
        if not Path(row["audio_path"]).is_file():
            raise FileNotFoundError(row["audio_path"])
        if not row["transcript"].strip():
            raise ValueError(f"Empty transcript in {split_name} data")

if not train_rows or not valid_rows:
    raise ValueError("The training or validation split is empty.")

print(f"Training examples: {len(train_rows)}")
print(f"Validation examples: {len(valid_rows)}")

processor = WhisperProcessor.from_pretrained(str(INITIAL_ADAPTER))
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(
    base_model,
    str(INITIAL_ADAPTER),
    is_trainable=True,
).to(DEVICE)

# Train only the LoRA parameters. Keep the base model weights frozen.
for name, param in model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False

trainable = [p for p in model.parameters() if p.requires_grad]
if not trainable:
    raise RuntimeError("No trainable LoRA parameters were found.")

print("Trainable parameters:", sum(p.numel() for p in trainable))
model.config.use_cache = False
optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

converter = OpenCC("t2s")

def normalize(text):
    text = converter.convert(unicodedata.normalize("NFKC", text).lower())
    return "".join(ch for ch in text if ch.isalnum())

def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio).astype(np.float32)

    if not len(audio):
        raise ValueError(f"Empty audio: {path}")
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)

    return audio

def encode_audio(path):
    audio = load_audio(path)
    result = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    return result.input_features.to(DEVICE)

def make_labels(text):
    labels = processor.tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TARGET_TOKENS,
    ).input_ids

    # Whisper shifts labels internally; remove a leading decoder-start token.
    if (
        labels.shape[1]
        and labels[0, 0].item() == model.config.decoder_start_token_id
    ):
        labels = labels[:, 1:]

    labels = labels.to(DEVICE)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    if labels.shape[1] == 0:
        raise ValueError("Transcript produced empty target tokens.")
    return labels

def evaluate(eval_rows, label):
    model.eval()
    distances = 0
    reference_chars = 0
    exact_matches = 0
    saved_predictions = []

    print(f"\nEvaluating {label} ({len(eval_rows)} samples)...")

    with torch.inference_mode():
        for i, row in enumerate(eval_rows, start=1):
            features = encode_audio(row["audio_path"])
            tokens = model.generate(
                input_features=features,
                language="chinese",
                task="transcribe",
                max_new_tokens=128,
                num_beams=1,
            )
            prediction = processor.batch_decode(
                tokens, skip_special_tokens=True
            )[0].strip()

            ref = normalize(row["transcript"])
            pred = normalize(prediction)

            distances += editdistance.eval(ref, pred)
            reference_chars += len(ref)
            exact_matches += int(ref == pred)

            saved_predictions.append({
                "audio_path": row["audio_path"],
                "reference": row["transcript"],
                "prediction": prediction,
            })

            if i % 10 == 0 or i == len(eval_rows):
                print(f"  Evaluated {i}/{len(eval_rows)}", flush=True)

            del features, tokens

    cer = distances / max(1, reference_chars)
    exact_rate = exact_matches / max(1, len(eval_rows))

    print(f"{label} CER: {cer * 100:.2f}%")
    print(f"{label} exact-match rate: {exact_rate * 100:.2f}%")

    model.train()
    return cer, exact_rate, saved_predictions

# Evaluate the original adapter before training.
initial_cer, initial_exact, initial_predictions = evaluate(
    valid_rows, "Before fine-tuning"
)

# Preserve the exact 500-example pilot; randomize its order reproducibly.
random.shuffle(train_rows)
model.train()

losses = []
OUTPUT_DIR.mkdir(parents=True)

print("\nStarting one training pass...")

for i, row in enumerate(train_rows, start=1):
    features = encode_audio(row["audio_path"])
    labels = make_labels(row["transcript"])

    optimizer.zero_grad(set_to_none=True)
    result = model(
        input_features=features,
        labels=labels,
        use_cache=False,
    )
    loss = result.loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at training sample {i}; stopping."
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()

    losses.append(float(loss.detach().cpu()))

    if i % 25 == 0 or i == len(train_rows):
        print(
            f"Training {i}/{len(train_rows)} | "
            f"loss={losses[-1]:.4f} | "
            f"mean_loss={np.mean(losses):.4f}",
            flush=True,
        )

    del features, labels, result, loss

    if i % SAVE_EVERY == 0 or i == len(train_rows):
        checkpoint_dir = OUTPUT_DIR / f"checkpoint_step_{i}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        print("Saved recovery checkpoint:", checkpoint_dir)

    if i % 25 == 0:
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

# Save the final adapter into the new directory, not the original one.
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Evaluate the fine-tuned adapter on the exact same validation set.
final_cer, final_exact, final_predictions = evaluate(
    valid_rows, "After fine-tuning"
)

report_dir = ROOT / "outputs" / "evaluation"
report_dir.mkdir(parents=True, exist_ok=True)

for filename, predictions in [
    ("aishell6_2k_before.csv", initial_predictions),
    ("aishell6_2k_after.csv", final_predictions),
]:
    with (report_dir / filename).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "reference", "prediction"]
        )
        writer.writeheader()
        writer.writerows(predictions)

summary = report_dir / "aishell6_2k_summary.txt"
summary.write_text(
    f"Device: {DEVICE}\n"
    f"Training examples: {len(train_rows)}\n"
    f"Validation examples: {len(valid_rows)}\n"
    f"Learning rate: {LEARNING_RATE}\n"
    f"Training passes: 1\n"
    f"Mean training loss: {np.mean(losses):.4f}\n"
    f"Initial validation CER: {initial_cer * 100:.2f}%\n"
    f"Final validation CER: {final_cer * 100:.2f}%\n"
    f"Initial exact-match rate: {initial_exact * 100:.2f}%\n"
    f"Final exact-match rate: {final_exact * 100:.2f}%\n"
    f"New adapter: {OUTPUT_DIR}\n",
    encoding="utf-8",
)

print("\nAISHELL6 2K FINE-TUNING COMPLETE")
print(f"Validation CER: {initial_cer*100:.2f}% -> {final_cer*100:.2f}%")
print(f"New adapter saved to: {OUTPUT_DIR}")
print(f"Evaluation summary: {summary}")
print("The original adapter was not modified.")
