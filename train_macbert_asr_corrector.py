#!/usr/bin/env python3
"""Train MacBERT for the project's own Whisper ASR errors.

The script reads the 2k LoRA train/valid manifest, runs the original LoRA+AV
path where matching videos are available and the new 2k LoRA audio-only path,
then fine-tunes MacBERT on TRAIN-split prediction/reference pairs only.
Validation references are used only for reporting, never for optimization.

Important limitation: MacBERT CSC emits one token per input-token position. To
avoid corrupting alignment, training uses examples whose source/reference have
the same character length and tokenizer length. Thus this version targets
substitution-style errors; it cannot reliably learn insertions/deletions.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from opencc import OpenCC
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "data" / "aishell6_lora_2k" / "manifest.csv"
DEFAULT_AV_MANIFEST = ROOT / "data" / "aishell6_lora_2k" / "av_manifest.csv"
OLD_LORA_DIR = ROOT / "outputs" / "whisper_lora_small"
NEW_LORA_DIR = ROOT / "outputs" / "whisper_lora_aishell6_2k"
OLD_AV_CHECKPOINT = ROOT / "outputs" / "av_transcription_full.pt"
BASE_CORRECTOR = "shibing624/macbert4csc-base-chinese"
OUT_DIR = ROOT / "outputs" / "macbert_asr_corrector"
CANDIDATE_DIR = ROOT / "outputs" / "macbert_asr_corrector_candidate"
EVAL_DIR = ROOT / "outputs" / "evaluation"
TRAIN_PRED_CSV = EVAL_DIR / "macbert_asr_corrector_train_predictions.csv"
VALID_PRED_CSV = EVAL_DIR / "macbert_asr_corrector_valid_predictions.csv"
VALID_RESULTS_CSV = EVAL_DIR / "macbert_asr_corrector_validation_results.csv"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CC = OpenCC("t2s")
MAX_AUDIO_SECONDS = 120
MAX_NEW_TOKENS = 128


def normalise_text(text: str) -> str:
    text = CC.convert(unicodedata.normalize("NFKC", text or "").lower())
    return "".join(ch for ch in text if ch.isalnum())


def edit_distance(a: str, b: str) -> int:
    """Small Levenshtein implementation to avoid an extra package dependency."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def find_value(row: Dict[str, str], candidates: List[str], required=True) -> str:
    lower = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for name in candidates:
        value = lower.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    if required:
        raise KeyError(
            f"Could not find any of columns {candidates}; manifest columns: {list(row.keys())}"
        )
    return ""


def normalise_split(value: str) -> str:
    value = (value or "").strip().lower()
    if value in {"val", "valid", "validation", "dev", "development"}:
        return "valid"
    if value in {"train", "training"}:
        return "train"
    if value in {"test", "testing"}:
        return "test"
    return value


def make_path_index() -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    folders = [ROOT / "data" / "aishell6_lora_2k", ROOT / "data"]
    for folder in folders:
        if folder.exists():
            for pattern in ("*.wav", "*.avi", "*.mp4", "*.mov", "*.mkv"):
                for p in folder.rglob(pattern):
                    index.setdefault(p.name.lower(), p.resolve())
    return index


def resolve_path(raw: str, path_index: Dict[str, Path]) -> Optional[Path]:
    if not raw:
        return None
    raw = os.path.expanduser(raw.strip())
    candidate = Path(raw)
    choices = [candidate]
    if not candidate.is_absolute():
        choices.append(ROOT / candidate)
    for choice in choices:
        try:
            if choice.is_file():
                return choice.resolve()
        except OSError:
            pass
    return path_index.get(candidate.name.lower())


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Manifest contains no data rows: {path}")
    return rows


def load_manifests(manifest_path: Path, av_manifest_path: Path):
    index = make_path_index()
    base_records = []
    for row in read_rows(manifest_path):
        split = normalise_split(find_value(row, ["split", "partition", "set", "subset"]))
        if split not in {"train", "valid"}:
            continue
        audio_raw = find_value(row, ["audio_path", "audio", "wav_path", "whisper_audio", "path"])
        reference = find_value(row, ["transcript", "reference", "text", "sentence", "reference_text", "label"])
        audio_path = resolve_path(audio_raw, index)
        if audio_path is None:
            raise FileNotFoundError(
                f"Could not locate audio file {audio_raw!r}. Check extracted audio files and manifest paths."
            )
        video_raw = find_value(row, ["video_path", "video", "avi_path"], required=False)
        video_path = resolve_path(video_raw, index) if video_raw else None
        base_records.append({
            "split": split,
            "audio_path": str(audio_path),
            "video_path": str(video_path) if video_path else "",
            "reference": CC.convert(reference.strip()),
        })

    # Merge video paths from AV manifest by full resolved path or filename.
    if av_manifest_path.is_file():
        video_map: Dict[str, Path] = {}
        for row in read_rows(av_manifest_path):
            audio_raw = find_value(row, ["audio_path", "audio", "wav_path", "whisper_audio", "path"], required=False)
            video_raw = find_value(row, ["video_path", "video", "avi_path"], required=False)
            if not audio_raw or not video_raw:
                continue
            audio_path = resolve_path(audio_raw, index)
            video_path = resolve_path(video_raw, index)
            if audio_path and video_path:
                video_map[str(audio_path.resolve())] = video_path
                video_map[audio_path.name.lower()] = video_path
        for rec in base_records:
            if not rec["video_path"]:
                rec["video_path"] = str(
                    video_map.get(str(Path(rec["audio_path"]).resolve()))
                    or video_map.get(Path(rec["audio_path"]).name.lower())
                    or ""
                )
    return base_records


def load_audio_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if len(audio) / sr > MAX_AUDIO_SECONDS:
        raise ValueError(f"Audio is longer than {MAX_AUDIO_SECONDS}s: {path}")
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    return np.nan_to_num(audio).astype(np.float32)


def extract_video_frames(path: str, num_frames=16) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {path}")
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            positions = np.linspace(0, total - 1, num_frames).astype(int)
            for position in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                frames.append(gray.astype(np.float32) / 255.0)
        else:
            while len(frames) < num_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                frames.append(gray.astype(np.float32) / 255.0)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No video frames found: {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:num_frames])


class SpeechPredictors:
    def __init__(self):
        for p in (OLD_LORA_DIR, NEW_LORA_DIR):
            if not p.is_dir():
                raise FileNotFoundError(f"Required LoRA directory missing: {p}")
        self.old_processor = WhisperProcessor.from_pretrained(str(OLD_LORA_DIR))
        old_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.old_model = PeftModel.from_pretrained(old_base, str(OLD_LORA_DIR)).to(DEVICE).eval()
        self.new_processor = WhisperProcessor.from_pretrained(str(NEW_LORA_DIR))
        new_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.new_model = PeftModel.from_pretrained(new_base, str(NEW_LORA_DIR)).to(DEVICE).eval()

        self.av_adapter = None
        if OLD_AV_CHECKPOINT.is_file():
            # Keep architecture compatible with the existing app's checkpoint.
            import torch.nn as nn
            class AVResidualAdapter(nn.Module):
                def __init__(self, audio_dim=768, hidden_dim=256, video_dim=128):
                    super().__init__()
                    self.audio_proj = nn.Linear(audio_dim, hidden_dim)
                    self.video_encoder = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, padding=1),
                        nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                        nn.Flatten(start_dim=1), nn.Linear(256, video_dim), nn.ReLU(),
                    )
                    self.video_proj = nn.Linear(video_dim, hidden_dim)
                    self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
                    self.out_proj = nn.Linear(hidden_dim, audio_dim)
                    self.gate = nn.Parameter(torch.tensor(0.1))
                def forward(self, audio_hidden, video):
                    batch, frames, height, width = video.shape
                    q = self.audio_proj(audio_hidden)
                    feats = self.video_encoder(video.reshape(batch * frames, 1, height, width))
                    feats = self.video_proj(feats.reshape(batch, frames, -1))
                    attended, _ = self.cross_attention(q, feats, feats, need_weights=False)
                    return audio_hidden + self.gate * self.out_proj(attended)
            checkpoint = torch.load(OLD_AV_CHECKPOINT, map_location="cpu")
            if "av_adapter" in checkpoint:
                self.av_adapter = AVResidualAdapter(
                    audio_dim=self.old_model.config.d_model
                )
                self.av_adapter.load_state_dict(checkpoint["av_adapter"])
                self.av_adapter.to(DEVICE).eval()

    @torch.inference_mode()
    def predict_audio(self, audio_path: str) -> str:
        audio = load_audio_16k(audio_path)
        inputs = self.new_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        tokens = self.new_model.generate(
            input_features=inputs, language="chinese", task="transcribe",
            max_new_tokens=MAX_NEW_TOKENS, num_beams=1,
        )
        return self.new_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def predict_old_av(self, audio_path: str, video_path: str) -> str:
        if self.av_adapter is None:
            return ""
        audio = load_audio_16k(audio_path)
        inputs = self.old_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        frames = extract_video_frames(video_path)
        video = torch.as_tensor(frames, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        hidden = self.old_model.get_encoder()(input_features=inputs).last_hidden_state
        hidden = self.av_adapter(hidden, video)
        tokens = self.old_model.generate(
            input_features=inputs,
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            language="chinese", task="transcribe", max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
        )
        return self.old_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def create_or_load_prediction_csv(records: List[dict], output_csv: Path,
                                  predictors: Optional[SpeechPredictors],
                                  regenerate: bool) -> List[dict]:
    if output_csv.is_file() and not regenerate:
        print(f"Reusing cached predictions: {output_csv}")
        with output_csv.open("r", encoding="utf-8-sig", newline="") as f:
            cached = list(csv.DictReader(f))
        # Use cache only when it covers the requested split and has required fields.
        if cached and all(k in cached[0] for k in ("split", "source_model", "audio_path", "source_text", "reference")):
            return cached
        print("Cached prediction CSV is incompatible; regenerating it.")

    if predictors is None:
        raise RuntimeError("Predictor models were not loaded for prediction generation.")
    candidates = []
    total = len(records)
    for i, rec in enumerate(records, 1):
        audio_path = rec["audio_path"]
        # Use new 2k LoRA predictions for every sample.
        try:
            new_text = predictors.predict_audio(audio_path)
            if new_text:
                candidates.append({
                    "split": rec["split"], "source_model": "new_2k_audio",
                    "audio_path": audio_path, "video_path": rec.get("video_path", ""),
                    "source_text": new_text, "reference": rec["reference"],
                })
        except Exception as exc:
            print(f"[warning] New audio prediction failed for {audio_path}: {type(exc).__name__}: {exc}")
        # Add the old AV prediction where a matching video exists.
        video_path = rec.get("video_path", "")
        if video_path and Path(video_path).is_file() and predictors.av_adapter is not None:
            try:
                old_text = predictors.predict_old_av(audio_path, video_path)
                if old_text:
                    candidates.append({
                        "split": rec["split"], "source_model": "original_av",
                        "audio_path": audio_path, "video_path": video_path,
                        "source_text": old_text, "reference": rec["reference"],
                    })
            except Exception as exc:
                print(f"[warning] Original AV prediction failed for {audio_path}: {type(exc).__name__}: {exc}")
        if i % 25 == 0 or i == total:
            print(f"Generated predictions for {i}/{total} {rec['split']} rows", flush=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["split", "source_model", "audio_path", "video_path", "source_text", "reference"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    return candidates


def load_corrector(model_path: str = BASE_CORRECTOR):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    return tokenizer, model


def macbert_predict(tokenizer, model, text: str, max_length=128) -> str:
    if not text or not text.strip():
        return text
    source = CC.convert(unicodedata.normalize("NFKC", text.strip()))
    enc = tokenizer(source, return_tensors="pt", truncation=True, max_length=max_length).to(DEVICE)
    with torch.inference_mode():
        token_ids = model(**enc).logits.argmax(dim=-1)[0]
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True).replace(" ", "").strip()
    corrected = decoded[:len(source)]
    if not corrected:
        return source
    if len(corrected) < len(source):
        corrected += source[len(corrected):]
    return corrected


class CorrectionDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def build_examples(rows, tokenizer, max_length=128):
    examples = []
    seen = set()
    skipped_len = 0
    exact_count = 0
    for row in rows:
        source = CC.convert(unicodedata.normalize("NFKC", row.get("source_text", "").strip()))
        target = CC.convert(unicodedata.normalize("NFKC", row.get("reference", "").strip()))
        if not source or not target:
            continue
        if len(source) != len(target):
            skipped_len += 1
            continue
        src_ids = tokenizer(source, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
        tgt_ids = tokenizer(target, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
        if len(src_ids) != len(tgt_ids) or len(src_ids) > max_length:
            skipped_len += 1
            continue
        # Deduplicate repeated identical source/reference pairs but retain source model variation.
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        if source == target:
            exact_count += 1
        examples.append({"input_ids": src_ids, "labels": tgt_ids, "source": source, "target": target})
    print(
        f"Correction pairs usable: {len(examples)}; exact source/reference: {exact_count}; "
        f"skipped due to length/token alignment: {skipped_len}"
    )
    return examples


def collate_examples(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        src = item["input_ids"]
        tgt = item["labels"]
        n_pad = max_len - len(src)
        input_ids.append(src + [pad_token_id] * n_pad)
        attention_mask.append([1] * len(src) + [0] * n_pad)
        labels.append(tgt + [-100] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def calc_metrics(rows, tokenizer, model, title: str, output_csv: Optional[Path] = None):
    total_raw_edits = total_corrected_edits = total_base_edits = total_chars = 0
    exact_raw = exact_corrected = 0
    results = []
    model.eval()
    for i, row in enumerate(rows, 1):
        source = row.get("source_text", "")
        reference = row.get("reference", "")
        raw_norm, ref_norm = normalise_text(source), normalise_text(reference)
        if not ref_norm:
            continue
        corrected = macbert_predict(tokenizer, model, source)
        corr_norm = normalise_text(corrected)
        raw_ed = edit_distance(raw_norm, ref_norm)
        corr_ed = edit_distance(corr_norm, ref_norm)
        total_raw_edits += raw_ed
        total_corrected_edits += corr_ed
        total_chars += len(ref_norm)
        exact_raw += int(raw_norm == ref_norm)
        exact_corrected += int(corr_norm == ref_norm)
        results.append({
            "split": row.get("split", "valid"),
            "source_model": row.get("source_model", "unknown"),
            "audio_path": row.get("audio_path", ""),
            "reference": reference,
            "raw_prediction": source,
            "macbert_prediction": corrected,
            "raw_edits": raw_ed,
            "corrected_edits": corr_ed,
            "reference_chars": len(ref_norm),
            "raw_exact_match": int(raw_norm == ref_norm),
            "corrected_exact_match": int(corr_norm == ref_norm),
        })
        if i % 50 == 0:
            print(f"{title}: evaluated {i}/{len(rows)} candidates", flush=True)
    raw_cer = 100.0 * total_raw_edits / max(1, total_chars)
    corrected_cer = 100.0 * total_corrected_edits / max(1, total_chars)
    print(f"\n{title}")
    print(f"Raw candidate CER: {raw_cer:.2f}%")
    print(f"MacBERT-corrected CER: {corrected_cer:.2f}%")
    print(f"Raw exact matches: {exact_raw}/{len(results)}")
    print(f"Corrected exact matches: {exact_corrected}/{len(results)}")
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
            if results:
                writer.writeheader()
                writer.writerows(results)
    return raw_cer, corrected_cer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--av-manifest", type=Path, default=DEFAULT_AV_MANIFEST)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Optional cap for a quick smoke test; 0 uses all rows.")
    parser.add_argument("--regenerate-predictions", action="store_true",
                        help="Re-run Whisper predictions even if cached CSVs exist.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Device:", DEVICE)
    print("Training MacBERT using only training-split references.")
    print("Main manifest:", args.manifest)
    print("AV manifest:", args.av_manifest)
    print("Corrector checkpoint:", OUT_DIR)

    records = load_manifests(args.manifest, args.av_manifest)
    train_records = [r for r in records if r["split"] == "train"]
    valid_records = [r for r in records if r["split"] == "valid"]
    if not train_records or not valid_records:
        raise RuntimeError(
            f"Need both train and valid manifest rows; found train={len(train_records)}, valid={len(valid_records)}."
        )
    if args.max_train_samples > 0:
        train_records = train_records[:args.max_train_samples]

    # Avoid loading duplicate heavyweight Whisper models if cached transcripts exist.
    cache_train_ok = TRAIN_PRED_CSV.is_file() and not args.regenerate_predictions
    cache_valid_ok = VALID_PRED_CSV.is_file() and not args.regenerate_predictions
    predictors = None
    if not (cache_train_ok and cache_valid_ok):
        predictors = SpeechPredictors()
    train_candidates = create_or_load_prediction_csv(
        train_records, TRAIN_PRED_CSV, predictors, args.regenerate_predictions
    )
    valid_candidates = create_or_load_prediction_csv(
        valid_records, VALID_PRED_CSV, predictors, args.regenerate_predictions
    )
    del predictors
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # Load baseline corrector and measure it on validation before fine-tuning.
    tokenizer, base_corrector = load_corrector(BASE_CORRECTOR)
    base_corrector.to(DEVICE).eval()
    valid_rows = [r for r in valid_candidates if r.get("source_text") and r.get("reference")]
    if not valid_rows:
        raise RuntimeError("No validation prediction/reference pairs were generated or loaded.")
    base_raw_cer, base_corrected_cer = calc_metrics(
        valid_rows, tokenizer, base_corrector, "BASE PRETRAINED MACBERT", None
    )

    examples = build_examples(train_candidates, tokenizer, args.max_length)
    if len(examples) < 50:
        raise RuntimeError(
            f"Only {len(examples)} aligned training examples are usable. "
            "Check prediction/reference data and tokenizer alignment before training."
        )

    dataset = CorrectionDataset(examples)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
    )
    model = base_corrector
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(0.05 * total_steps))
    step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        batches = 0
        model.train()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                print("[warning] Non-finite loss; skipping this batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Simple linear warmup followed by constant learning rate.
            step += 1
            if step <= warmup_steps:
                factor = step / warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * factor
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        mean_loss = epoch_loss / max(1, batches)
        print(f"Epoch {epoch}/{args.epochs}: mean loss {mean_loss:.4f}", flush=True)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(str(CANDIDATE_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(CANDIDATE_DIR))
    print("Saved experimental candidate:", CANDIDATE_DIR)

    after_raw_cer, after_corrected_cer = calc_metrics(
        valid_rows, tokenizer, model, "FINE-TUNED MACBERT VALIDATION",
        VALID_RESULTS_CSV,
    )

    # Promote only when the trained correction model improves validation CER
    # over BOTH raw ASR and the original pretrained MacBERT corrector.
    promoted = (
        after_corrected_cer < after_raw_cer
        and after_corrected_cer < base_corrected_cer
    )
    if promoted:
        if OUT_DIR.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = OUT_DIR.with_name(f"{OUT_DIR.name}_backup_{timestamp}")
            OUT_DIR.rename(backup_dir)
            print("Preserved previous approved corrector at:", backup_dir)
        shutil.copytree(CANDIDATE_DIR, OUT_DIR)
        print("VALIDATION GATE PASSED. Active corrector updated:", OUT_DIR)
    else:
        print("VALIDATION GATE NOT PASSED. Active app model was not changed.")
        print("The experimental candidate remains at:", CANDIDATE_DIR)

    # Create a short text summary that can be reviewed after the run.
    summary_path = EVAL_DIR / "macbert_asr_corrector_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Fine-tuned MacBERT ASR corrector\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Train manifest rows used for inference: {len(train_records)}\n")
        f.write(f"Train prediction candidates: {len(train_candidates)}\n")
        f.write(f"Validation prediction candidates: {len(valid_rows)}\n")
        f.write(f"Usable training pairs: {len(examples)}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Final epoch mean loss: {mean_loss:.6f}\n")
        f.write(f"Validation raw CER: {after_raw_cer:.2f}%\n")
        f.write(f"Base MacBERT corrected CER: {base_corrected_cer:.2f}%\n")
        f.write(f"Fine-tuned MacBERT corrected CER: {after_corrected_cer:.2f}%\n")
        f.write(f"Promoted to app: {promoted}\n")
        f.write("Note: this trainer filters out source/reference character-length mismatches; it targets substitutions, not insertions/deletions.\n")
        f.write("Note: only train-split references are used for optimizer updates. Validation references are used for evaluation only.\n")
    print("Saved validation details:", VALID_RESULTS_CSV)
    print("Saved summary:", summary_path)
    print("The active app model changes only if validation CER improves over raw ASR and base MacBERT.")


if __name__ == "__main__":
    main()
