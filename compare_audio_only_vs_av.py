from pathlib import Path, PurePosixPath
import argparse
import csv
import unicodedata

import cv2
import editdistance
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parent
LORA_DIR = ROOT / "outputs/whisper_lora_small"
AV_CHECKPOINT = ROOT / "outputs/av_transcription_full.pt"

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
        vf = self.video_encoder(
            video.reshape(batch * frames, 1, height, width)
        )
        vf = self.video_proj(vf.reshape(batch, frames, -1))
        attended, _ = self.cross_attention(
            audio_q, vf, vf, need_weights=False
        )
        return audio_hidden + self.gate * self.out_proj(attended)


def get_sample(sample_number):
    folders = [
        ROOT / "demo_samples_5pairs",
        ROOT / "demo_samples",
    ]
    for folder in folders:
        manifest = folder / "manifest.csv"
        if manifest.is_file():
            with manifest.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            row = next(
                (r for r in rows if str(r.get("sample", "")) == str(sample_number)),
                None,
            )
            if row:
                audio = folder / row.get("audio_file", row.get("audio", ""))
                video_name = row.get("video_file", row.get("video", ""))
                video = folder / video_name
                if audio.is_file() and video.is_file():
                    return audio, video, row

    # Also support the earlier demo_samples/sample.wav + sample.avi layout.
    folder = ROOT / "demo_samples"
    if sample_number == 1 and (folder / "sample.wav").is_file():
        videos = [
            p for p in folder.glob("sample.*")
            if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".webm"}
        ]
        if videos:
            return folder / "sample.wav", videos[0], {}

    raise FileNotFoundError(
        f"Cannot find sample {sample_number} with a matching audio/video pair. "
        "Check demo_samples_5pairs/manifest.csv and its filenames."
    )


def extract_video_frames(path, count=16):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = (
            np.linspace(0, total - 1, count).astype(int)
            if total > 0 else []
        )
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (48, 48))
                frames.append(gray.astype(np.float32) / 255.0)
    finally:
        cap.release()
    if not frames:
        raise ValueError("No video frames extracted.")
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return np.stack(frames[:count])


from opencc import OpenCC
_cc = OpenCC("t2s")

def normalize(text):
    text = _cc.convert(unicodedata.normalize("NFKC", text).lower())
    return "".join(ch for ch in text if ch.isalnum())


def find_reference(row, audio):
    reference = (row.get("reference_transcript") or "").strip()
    if reference:
        return reference

    names = {
        audio.stem,
        PurePosixPath(row.get("original_audio_path", "")).stem,
        row.get("recording_stem", ""),
    }
    transcript_file = (
        Path.home() / "Downloads" / "aishell6_whisper_test" / "text_sentence"
    )
    if transcript_file.is_file():
        with transcript_file.open(encoding="utf-8-sig") as f:
            for line in f:
                fields = line.strip().split(maxsplit=1)
                if len(fields) == 2 and (
                    fields[0] in names or PurePosixPath(fields[0]).stem in names
                ):
                    return fields[1]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1)
    args = parser.parse_args()

    if not LORA_DIR.is_dir():
        raise FileNotFoundError(f"Missing LoRA adapter: {LORA_DIR}")
    if not AV_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing AV checkpoint: {AV_CHECKPOINT}")

    audio_path, video_path, row = get_sample(args.sample)
    reference = find_reference(row, audio_path)

    print("Audio:", audio_path)
    print("Video:", video_path)
    print("Device: CPU on this Mac unless CUDA is available")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("Audio file is empty.")
    if len(audio) / sr > 120:
        raise ValueError("Please choose an audio clip of 120 seconds or less.")
    if sr != 16000:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=16000
        ).astype(np.float32)

    processor = WhisperProcessor.from_pretrained(str(LORA_DIR))
    base = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    )
    whisper = PeftModel.from_pretrained(base, str(LORA_DIR)).to(device).eval()

    checkpoint = torch.load(AV_CHECKPOINT, map_location="cpu")
    adapter = AVResidualAdapter(audio_dim=whisper.config.d_model)
    adapter.load_state_dict(checkpoint["av_adapter"])
    adapter = adapter.to(device).eval()

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    print("\nRunning audio-only inference...")
    with torch.inference_mode():
        audio_tokens = whisper.generate(
            input_features=features,
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
        )
    audio_text = processor.batch_decode(
        audio_tokens, skip_special_tokens=True
    )[0].strip()

    print("Running audio-visual inference...")
    frames = extract_video_frames(video_path)
    video = torch.tensor(
        frames, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.inference_mode():
        hidden = whisper.get_encoder()(
            input_features=features
        ).last_hidden_state
        fused = adapter(hidden, video)
        av_tokens = whisper.generate(
            input_features=features,
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            language="chinese",
            task="transcribe",
            max_new_tokens=128,
        )
    av_text = processor.batch_decode(
        av_tokens, skip_special_tokens=True
    )[0].strip()

    print("\n" + "=" * 60)
    print("AUDIO ONLY (Whisper + LoRA):")
    print(audio_text)
    print("\nAUDIO + VIDEO (Whisper + LoRA + AV adapter):")
    print(av_text)
    print("\nREFERENCE TRANSCRIPT:")
    print(reference or "NOT FOUND — CER cannot be calculated reliably.")

    if reference:
        ref = normalize(reference)
        for label, prediction in [
            ("Audio-only CER", audio_text),
            ("Audio-visual CER", av_text),
        ]:
            pred = normalize(prediction)
            cer = editdistance.eval(ref, pred) / max(1, len(ref))
            print(f"{label}: {cer * 100:.2f}%")

    out = ROOT / "outputs" / "class_demo"
    out.mkdir(parents=True, exist_ok=True)
    report = (
        f"Audio: {audio_path}\nVideo: {video_path}\n"
        f"Reference: {reference}\n"
        f"Audio only: {audio_text}\nAudio + video: {av_text}\n"
    )
    (out / f"comparison_sample_{args.sample:02d}.txt").write_text(
        report, encoding="utf-8"
    )
    print("\nSaved comparison report to:", out / f"comparison_sample_{args.sample:02d}.txt")


if __name__ == "__main__":
    main()
