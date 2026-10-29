from pathlib import Path, PurePosixPath
import csv
import shutil
import tarfile

import cv2
import numpy as np

ROOT = Path.cwd()
DATA = Path.home() / "Downloads" / "aishell6_whisper_training"
SOURCE_MANIFEST = ROOT / "data/aishell6_lora_2k/manifest.csv"
OUT = ROOT / "data/aishell6_av_2k"
VIDEO_DIR = OUT / "videos"
FEATURE_DIR = OUT / "video_features"
OUTPUT_MANIFEST = OUT / "manifest.csv"

NUM_FRAMES = 16
FRAME_SIZE = 48

if not SOURCE_MANIFEST.is_file():
    raise FileNotFoundError(SOURCE_MANIFEST)

for name in ("train.tar.gz", "valid.tar.gz"):
    if not (DATA / name).is_file():
        raise FileNotFoundError(DATA / name)

if OUTPUT_MANIFEST.exists():
    raise FileExistsError(
        f"{OUTPUT_MANIFEST} already exists. "
        "Move it aside before regenerating the dataset."
    )

with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    source_rows = list(csv.DictReader(f))

selected = [
    row for row in source_rows
    if row.get("split") in ("train", "valid")
]

if not selected:
    raise RuntimeError("No training or validation rows found.")

# Index the selected recordings by unique utterance stem.
wanted = {}
for row in selected:
    source_audio = PurePosixPath(row["source_audio"])
    split = row["split"]

    if source_audio.parts[0] != split:
        raise ValueError(f"Unexpected split path: {source_audio}")

    stem = source_audio.stem
    key = (split, stem)

    if key in wanted:
        raise ValueError(f"Duplicate selected recording: {key}")

    wanted[key] = row

print("Selected audio clips:", len(selected))
print("Training:", sum(r["split"] == "train" for r in selected))
print("Validation:", sum(r["split"] == "valid" for r in selected))

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

found = set()
video_for_row = {}

for split in ("train", "valid"):
    archive_path = DATA / f"{split}.tar.gz"
    targets = {
        key: row for key, row in wanted.items()
        if key[0] == split
    }

    print(f"\nSearching {archive_path.name} for matching videos...")

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue

            path = PurePosixPath(member.name)
            parts = {part.lower() for part in path.parts}

            if path.suffix.lower() != ".avi":
                continue
            if "video" not in parts or "whisper" not in parts:
                continue
            if not path.parts or path.parts[0] != split:
                continue

            key = (split, path.stem)
            row = targets.get(key)

            if row is None or key in found:
                continue

            target = VIDEO_DIR / split / (path.stem + ".avi")
            target.parent.mkdir(parents=True, exist_ok=True)

            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read archive file: {member.name}")

            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)

            video_for_row[key] = (target, member.name)
            found.add(key)

            if len(found) % 100 == 0:
                print(f"Matched and extracted videos: {len(found)}", flush=True)

            if len(found) == len(wanted):
                break

    print(f"{split}: found {sum(k[0] == split for k in found)} videos")

if not found:
    raise RuntimeError(
        "No matching videos found. Check the video paths in the archives."
    )

def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            indices = np.linspace(
                0, total - 1, NUM_FRAMES
            ).astype(int)

            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
                frames.append(gray.astype(np.float32) / 255.0)
        else:
            # Fallback for videos whose frame count is unavailable.
            while len(frames) < NUM_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
                frames.append(gray.astype(np.float32) / 255.0)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No readable frames in {video_path}")

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())

    return np.stack(frames[:NUM_FRAMES]).astype(np.float32)

manifest_rows = []
unreadable = []

for index, row in enumerate(selected, start=1):
    key = (row["split"], PurePosixPath(row["source_audio"]).stem)

    if key not in video_for_row:
        continue

    audio_path = Path(row["audio_path"])
    if not audio_path.is_file():
        print("Skipping missing audio:", audio_path)
        continue

    video_path, original_video = video_for_row[key]
    feature_path = (
        FEATURE_DIR / row["split"] / f"{video_path.stem}.npy"
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        frames = extract_frames(video_path)
        np.save(feature_path, frames)
    except Exception as exc:
        unreadable.append((str(video_path), str(exc)))
        continue

    manifest_rows.append({
        "split": row["split"],
        "audio_path": str(audio_path.resolve()),
        "video_path": str(video_path.resolve()),
        "feature_path": str(feature_path.resolve()),
        "transcript": row["transcript"],
        "source_audio": row["source_audio"],
        "source_video": original_video,
        "speaker": row["speaker"],
    })

    if len(manifest_rows) % 100 == 0:
        print(f"Video features prepared: {len(manifest_rows)}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
columns = [
    "split", "audio_path", "video_path", "feature_path",
    "transcript", "source_audio", "source_video", "speaker",
]

with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(manifest_rows)

print("\nAV DATA PREPARATION COMPLETE")
print("Selected audio clips:", len(selected))
print("Matching videos found:", len(found))
print("Usable audio/video/transcript rows:", len(manifest_rows))
print("Training rows:", sum(r["split"] == "train" for r in manifest_rows))
print("Validation rows:", sum(r["split"] == "valid" for r in manifest_rows))
print("Unreadable videos:", len(unreadable))
print("Manifest:", OUTPUT_MANIFEST)

if unreadable:
    print("First unreadable video:", unreadable[0])
