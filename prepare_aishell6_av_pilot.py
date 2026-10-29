from pathlib import Path, PurePosixPath
import csv
import tarfile
import shutil

DATA = Path.home() / "Downloads" / "aishell6_whisper_training"
ROOT = Path("data/aishell6_lora_2k").resolve()
MANIFEST = ROOT / "manifest.csv"
VIDEO_ROOT = ROOT / "videos"
OUT_MANIFEST = ROOT / "av_manifest.csv"

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Link each audio clip to the video with the same recording stem.
for row in rows:
    audio = PurePosixPath(row["source_audio"])
    split = audio.parts[0]
    speaker = audio.parts[-2]
    stem = audio.stem

    row["source_video"] = (
        f"{split}/video/camera/whisper/{speaker}/{stem}.avi"
    )
    row["video_path"] = ""

print("Audio clips requested:", len(rows))

for split in ("train", "valid"):
    archive_path = DATA / f"{split}.tar.gz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    targets = {
        row["source_video"]: row
        for row in rows
        if row["split"] == split
    }
    found = set()

    print(f"\nSearching {archive_path.name} for {len(targets)} videos...")

    # Stream the archive and extract only the requested videos.
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            row = targets.get(member.name)
            if row is None or not member.isfile():
                continue

            relative = PurePosixPath(member.name)
            destination = (
                VIDEO_ROOT / split / relative.parts[-2] / relative.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Reuse files extracted by the previous run.
            if not destination.is_file() or destination.stat().st_size == 0:
                source = tar.extractfile(member)
                if source is None:
                    print("Could not read:", member.name)
                    continue
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

            if destination.is_file() and destination.stat().st_size > 0:
                row["video_path"] = str(destination)
                found.add(member.name)

            if len(found) % 100 == 0 and found:
                print(f"Matched {len(found)}/{len(targets)}", flush=True)

            if len(found) == len(targets):
                break

    missing = len(targets) - len(found)
    print(f"{split}: matched videos={len(found)}, missing videos={missing}")

# Keep only usable AV pairs in the AV manifest.
matched_rows = [
    row for row in rows
    if row["video_path"]
    and Path(row["audio_path"]).is_file()
    and Path(row["video_path"]).is_file()
    and row["transcript"].strip()
]

if not matched_rows:
    raise RuntimeError("No complete audio/video/transcript pairs found.")

fields = list(rows[0].keys())
with OUT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(matched_rows)

print("\nAV DATA PREPARATION COMPLETE")
print("Matched training pairs:", sum(r["split"] == "train" for r in matched_rows))
print("Matched validation pairs:", sum(r["split"] == "valid" for r in matched_rows))
print("AV manifest:", OUT_MANIFEST)
print("Original audio manifest and model checkpoints were not modified.")
