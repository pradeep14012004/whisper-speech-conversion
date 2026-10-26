import csv
import random
from pathlib import Path

ROOT = Path(".")
PAIR_FILE = ROOT / "data/w2n.txt"
TEXT_FILE = ROOT / "data/text_sentence"
OUT = ROOT / "outputs/conversion_pairs.csv"
SEED = 42

transcripts = {}
with TEXT_FILE.open(encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            transcripts[parts[0]] = parts[1]

rows = []
counts = {
    "malformed": 0, "missing_whisper": 0, "missing_normal": 0,
    "missing_transcript": 0, "complete": 0
}

with PAIR_FILE.open(encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            counts["malformed"] += 1
            continue

        whisper_rel, normal_rel = parts
        whisper = ROOT / whisper_rel
        normal = ROOT / normal_rel

        if not whisper.is_file():
            counts["missing_whisper"] += 1
            continue
        if not normal.is_file():
            counts["missing_normal"] += 1
            continue

        # Transcript IDs are the filenames without extension.
        whisper_id = whisper.stem
        normal_id = normal.stem
        transcript = transcripts.get(whisper_id, "")
        normal_transcript = transcripts.get(normal_id, "")

        if not transcript or not normal_transcript:
            counts["missing_transcript"] += 1
            continue
        if transcript != normal_transcript:
            continue

        speaker = whisper.parent.name
        rows.append({
            "speaker_id": speaker,
            "whisper_audio": str(whisper),
            "normal_audio": str(normal),
            "transcript": transcript,
            "whisper_id": whisper_id,
            "normal_id": normal_id,
            "split": ""
        })

# Split by speaker, not by utterance, to prevent speaker leakage.
speakers = sorted({r["speaker_id"] for r in rows})
random.Random(SEED).shuffle(speakers)
n_val = max(1, round(len(speakers) * 0.15)) if len(speakers) > 1 else 0
val_speakers = set(speakers[:n_val])

for r in rows:
    r["split"] = "valid" if r["speaker_id"] in val_speakers else "train"

OUT.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "split", "speaker_id", "whisper_audio", "normal_audio",
    "transcript", "whisper_id", "normal_id"
]
with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("Counts:", counts)
print("Verified same-transcript pairs:", len(rows))
print("Speakers:", len(speakers), "| train:", sum(s not in val_speakers for s in speakers),
      "| validation:", len(val_speakers))
print("Rows:", {s: sum(r["split"] == s for r in rows) for s in ("train", "valid")})
print("Saved:", OUT)
if not rows:
    print("WARNING: no verified pairs. Do not train until the transcript mapping is corrected.")
