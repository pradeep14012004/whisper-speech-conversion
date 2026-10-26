import os
import glob
import csv

ROOT = "data/valid"
TRANSCRIPT_FILE = "data/text_sentence"
OUT = "outputs/valid_manifest.csv"

transcripts = {}
with open(TRANSCRIPT_FILE, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            transcripts[parts[0]] = parts[1]

rows = []
audio_files = glob.glob(ROOT + "/audio/hfm/whisper/*/*.wav")

for audio in audio_files:
    stem = os.path.splitext(os.path.basename(audio))[0]
    transcript = transcripts.get(stem)

    # Find video with the same utterance ID, if available.
    matches = []
    for ext in ("mp4", "avi", "mpg", "mpeg", "mov"):
        matches.extend(glob.glob(
            ROOT + "/video/**/" + stem + "." + ext,
            recursive=True
        ))

    rows.append({
        "audio": audio,
        "video": matches[0] if matches else "",
        "transcript": transcript or "",
        "has_video": bool(matches),
        "has_transcript": bool(transcript)
    })

os.makedirs("outputs", exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["audio", "video", "transcript",
                    "has_video", "has_transcript"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Manifest:", OUT)
print("Whisper audio samples:", len(rows))
print("Samples with matching video:", sum(r["has_video"] for r in rows))
print("Samples with transcripts:", sum(r["has_transcript"] for r in rows))
print("Samples with both:", sum(
    r["has_video"] and r["has_transcript"] for r in rows
))
