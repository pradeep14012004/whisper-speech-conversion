import os, csv

BASE = "data"
OUT = "outputs/valid_pairs.csv"

transcripts = {}
with open(os.path.join(BASE, "text_sentence"), encoding="utf-8") as f:
    for line in f:
        p = line.strip().split(maxsplit=1)
        if len(p) == 2:
            transcripts[p[0]] = p[1]

rows = []
missing_whisper = missing_normal = missing_text = 0

with open(os.path.join(BASE, "w2n.txt"), encoding="utf-8") as f:
    for line in f:
        p = line.strip().split()
        if len(p) != 2 or p[0].split("/")[0] != "valid":
            continue

        whisper, normal = p
        stem = os.path.splitext(os.path.basename(whisper))[0]
        wp, np = os.path.join(BASE, whisper), os.path.join(BASE, normal)

        if not os.path.isfile(wp):
            missing_whisper += 1
        if not os.path.isfile(np):
            missing_normal += 1
        if stem not in transcripts:
            missing_text += 1

        rows.append({
            "whisper_audio": wp,
            "normal_audio": np,
            "transcript": transcripts.get(stem, ""),
            "whisper_exists": os.path.isfile(wp),
            "normal_exists": os.path.isfile(np),
            "transcript_exists": stem in transcripts
        })

os.makedirs("outputs", exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("Validation pairs:", len(rows))
print("Missing whispered audio:", missing_whisper)
print("Missing normal audio:", missing_normal)
print("Missing transcripts:", missing_text)
print("Complete pairs:", sum(
    r["whisper_exists"] and r["normal_exists"] and r["transcript_exists"]
    for r in rows
))
print("Saved:", OUT)
