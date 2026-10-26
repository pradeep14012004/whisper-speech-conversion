import csv, os, random
import soundfile as sf

random.seed(42)

with open("outputs/training_manifest.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

valid = []
for r in rows:
    try:
        if not all(os.path.exists(r[k]) for k in
                   ["whisper_audio", "normal_audio", "feature"]):
            continue
        wi, ni = sf.info(r["whisper_audio"]), sf.info(r["normal_audio"])
        if wi.duration >= 1 and ni.duration >= 1:
            valid.append(r)
    except Exception:
        pass

random.shuffle(valid)
subset = valid[:min(500, len(valid))]

os.makedirs("outputs", exist_ok=True)
with open("outputs/train_subset.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(subset[0].keys()))
    w.writeheader()
    w.writerows(subset)

print("Verified samples:", len(valid))
print("Subset saved:", len(subset))
print("File: outputs/train_subset.csv")
