import csv, os, glob

features_dir = "outputs/video_features"
feature_names = {
    os.path.basename(p)[:-8]
    for p in glob.glob(features_dir + "/*.npy")
}

with open("outputs/valid_manifest.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

matched = []
for r in rows:
    video = (r.get("video") or "").strip()
    audio = (r.get("audio") or "").strip()
    transcript = (r.get("transcript") or "").strip()

    if not video or not audio or not transcript:
        continue

    stem = os.path.basename(video)
    if stem.endswith(".avi"):
        stem = stem[:-4]

    if stem in feature_names and os.path.exists(audio):
        matched.append({
            "audio": audio,
            "video": video,
            "feature": os.path.join(features_dir, stem + ".avi.npy"),
            "transcript": transcript
        })

os.makedirs("outputs", exist_ok=True)
with open("outputs/aligned_av_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["audio", "video", "feature", "transcript"])
    writer.writeheader()
    writer.writerows(matched)

print("Rows in original manifest:", len(rows))
print("Aligned audio-video-text samples:", len(matched))
print("Saved: outputs/aligned_av_manifest.csv")
