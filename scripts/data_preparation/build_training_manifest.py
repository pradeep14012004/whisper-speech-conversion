import csv, os, glob

feature_dir = "outputs/video_features"
feature_map = {}

for p in glob.glob(feature_dir + "/*.npy"):
    # Feature file: <video_filename>.avi.npy
    video_stem = os.path.basename(p)[:-8]
    feature_map[video_stem] = p

with open("outputs/valid_pairs.csv", encoding="utf-8-sig") as f:
    pairs = list(csv.DictReader(f))

with open("outputs/aligned_av_manifest.csv", encoding="utf-8-sig") as f:
    videos = list(csv.DictReader(f))

# Map video by whispered audio's utterance stem
video_map = {}
for v in videos:
    audio_stem = os.path.splitext(os.path.basename(v["audio"]))[0]
    video_stem = os.path.basename(v["video"])
    if video_stem.endswith(".avi"):
        video_stem = video_stem[:-4]
    if video_stem in feature_map:
        video_map[audio_stem] = (v["video"], feature_map[video_stem])

out = []
for r in pairs:
    whisper = r["whisper_audio"]
    normal = r["normal_audio"]
    transcript = r["transcript"]
    stem = os.path.splitext(os.path.basename(whisper))[0]

    if (r["whisper_exists"] == "True" and
        r["normal_exists"] == "True" and
        r["transcript_exists"] == "True" and
        stem in video_map):
        video, feature = video_map[stem]
        out.append({
            "whisper_audio": whisper,
            "normal_audio": normal,
            "video": video,
            "feature": feature,
            "transcript": transcript
        })

with open("outputs/training_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "whisper_audio", "normal_audio", "video", "feature", "transcript"
    ])
    writer.writeheader()
    writer.writerows(out)

print("Training rows with paired normal audio + video:", len(out))
print("Saved outputs/training_manifest.csv")
if out:
    print("Example:", out[0])
