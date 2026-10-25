import os
import csv
import cv2
import numpy as np

OUT_DIR = "outputs/train_video_features"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_video_features(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened() or total <= 0:
        cap.release()
        raise ValueError("Cannot read video: " + path)

    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            frames.append(gray.astype(np.float32) / 255.0)

    cap.release()

    if not frames:
        raise ValueError("No readable frames: " + path)

    # Fixed length: pad videos that yield fewer than 16 readable frames
    arr = np.stack(frames)
    if len(arr) < num_frames:
        pad = np.repeat(arr[-1:], num_frames - len(arr), axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:num_frames]

with open("outputs/train_av_manifest.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r.get("whisper_video")]

success, failed = 0, 0
errors = []

for i, row in enumerate(rows, 1):
    video = row["whisper_video"]
    # Stable filename based on video basename
    name = os.path.basename(video) + ".npy"
    output = os.path.join(OUT_DIR, name)

    if os.path.exists(output):
        success += 1
        continue

    try:
        features = extract_video_features(video)
        np.save(output, features)
        success += 1
    except Exception as e:
        failed += 1
        errors.append((video, str(e)))

    if i % 100 == 0:
        print(f"Processed {i}/{len(rows)} | saved/skipped={success} | failed={failed}", flush=True)

print("\nExtraction finished")
print("Videos in manifest:", len(rows))
print("Saved or already present:", success)
print("Failed:", failed)

if errors:
    with open("outputs/train_video_feature_errors.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "error"])
        writer.writerows(errors)
    print("Errors saved to outputs/video_feature_errors.csv")

print("Feature directory:", OUT_DIR)
