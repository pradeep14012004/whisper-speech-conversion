import os
import csv
import cv2
import numpy as np
import pandas as pd

OUT_DIR = "outputs/test_video_features"
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

    arr = np.stack(frames)
    if len(arr) < num_frames:
        arr = np.concatenate(
            [arr, np.repeat(arr[-1:], num_frames - len(arr), axis=0)],
            axis=0
        )
    return arr[:num_frames]

df = pd.read_csv("outputs/evaluation/official_test_manifest.csv")
success, failed = 0, 0
errors = []

for i, row in df.iterrows():
    audio = row["whisper_audio"]
    stem = os.path.splitext(os.path.basename(audio))[0]
    video = os.path.join(
        "test/video/camera/whisper",
        os.path.basename(os.path.dirname(audio)),
        stem + ".avi"
    )
    output = os.path.join(OUT_DIR, os.path.basename(video) + ".npy")

    if not os.path.isfile(video):
        failed += 1
        errors.append((audio, video, "video not found"))
        continue

    if os.path.exists(output):
        success += 1
        continue

    try:
        np.save(output, extract_video_features(video))
        success += 1
    except Exception as e:
        failed += 1
        errors.append((audio, video, str(e)))

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(df)} | success={success} | failed={failed}", flush=True)

if errors:
    with open("outputs/test_video_feature_errors.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio", "video", "error"])
        writer.writerows(errors)

print("\nExtraction finished")
print("Manifest rows:", len(df))
print("Saved or already present:", success)
print("Failed:", failed)
print("Feature directory:", OUT_DIR)
