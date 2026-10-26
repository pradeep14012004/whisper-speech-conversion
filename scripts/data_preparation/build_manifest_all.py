import os
from collections import Counter

import pandas as pd
import soundfile as sf

BASE_ROOTS = [".", "data"]
OUT = "outputs/manifest_all.csv"
INFO = "data/AISHELL6-Whisper_info.csv"


def split_root(split):
    for r in BASE_ROOTS:
        if os.path.isdir(os.path.join(r, split)):
            return r
    return "."


def video_rel(audio_rel):
    # <split>/audio/hfm/<mode>/<spk>/<stem>.wav -> <split>/video/camera/<mode>/<spk>/<stem>.avi
    parts = audio_rel.split("/")
    if len(parts) > 3 and parts[1] == "audio" and parts[2] == "hfm":
        parts = [parts[0], "video", "camera"] + parts[3:]
    parts[-1] = os.path.splitext(parts[-1])[0] + ".avi"
    return "/".join(parts)


transcripts = {}
with open("data/text_sentence", encoding="utf-8") as f:
    for line in f:
        p = line.strip().split(maxsplit=1)
        if len(p) == 2:
            transcripts[p[0]] = p[1]
print("Transcript entries:", len(transcripts))

roots = {}
rows = []
bad_lines = 0
with open("data/w2n.txt", encoding="utf-8") as f:
    for line in f:
        p = line.strip().split()
        if len(p) != 2:
            bad_lines += 1
            continue
        w_rel, n_rel = p
        parts = w_rel.split("/")
        split = parts[0]
        if split not in roots:
            roots[split] = split_root(split)
        root = roots[split]
        speaker = parts[4] if len(parts) > 4 else ""
        stem = os.path.splitext(os.path.basename(w_rel))[0]
        wa = os.path.join(root, w_rel)
        na = os.path.join(root, n_rel)
        wv = os.path.join(root, video_rel(w_rel))
        nv = os.path.join(root, video_rel(n_rel))
        rows.append({
            "split": split,
            "speaker_id": speaker,
            "utt_stem": stem,
            "whisper_audio": wa,
            "normal_audio": na,
            "whisper_video": wv,
            "normal_video": nv,
            "transcript": transcripts.get(stem, ""),
            "transcript_exists": stem in transcripts,
            "whisper_exists": os.path.isfile(wa),
            "normal_exists": os.path.isfile(na),
            "whisper_video_exists": os.path.isfile(wv),
            "normal_video_exists": os.path.isfile(nv),
        })
print("Pairs read:", len(rows), "| malformed lines skipped:", bad_lines)

# Audio duration and sample rate (header read only)
for r in rows:
    r["whisper_dur_s"] = None
    r["whisper_sr"] = None
    if r["whisper_exists"]:
        try:
            i = sf.info(r["whisper_audio"])
            r["whisper_dur_s"] = round(i.duration, 3)
            r["whisper_sr"] = i.samplerate
        except Exception:
            pass
    r["usable_audio"] = bool(r["whisper_exists"] and r["normal_exists"] and r["transcript_exists"])
    r["usable_av"] = bool(r["usable_audio"] and r["whisper_video_exists"])

df = pd.DataFrame(rows)

# Whether the info CSV says this speaker has video
info = pd.read_csv(INFO)
spk_video = {(s, int(i)): v for s, i, v in zip(info["Split"], info["SpeakerID"], info["Video"])}
def flag(r):
    try:
        return spk_video.get((r["split"], int(r["speaker_id"].lstrip("S"))))
    except Exception:
        return None
df["info_video_flag"] = df.apply(flag, axis=1)

os.makedirs("outputs", exist_ok=True)
df.to_csv(OUT, index=False, encoding="utf-8-sig")

print("\n=== per split ===")
print(df.groupby("split").agg(
    rows=("utt_stem", "count"),
    speakers=("speaker_id", "nunique"),
    whisper_ok=("whisper_exists", "sum"),
    normal_ok=("normal_exists", "sum"),
    text_ok=("transcript_exists", "sum"),
    whisper_video_ok=("whisper_video_exists", "sum"),
    usable_audio=("usable_audio", "sum"),
    usable_av=("usable_av", "sum"),
))

exp = info[info["SpeechType"] == "whisper"].groupby("Split")["AudioNum"].sum()
print("\nExpected whisper clips from info CSV:", exp.to_dict())

print("\n=== checks ===")
sp = df.groupby("split")["speaker_id"].apply(set)
splits = list(sp.index)
for i in range(len(splits)):
    for j in range(i + 1, len(splits)):
        print("shared speakers", splits[i], "vs", splits[j], ":", len(sp[splits[i]] & sp[splits[j]]))
print("Duplicate whisper paths:", int(df["whisper_audio"].duplicated().sum()))
print("Info CSV flag matched rows:", int(df["info_video_flag"].notna().sum()), "of", len(df))
y = df[df["info_video_flag"] == "Y"]
print("Speakers marked video=Y but whisper video file missing:", int((~y["whisper_video_exists"]).sum()), "of", len(y))
n = df[df["info_video_flag"] == "N"]
print("Speakers marked video=N but whisper video file exists:", int(n["whisper_video_exists"].sum()))
d = df["whisper_dur_s"].dropna()
print("Whisper duration s: min %.2f  median %.2f  max %.2f  over 30s: %d" % (d.min(), d.median(), d.max(), int((d > 30).sum())))
print("Sample rates:", Counter(df["whisper_sr"].dropna().astype(int)).most_common(5))
print("\nSaved:", OUT)
