import os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SRC = "outputs/manifest_all.csv"
DST = "outputs/manifest_all_16k.csv"
CACHE = "cache16k"
WORKERS = min(8, os.cpu_count() or 4)


def out_path(p, split):
    return os.path.join(CACHE, split, p.split(split + "/", 1)[1])


def convert(job):
    src, dst = job
    if os.path.isfile(dst):
        return 0
    try:
        x, sr = sf.read(src, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            g = np.gcd(sr, 16000)
            x = resample_poly(x, 16000 // g, sr // g).astype("float32")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        sf.write(tmp, x, 16000, subtype="PCM_16", format="WAV")
        os.replace(tmp, dst)
        return 0
    except Exception as e:
        print("FAILED", src, e, flush=True)
        return 1


if __name__ == "__main__":
    df = pd.read_csv(SRC)
    df = df[df["usable_audio"]].copy()
    df["whisper_16k"] = [out_path(p, s) for p, s in zip(df["whisper_audio"], df["split"])]
    df["normal_16k"] = [out_path(p, s) for p, s in zip(df["normal_audio"], df["split"])]

    jobs = list(zip(df["whisper_audio"], df["whisper_16k"])) + list(zip(df["normal_audio"], df["normal_16k"]))
    print("Files to check/convert:", len(jobs), "| workers:", WORKERS, flush=True)

    failed = 0
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(convert, jobs, chunksize=32), 1):
            failed += r
            if i % 2000 == 0:
                print(i, "/", len(jobs), "failed:", failed, flush=True)

    df.to_csv(DST, index=False, encoding="utf-8-sig")
    print("Done. failed =", failed, "| saved", DST)
