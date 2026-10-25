import os, re, sys
import pandas as pd
import jiwer

try:
    from opencc import OpenCC
    t2s = OpenCC("t2s").convert
except Exception:
    print("WARNING: opencc not available, Traditional characters will not be converted")
    t2s = lambda s: s

def norm(s):
    s = "" if pd.isna(s) else str(s)
    return re.sub(r"[\W_]", "", t2s(s))

FILES = {
    "large-v3": "outputs/evaluation/official_test_large_v3_predictions.csv",
    "small":    "outputs/evaluation/official_test_small_predictions.csv",
}
EXPECTED = 3289

for name, path in FILES.items():
    print("\n=====", name, "=====")
    if not os.path.isfile(path):
        print("file not found:", path); continue
    d = pd.read_csv(path)
    print("rows:", len(d), "of", EXPECTED, "| unique clips:", d["whisper_audio"].nunique())
    d["ref"] = d["reference"].map(norm)
    d["hyp"] = d["prediction"].map(norm)
    d = d[d["ref"].str.len() > 0].copy()
    cer = jiwer.cer(list(d["ref"]), list(d["hyp"]))
    print("CER (normalized): %.2f%%" % (100 * cer))
    print("empty predictions:", int((d["hyp"] == "").sum()))
    d["speaker"] = d["whisper_audio"].str.split("/").str[4]
    ps = d.groupby("speaker").apply(lambda g: 100 * jiwer.cer(list(g["ref"]), list(g["hyp"])))
    print("per-speaker CER: min %.1f  median %.1f  max %.1f  (%d speakers)" % (ps.min(), ps.median(), ps.max(), len(ps)))
    print("worst 5 speakers:", ps.sort_values(ascending=False).head(5).round(1).to_dict())
    d["clip_cer"] = [jiwer.cer(r, h) for r, h in zip(d["ref"], d["hyp"])]
    d.to_csv(path.replace(".csv", "_scored.csv"), index=False, encoding="utf-8-sig")
