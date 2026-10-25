import re
import pandas as pd
import editdistance
from opencc import OpenCC

cc = OpenCC("t2s")
def norm(s):
    s = "" if pd.isna(s) else str(s)
    return re.sub(r"[\W_]", "", cc.convert(s))

small = pd.read_csv("outputs/evaluation/whisper_comparison_full.csv", keep_default_na=False)
large = pd.read_csv("outputs/evaluation/large_v3_predictions.csv", keep_default_na=False)
n = min(len(small), len(large))
small = small.iloc[:n].reset_index(drop=True)
large = large.iloc[:n].reset_index(drop=True)
assert (small["reference"].astype(str) == large["reference"].astype(str)).all(), "clip order differs"

ref = small["reference"].map(norm)
total = sum(len(r) for r in ref)
def errs(col):
    return [editdistance.eval(r, norm(h)) for r, h in zip(ref, col)]

e = {
    "whisper-small": errs(small["base_prediction"]),
    "small + LoRA": errs(small["lora_prediction"]),
    "large-v3": errs(large["prediction"]),
}
print("Clips:", n)
for k, v in e.items():
    print(f"{k:15s} CER {sum(v) / total * 100:.2f}%")
print("Large-v3 empty outputs:", int((large["prediction"].map(norm) == "").sum()))
d = pd.Series(e["large-v3"]) - pd.Series(e["whisper-small"])
print("Large-v3 vs small -> better:", int((d < 0).sum()), "worse:", int((d > 0).sum()), "same:", int((d == 0).sum()))
