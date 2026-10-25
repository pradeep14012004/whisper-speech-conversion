import re
import pandas as pd
import editdistance
from opencc import OpenCC

cc = OpenCC("t2s")

def norm(s):
    s = "" if pd.isna(s) else str(s)
    s = cc.convert(s)
    return re.sub(r"[\W_]", "", s)

df = pd.read_csv("outputs/evaluation/whisper_comparison_full.csv")
ref = df["reference"].map(norm)
total = sum(len(r) for r in ref)

for col in ["base_prediction", "lora_prediction"]:
    hyp = df[col].map(norm)
    df[col + "_err"] = [editdistance.eval(r, h) for r, h in zip(ref, hyp)]
    print(f"{col}: CER {df[col + '_err'].sum() / total * 100:.2f}%")

d = df["lora_prediction_err"] - df["base_prediction_err"]
print("LoRA better:", int((d < 0).sum()), " worse:", int((d > 0).sum()), " same:", int((d == 0).sum()))
print("Identical outputs:", round((df["base_prediction"] == df["lora_prediction"]).mean() * 100, 2), "%")
df.to_csv("outputs/evaluation/whisper_comparison_rescored.csv", index=False)
print("Saved rescored CSV")
