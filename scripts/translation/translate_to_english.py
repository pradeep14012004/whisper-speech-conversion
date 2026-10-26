import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MANIFEST = "outputs/training_manifest.csv"
OUT = "outputs/english_translations.csv"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(MANIFEST)
df = df.dropna(subset=["transcript"]).reset_index(drop=True)

print("Loading translation model:", MODEL_NAME, flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = "zho_Hans"

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()

english = []

for start in range(0, len(df), 8):
    texts = df["transcript"].iloc[start:start + 8].astype(str).tolist()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=128,
            num_beams=2
        )

    english.extend(tokenizer.batch_decode(
        generated, skip_special_tokens=True
    ))

    print(
        "Translated {}/{}".format(len(english), len(df)),
        flush=True
    )

df["english_translation"] = english
df.to_csv(OUT, index=False)
print("Saved:", OUT)
print(df[["transcript", "english_translation"]].head(10).to_string(index=False))
