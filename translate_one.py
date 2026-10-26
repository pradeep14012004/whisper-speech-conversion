import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if len(sys.argv) < 2:
    raise SystemExit('Usage: python translate_one.py "Chinese text"')

text = " ".join(sys.argv[1:]).strip()
if not text:
    raise SystemExit("Error: empty input text")

model_name = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading translation model:", model_name, flush=True)
print("Device:", device, flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    src_lang="zho_Hans"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_new_tokens=128,
        num_beams=4
    )

translation = tokenizer.batch_decode(
    output, skip_special_tokens=True
)[0].strip()

print("\nChinese:", text)
print("English:", translation)

with open("outputs/demo_english.txt", "w", encoding="utf-8") as f:
    f.write(translation + "\n")

print("Saved: outputs/demo_english.txt")
