import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

BASE = "openai/whisper-small"
MANIFEST = "outputs/training_manifest.csv"
OUT = "outputs/whisper_lora_small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
print("Device:", DEVICE)

processor = WhisperProcessor.from_pretrained(BASE)
model = WhisperForConditionalGeneration.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="chinese", task="transcribe"
)
model.config.use_cache = False

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.to(DEVICE)
model.print_trainable_parameters()

df = pd.read_csv(MANIFEST)
df = df.sample(n=min(32, len(df)), random_state=42).reset_index(drop=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5
)

model.train()
losses = []

for i, row in df.iterrows():
    try:
        audio, sr = sf.read(row["whisper_audio"])
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.nan_to_num(audio.astype(np.float32))

        if sr != 16000:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=16000
            )

        inputs = processor(
            audio.astype(np.float32),
            sampling_rate=16000,
            return_tensors="pt"
        )

        labels = processor.tokenizer(
            str(row["transcript"]),
            return_tensors="pt"
        ).input_ids

        input_features = inputs.input_features.to(DEVICE)
        labels = labels.to(DEVICE)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        optimizer.zero_grad(set_to_none=True)
        result = model(input_features=input_features, labels=labels)
        loss = result.loss

        if not torch.isfinite(loss):
            print("Non-finite loss at sample", i, "- stopping.")
            raise RuntimeError("Non-finite loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        losses.append(loss.item())
        print(
            "Sample {}/{} | loss {:.4f}".format(
                i + 1, len(df), loss.item()
            ),
            flush=True
        )

    except Exception as e:
        print("Stopped at sample {}: {}".format(i, repr(e)), flush=True)
        raise

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
processor.save_pretrained(OUT)

print("Mean training loss:", float(np.mean(losses)))
print("LoRA adapter saved to:", OUT)
