import csv
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import cer

MODEL = "openai/whisper-large-v3"

with open("outputs/valid_pairs.csv", encoding="utf-8-sig") as f:
    row = next(csv.DictReader(f))

processor = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda").eval()

def transcribe(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to("cuda", dtype=torch.float16)

    with torch.inference_mode():
        ids = model.generate(
            features, language="chinese",
            task="transcribe", max_new_tokens=128
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

reference = row["transcript"]
print("Reference:", reference, flush=True)

for key in ("whisper_audio", "normal_audio"):
    prediction = transcribe(row[key])
    print("\nAudio type:", key, flush=True)
    print("Prediction:", prediction, flush=True)
    print("CER:", round(cer(reference, prediction), 4), flush=True)
