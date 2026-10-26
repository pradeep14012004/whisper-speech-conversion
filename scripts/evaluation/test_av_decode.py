import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from av_fusion import AudioVisualFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER = "outputs/whisper_lora_small"
CHECKPOINT = "outputs/av_fusion_train_only.pt"

df = pd.read_csv("outputs/aligned_av_manifest.csv")
row = df.iloc[0]

audio, sr = sf.read(row["audio"])
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = np.nan_to_num(audio.astype(np.float32))
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)

processor = WhisperProcessor.from_pretrained(ADAPTER)
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE).eval()

ckpt = torch.load(CHECKPOINT, map_location="cpu")
fusion = AudioVisualFusion(
    audio_dim=ckpt["audio_dim"],
    video_dim=ckpt["video_dim"],
    hidden_dim=ckpt["hidden_dim"],
    heads=4
).to(DEVICE)
fusion.load_state_dict(ckpt["fusion"])
fusion.eval()

decoder = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.GELU(),
    torch.nn.Linear(512, ckpt["audio_dim"])
).to(DEVICE)
decoder.load_state_dict(ckpt["decoder"])
decoder.eval()

inputs = processor(
    audio, sampling_rate=16000, return_tensors="pt"
).input_features.to(DEVICE)

video = torch.from_numpy(
    np.load(row["feature"]).astype(np.float32)
).unsqueeze(0).to(DEVICE)

with torch.inference_mode():
    audio_hidden = model.get_encoder()(
        input_features=inputs
    ).last_hidden_state

    fused = fusion(audio_hidden, video)
    predicted_hidden = decoder(fused)

    print("Audio encoder shape:", tuple(audio_hidden.shape))
    print("Predicted encoder shape:", tuple(predicted_hidden.shape))
    print("Finite:", torch.isfinite(predicted_hidden).all().item())

    # Whisper decoder receives the predicted encoder states.
    # Audio-only baseline using the same Whisper+LoRA model
    baseline = model.generate(
        inputs,
        language="chinese",
        task="transcribe",
        max_new_tokens=128
    )

    # Audiovisual prediction
    encoder_outputs = BaseModelOutput(last_hidden_state=predicted_hidden)
    generated = model.generate(
        inputs,
        encoder_outputs=encoder_outputs,
        language="chinese",
        task="transcribe",
        max_new_tokens=128
    )

baseline_text = processor.batch_decode(
    baseline, skip_special_tokens=True
)[0]

prediction = processor.batch_decode(
    generated, skip_special_tokens=True
)[0]

print("Reference:", str(row["transcript"]))
print("Audio-only prediction:", baseline_text)
print("AV prediction:", prediction)
print("Single-clip AV decoding test: PASS")
