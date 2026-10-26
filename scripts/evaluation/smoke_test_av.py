import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from peft import PeftModel

BASE = "openai/whisper-small"
ADAPTER = "outputs/whisper_lora_small"
CHECKPOINT = "outputs/av_transcription_pilot.pt"
MANIFEST = "outputs/aligned_av_manifest.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class AVResidualAdapter(nn.Module):
    def __init__(self, audio_dim=768, hidden_dim=256, video_dim=128):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(start_dim=1),
            nn.Linear(16 * 4 * 4, video_dim), nn.ReLU()
        )
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, audio_dim)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, audio_hidden, video):
        audio_q = self.audio_proj(audio_hidden)
        b, t, h, w = video.shape
        vf = self.video_encoder(video.reshape(b * t, 1, h, w))
        vf = self.video_proj(vf.reshape(b, t, -1))
        attended, _ = self.cross_attention(audio_q, vf, vf, need_weights=False)
        return audio_hidden + self.gate * self.out_proj(attended)

def main():
    assert os.path.isfile(CHECKPOINT), f"Missing {CHECKPOINT}"
    assert os.path.isdir(ADAPTER), f"Missing {ADAPTER}"
    df = pd.read_csv(MANIFEST).dropna(subset=["audio", "feature", "transcript"])
    df = df[df.audio.map(os.path.isfile) & df.feature.map(os.path.isfile)]
    assert len(df), "No complete validation examples"
    row = df.iloc[0]
    print("Device:", DEVICE)
    print("Audio:", row.audio)
    print("Video feature:", row.feature)

    audio, sr = sf.read(row.audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio.astype(np.float32))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)

    video_np = np.load(row.feature).astype(np.float32)
    assert video_np.shape == (16, 48, 48), f"Unexpected feature shape {video_np.shape}"
    assert np.isfinite(video_np).all(), "Video feature has NaN/Inf"

    processor = WhisperProcessor.from_pretrained(ADAPTER)
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER).to(DEVICE).eval()
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    av = AVResidualAdapter(audio_dim=model.config.d_model).to(DEVICE)
    av.load_state_dict(ckpt["av_adapter"])
    av.eval()

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    video = torch.from_numpy(video_np).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        audio_hidden = model.get_encoder()(input_features=inputs).last_hidden_state
        fused_hidden = av(audio_hidden, video)
        base_tokens = model.generate(inputs, language="chinese", task="transcribe", max_new_tokens=128)
        av_tokens = model.generate(
            inputs,
            encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
            language="chinese", task="transcribe", max_new_tokens=128
        )

    baseline = processor.batch_decode(base_tokens, skip_special_tokens=True)[0]
    prediction = processor.batch_decode(av_tokens, skip_special_tokens=True)[0]
    print("\nREFERENCE:", row.transcript)
    print("AUDIO-ONLY:", baseline)
    print("AV PREDICTION:", prediction)
    assert isinstance(prediction, str) and prediction.strip(), "AV output is empty"
    print("\nSMOKE TEST PASSED: model loaded and generated non-empty AV transcription.")

if __name__ == "__main__":
    main()
