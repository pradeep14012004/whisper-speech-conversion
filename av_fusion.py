import torch
import torch.nn as nn
import numpy as np

class AudioVisualFusion(nn.Module):
    def __init__(self, audio_dim=768, video_dim=128, hidden_dim=256, heads=4):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.video_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 4 * 4, video_dim),
            nn.ReLU()
        )

        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, audio_features, video_frames):
        # audio_features: [B, T_audio, 768]
        # video_frames: [B, 16, 48, 48]
        audio = self.audio_proj(audio_features)

        b, t, h, w = video_frames.shape
        video = self.video_encoder(
            video_frames.reshape(b * t, 1, h, w)
        )
        video = video.reshape(b, t, -1)
        video = self.video_proj(video)

        fused, _ = self.cross_attention(
            query=audio,
            key=video,
            value=video
        )

        return self.norm(audio + fused)


if __name__ == "__main__":
    model = AudioVisualFusion()
    audio = torch.randn(2, 150, 768)
    video = torch.randn(2, 16, 48, 48)
    output = model(audio, video)

    print("Audio input:", audio.shape)
    print("Video input:", video.shape)
    print("Fused output:", output.shape)
    print("Fusion module test passed.")
