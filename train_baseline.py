import os, csv, random
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR, N_MELS, FRAMES = 16000, 80, 256
BATCH, EPOCHS = 8, 5

with open("outputs/train_subset.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
random.shuffle(rows)
cut = max(1, int(0.9 * len(rows)))
train_rows, val_rows = rows[:cut], rows[cut:]

def load_mel(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=SR, n_fft=400, hop_length=160,
        n_mels=N_MELS, fmin=0, fmax=SR // 2
    )
    mel = np.log(np.maximum(mel, 1e-5))
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)
    # Resize time axis to fixed length
    mel = librosa.util.fix_length(mel, size=FRAMES, axis=1)
    return mel.astype(np.float32)

class AudioPairs(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = load_mel(r["whisper_audio"])
        y = load_mel(r["normal_audio"])
        return torch.from_numpy(x), torch.from_numpy(y)

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_MELS, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, N_MELS, 5, padding=2)
        )
    def forward(self, x):
        return self.net(x)

train_loader = DataLoader(AudioPairs(train_rows), batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(AudioPairs(val_rows), batch_size=BATCH, shuffle=False, num_workers=0)

model = ConvModel().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()

for epoch in range(EPOCHS):
    model.train()
    total = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        total += loss.item()
    model.eval()
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_total += loss_fn(model(x), y).item()
    print(f"Epoch {epoch+1}/{EPOCHS} | train L1={total/len(train_loader):.4f} | val L1={val_total/max(1,len(val_loader)):.4f}", flush=True)

os.makedirs("outputs", exist_ok=True)
torch.save({
    "model_state_dict": model.cpu().state_dict(),
    "n_mels": N_MELS,
    "frames": FRAMES,
    "sample_rate": SR
}, "outputs/acoustic_baseline.pt")
print("Saved outputs/acoustic_baseline.pt")
print("Note: this predicts log-mel features; a vocoder is still needed for audio output.")
