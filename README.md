 # Audio-Visual Chinese Whisper Transcription → English Speech

A Streamlit application that accepts **whispered Chinese audio plus a matching video**, predicts Chinese text with an audio-visual Whisper pipeline, translates the transcript into English, and synthesizes English speech. The active application uses text-to-speech for English output; it does **not** reconstruct normal-sounding Chinese speech or preserve the speaker's voice.

> **Current demo flow:** Audio + video → Chinese transcription → English translation → English speech.

## Contents

- [Quick start](#quick-start)
- [Architecture and models](#architecture-and-models)
- [Evaluation results](#evaluation-results)
- [Data and reproducibility](#data-and-reproducibility)
- [Training and evaluation commands](#training-and-evaluation-commands)
- [Repository structure](#repository-structure)
- [Known limitations](#known-limitations)

## Quick start

### 1. Get the code

Clone the repository (or open it with VS Code's **Git: Clone** command):

```bash
git clone https://github.com/pradeep14012004/whisper-speech-conversion.git
cd whisper-speech-conversion
```

The repository may be private; sign in with an account that has permission.

### 2. Install dependencies

Use a Python environment compatible with your PyTorch and Transformers versions. Install a PyTorch build appropriate for your CPU/GPU from the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run may download the base Whisper and NLLB checkpoints from Hugging Face. English speech generation uses Edge TTS and needs network access.

### 3. Check the app's model files

The app expects these files in the repository:

- `outputs/whisper_lora_small/` — trained PEFT/LoRA adapter and tokenizer files.
- `outputs/av_transcription_full.pt` — audio-visual adapter checkpoint.

The Whisper base model `openai/whisper-small` and translation model `facebook/nllb-200-distilled-600M` are loaded from Hugging Face as needed. A trained adapter does not replace the base model.

### 4. Launch the app

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (typically `http://localhost:8501`). For a remote server, use its forwarded/public port URL rather than assuming the server bind address is public.

### 5. Use the demo

1. Upload whispered Chinese audio and its corresponding video.
2. Click **Run complete pipeline**.
3. Review the predicted Chinese text and English translation.
4. Listen to, or download, the generated English MP3.

Supported uploads in the app: audio `wav, mp3, flac, ogg, m4a`; video `avi, mp4, mov, mkv, webm`. Audio is limited to 120 seconds by the app.

## Architecture and models

See [ARCHITECTURE.md](ARCHITECTURE.md) and the editable [Mermaid diagram](docs/architecture.mmd).

| Stage | Model/component | Purpose |
|---|---|---|
| Audio ASR backbone | OpenAI Whisper Small, `openai/whisper-small` | Encodes input audio and decodes Chinese text |
| Audio ASR adaptation | PEFT LoRA adapter in `outputs/whisper_lora_small/` | Adapts Whisper-small to this project's whispered-Chinese data |
| Audio-visual fusion | `AVResidualAdapter` defined in `app.py`; checkpoint `outputs/av_transcription_full.pt` | Adds visual frame features to the Whisper encoder representation before decoding |
| Translation | NLLB-200 distilled 600M, `facebook/nllb-200-distilled-600M` | Translates simplified Chinese (`zho_Hans`) into English (`eng_Latn`) |
| English speech | Edge TTS, voice `en-US-AriaNeural` | Synthesizes spoken English from the translated text |
| UI | Streamlit | Upload controls, text results, audio playback and download |

**Whisper LoRA configuration in the training script:** rank `r=8`, alpha `16`, dropout `0.05`, targeting the attention `q_proj` and `v_proj` modules. The training script is a small pilot run by default (up to 32 rows); change the script and carefully validate before treating it as a full training recipe.

**AV adapter structure:** 16 grayscale whole-frame images are resized to `48 × 48`. A small CNN encodes each frame; a 128-dimensional visual representation and 256-dimensional fusion space feed four-head cross-attention. The adapter returns a gated residual correction to the 768-dimensional Whisper-small encoder features. The app samples frames across the video; it does not perform a dedicated mouth/lip crop.

**Inactive experiments:** the older audio-to-mel/acoustic-conversion and HiFi-GAN experiments are under `archive/conversion_experiments/` and are not called by the active Streamlit app. They produced noisy reconstructed audio and should not be described as part of the current translation pipeline.

## Evaluation results

**Read the full methodology and caveats in [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md).** The repository has three official-test ASR prediction CSVs for 3,289 clips. Using the reproducible NFKC/lowercase/alphanumeric normalizer described in that report, corpus CER recomputed from those files is:

| ASR system | Clips | Corpus CER ↓ | Normalized exact matches |
|---|---:|---:|---:|
| Whisper-small (base) | 3,289 | 42.27% | 81 / 3,289 (2.46%) |
| Whisper-small + LoRA | 3,289 | 40.45% | 92 / 3,289 (2.80%) |
| Whisper-large-v3 (baseline comparison) | 3,289 | 18.81% | 798 / 3,289 (24.26%) |

Lower character error rate (CER) is better. These three scores use the same saved official-test clip list and 65,134 normalized reference characters. Whisper-large-v3 is an offline comparison model; **the active app uses Whisper-small + LoRA plus the AV adapter**, not Whisper-large-v3.

A separate historical AV pilot report in `FINAL_RESULTS.txt` evaluates `av_transcription_pilot.pt` (not the checkpoint loaded by the current app). It reports validation CER decreasing from 34.3943% audio-only to 28.1161% AV (2,239 clips), and matched official-test CER decreasing from 40.6707% to 35.5825% (2,274 clips, 69.14% visual-feature coverage). Do **not** attribute these pilot scores to `outputs/av_transcription_full.pt`. A committed benchmark for the app's current full checkpoint is still needed.

These numbers evaluate **Chinese transcription**, not translation quality or generated speech quality. The repository does not currently contain a validated BLEU/COMET translation score, MOS/audio-quality score, or completed end-to-end app evaluation. The comparison script contains CER/WER calculations, but no corresponding WER result artifact is committed.

## Data and reproducibility

The training and evaluation datasets are **not bundled in this Git repository**. On the development server, the local folders measured approximately:

- `data/` — 3.8 GB
- `train/` — 15 GB
- `test/` — 3.5 GB
- `cache16k/` — 6.5 GB

These are ignored by Git because the dataset is large. Check the source dataset license before sharing it. For reproducibility, use a separate dataset host (for example, a private Hugging Face Dataset repository if redistribution is permitted) and document how to obtain it. The app's upload-based demo does not need the entire training dataset; training/evaluation scripts do need their manifests, local audio, and video features.

A Hugging Face token is **not currently wired into the app for automatic dataset download**. Authenticate with `hf auth login` and download the data separately, or add a deliberate download/setup script. Never commit a token or other credentials.

## Training and evaluation commands

Run commands from the repository root. These commands require the matching local manifests/data/features and model checkpoints.

```bash
# Train/update the Whisper-small LoRA adapter (pilot script; GPU optional but slower on CPU)
python train_whisper_lora.py

# Train the expanded AV transcription adapter (requires CUDA and prepared train manifest/features)
python train_av_transcription_full.py

# Compare saved official-test Whisper predictions using the report's normalizer
python scripts/evaluation/score_official_predictions.py

# Legacy scorer: uses optional OpenCC Traditional-to-Simplified conversion
python score_test.py
```

Important: training overwrites or writes files under `outputs/`. Back up any checkpoint you intend to preserve before retraining. The `train_av_transcription_full.py` script expects `outputs/train_av_manifest.csv` and features under `outputs/train_video_features/`. Evaluation scripts refer to their own manifests noted at the top of each file.

## Repository structure

```text
.
├── app.py                         # Active Streamlit app
├── av_fusion.py                   # Shared experimental fusion module
├── project_paths.py               # Project-root path helpers
├── outputs/
│   ├── whisper_lora_small/        # Tracked Whisper LoRA adapter
│   ├── av_transcription_full.pt   # Tracked AV adapter used by app.py
│   └── evaluation/                # Official-test predictions and logs
├── scripts/
│   ├── data_preparation/          # Manifests, audio caches, video features
│   ├── training/                  # LoRA and AV training scripts
│   ├── evaluation/                # Benchmark, scoring and diagnostics scripts
│   ├── translation/               # NLLB helpers
│   └── tts/                       # Standalone Edge TTS helpers
├── archive/conversion_experiments/ # Inactive noisy audio conversion work
├── docs/                          # Architecture, file map and results
└── examples/                      # Example notes
```

Most scripts remain callable through root-level compatibility launchers. See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) and [scripts/README.md](scripts/README.md) for the map.

## Known limitations and caveats

- The current AV adapter's committed full checkpoint has not yet been benchmarked end-to-end in a committed results artifact.
- AV pilot results use a different checkpoint and only the matched subset of the official test set.
- Official-test CER depends on normalization. The new scorer in this repository uses NFKC, lowercase and alphanumeric-only text; the older `score_test.py` optionally converts Traditional Chinese to Simplified first, so its values may differ.
- Video coverage in the pilot official test was 2,274 / 3,289 (69.14%); unmatched examples were not included in AV CER.
- The app uses full grayscale frames, not mouth-region crops. Visual context may be weak or irrelevant.
- Audio-visual fusion is not guaranteed to improve recognition; compare on a speaker-independent split.
- Edge TTS requires network access. The base Whisper and NLLB models require initial model downloads.
