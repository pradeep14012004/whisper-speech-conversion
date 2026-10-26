# Audio-Visual Chinese Transcription → English Speech

This project takes whispered Chinese audio and a matching video, predicts Chinese text, translates it into English, and generates English speech. The active app does not use the archived acoustic-conversion / HiFi-GAN experiments.

## Start here

- **Run the app:** `app.py`
- **Architecture overview:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Project file map:** [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- **Model files:** `outputs/whisper_lora_small/` and `outputs/av_transcription_full.pt`

## Active pipeline

1. Upload whispered Chinese audio and its matching video in Streamlit.
2. Prepare audio (mono, 16 kHz) and sample video frames.
3. Transcribe Chinese with Whisper-small + LoRA and the AV adapter.
4. Translate Chinese text into English with NLLB-200.
5. Generate English speech using Edge TTS.
6. Display the Chinese transcript, English translation, and audio output.

## Run the app

From the repository root, with dependencies installed:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

The bind address `0.0.0.0:8501` is not itself the public browser URL; use the URL exposed by your hosting provider.

## Repository layout

```text
.
├── app.py                         # Streamlit app / active inference entry point
├── av_fusion.py                   # AV model components
├── project_paths.py               # Shared project path helpers
├── outputs/
│   ├── whisper_lora_small/        # Trained Whisper LoRA adapter
│   ├── av_transcription_full.pt   # Trained AV adapter checkpoint
│   └── evaluation/                # Evaluation logs and predictions
├── scripts/
│   ├── data_preparation/          # Manifests and video features
│   ├── training/                  # Whisper/AV training scripts
│   ├── evaluation/                # Evaluation and decoding checks
│   └── translation/               # Translation helpers
├── archive/
│   └── conversion_experiments/    # Inactive mel / acoustic / HiFi-GAN experiments
├── docs/                          # Architecture and project documentation
├── examples/                      # Example notes
├── backup_parts/                  # Legacy fragments
└── *.py                            # Compatibility launchers and standalone utilities
```

## Model and service dependencies

- Whisper base model: `openai/whisper-small` (downloaded by the runtime as needed)
- LoRA adapter: `outputs/whisper_lora_small/`
- AV adapter checkpoint: `outputs/av_transcription_full.pt`
- Translation model: `facebook/nllb-200-distilled-600M`
- English TTS: Edge TTS; requires network access to the service

Check that checkpoints exist in the environment before launching. Some outputs and data files may be intentionally ignored by Git.

## What is not in the active pipeline

`archive/conversion_experiments/` stores inactive acoustic conversion, mel, and HiFi-GAN experiments. The active demo generates English speech from translated text; it does not restore normal-sounding Chinese speech or preserve the original speaker's voice.

## Verification status and limitations

Syntax and app-import checks have passed in the development environment, but a real audio/video inference run is still needed to verify the complete path from input to playable English speech. Do not interpret the organization of files as proof of end-to-end runtime success.

The current visual preprocessing uses whole grayscale frames rather than a dedicated mouth crop. Audio-visual fusion is not guaranteed to improve accuracy; compare against an audio-only baseline.
