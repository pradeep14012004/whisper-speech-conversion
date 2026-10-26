# Audio-Visual Chinese Transcription → English Speech

This project takes whispered Chinese audio and a matching video, predicts Chinese text, translates it into English, and generates English speech. The active app does not use the archived acoustic-conversion / HiFi-GAN experiments.

## Start here

- **App entry point:** `app.py`
- **Active model components:** `av_fusion.py`, `outputs/whisper_lora_small/`, and `outputs/av_transcription_full.pt`
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Project structure:** [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- **Script directory guide:** [scripts/README.md](scripts/README.md)

## Active pipeline

1. Upload whispered Chinese audio and its matching video in Streamlit.
2. Prepare audio (mono, 16 kHz) and sample video frames.
3. Transcribe Chinese with Whisper-small + LoRA and the AV adapter.
4. Translate Chinese text into English with NLLB-200.
5. Generate English speech using Edge TTS.
6. Display the Chinese transcript, English translation, and generated audio.

## Run the app

From the repository root, with dependencies installed:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

The bind address `0.0.0.0:8501` is not itself the public browser URL; use the URL exposed by your hosting provider.

## Repository map

```text
.
├── app.py                         # Active Streamlit inference entry point
├── av_fusion.py                   # AV model components used by the app
├── project_paths.py               # Shared project path helpers
├── outputs/
│   ├── whisper_lora_small/        # Trained Whisper LoRA adapter
│   ├── av_transcription_full.pt   # Trained AV adapter checkpoint
│   └── evaluation/                # Evaluation logs and predictions
├── scripts/
│   ├── data_preparation/          # Manifests, caches, video features
│   ├── training/                  # Whisper and AV training scripts
│   ├── evaluation/                # Evaluation, comparisons, diagnostics
│   ├── translation/               # Translation helper implementations
│   └── tts/                       # English speech-generation utilities
├── archive/
│   └── conversion_experiments/    # Inactive acoustic/mel/HiFi-GAN experiments
├── docs/                          # Architecture and project documentation
├── examples/                      # Example notes
└── *.py                            # Compatibility launchers and small utilities
```

Most older data-preparation, training, evaluation, and translation command names remain as small root-level compatibility launchers, so prior commands continue to work while their implementations live under `scripts/`. The old acoustic-conversion entry points at the root now stop with a message pointing to their archived implementations.

## Model and service dependencies

- Whisper base model: `openai/whisper-small` (downloaded by the runtime as needed)
- LoRA adapter: `outputs/whisper_lora_small/`
- AV adapter checkpoint: `outputs/av_transcription_full.pt`
- Translation model: `facebook/nllb-200-distilled-600M`
- English TTS: Edge TTS; requires network access to the service

Check that checkpoints exist in the environment before launching. Some datasets, caches, and evaluation artifacts may be intentionally ignored by Git.

## What is not in the active pipeline

`archive/conversion_experiments/` stores inactive acoustic-conversion, mel, and HiFi-GAN experiments. The active demo generates English speech from translated text; it does not restore normal-sounding Chinese speech or preserve the original speaker's voice.

## Verification status and limitations

Syntax and app-import checks passed in the development environment, but a real audio/video inference run is still needed to verify the complete path from input to playable English speech. Do not interpret the organization of files as proof of end-to-end runtime success.

The current visual preprocessing uses whole grayscale frames rather than a dedicated mouth crop. Audio-visual fusion is not guaranteed to improve accuracy; compare against an audio-only baseline.
