# Audio-Visual Chinese Transcription → English Speech

This repository's **active demo pipeline** takes whispered Chinese audio and a matching video, predicts Chinese text, translates that text into English, and generates English speech. It does **not** use the noisy Chinese speech-reconstruction / HiFi-GAN experiment in the active app.

## Active pipeline

1. **Inputs** — upload whispered Chinese audio and the matching video in Streamlit.
2. **Audio preparation** — load the audio, convert it to mono, and resample to 16 kHz when needed.
3. **Video preparation** — sample 16 frames, convert them to grayscale, and resize to 48 × 48.
4. **Audio-visual Chinese transcription** — Whisper-small + LoRA encodes the audio; the trained AV adapter fuses the video-frame features with the audio representations and predicts Chinese text.
5. **Chinese → English translation** — NLLB-200 translates the recognized Chinese text to English.
6. **English speech generation** — Edge TTS generates spoken English.
7. **Results** — Streamlit displays the Chinese transcript, English translation, and generated English audio.

## Entry point and required model files

- **App entry point:** `app.py`
- Whisper LoRA adapter: `outputs/whisper_lora_small/`
- Audio-visual transcription adapter checkpoint: `outputs/av_transcription_full.pt`
- Translation model: `facebook/nllb-200-distilled-600M` (downloaded by Transformers as needed)
- English TTS: Edge TTS (requires network access to the service)

The model checkpoint files may be kept on the server and may be excluded from GitHub by `.gitignore`. Confirm they are present in your runtime before launching the app.

## Run the app

From the repository root, in the server environment with dependencies installed:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Open the app using your hosting provider's exposed/forwarded port URL. `0.0.0.0:8501` is a bind address, not normally the public browser URL.

## Repository guide

- `app.py` — active Streamlit application and inference pipeline.
- `outputs/` — local model checkpoints and evaluation artifacts; some are intentionally not tracked by Git.
- `docs/architecture.mmd` and `ARCHITECTURE.md` — current pipeline diagram and explanation.
- `train_av_transcription*.py`, `train_whisper_lora.py`, `train_av_fusion.py`, `evaluate_*.py` — model training and evaluation scripts; run only when intentionally training/evaluating.
- `build_*.py`, `extract_*video*.py` — data-manifest and video-feature preparation utilities.
- `archive/conversion_experiments/` — inactive acoustic-conversion / mel / HiFi-GAN experiments retained for reference; not part of the active demo.
- `backup_parts/` — legacy backup fragments; not part of runtime.

## Verification status

The app syntax and required checkpoint presence have been checked in the server environment. A real audio/video inference run is still required to confirm the complete runtime path (Chinese transcription → English translation → playable English speech). Do not claim end-to-end success until that test passes.

## Notes

- Video input must correspond to the whispered audio.
- The current visual preprocessing uses whole grayscale frames, not a dedicated mouth crop.
- Audio-visual fusion is not guaranteed to improve transcription accuracy; evaluate against an audio-only baseline.
- The active pipeline generates English speech from translated text. It does not preserve the original speaker's voice and does not convert whispered Chinese audio into normal-sounding Chinese speech.
