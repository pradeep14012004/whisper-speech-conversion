# Scripts Guide

Run commands from the repository root. The folders contain the implementations; root-level launchers are kept for compatibility.

- `data_preparation/`: manifests, training data checks, audio cache preparation, and video feature extraction.
- `training/`: Whisper LoRA and audio-visual adapter training.
- `evaluation/`: evaluation runs, comparisons, smoke tests, rescoring, and decoding diagnostics.
- `translation/`: Chinese-to-English translation helpers.
- `tts/`: English speech generation utilities.

The active Streamlit app is `../app.py`. It performs the intended audio/video → Chinese transcription → English translation → English TTS flow. Acoustic conversion and HiFi-GAN experiments live in `../archive/conversion_experiments/` and are not part of the active app.

Examples:
```bash
python -m streamlit run app.py
python compare_whisper.py
python english_tts_one.py
```

For the full file map and archived-component notes, see [../docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md).
