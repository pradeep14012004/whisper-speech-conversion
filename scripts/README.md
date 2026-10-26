# Scripts Guide

Run commands from the repository root. The subfolders contain implementations; root-level launchers are kept for compatibility.

- `data_preparation/`: manifests, data checks, audio caches, and video feature extraction.
- `training/`: Whisper LoRA and AV adapter training.
- `evaluation/`: evaluation, model comparisons, smoke tests, rescoring, and decoding diagnostics.
- `translation/`: Chinese-to-English translation helpers.
- `tts/`: standalone English speech-generation utilities.

The active Streamlit app is `../app.py`. It is intended to run audio/video → Chinese transcription → English translation → English TTS. Archived acoustic-conversion/HiFi-GAN experiments under `../archive/conversion_experiments/` are not part of the active app.

## Common commands

```bash
# Launch the interactive app
python -m streamlit run app.py

# Recompute the documented CER metrics from tracked official-test predictions
python scripts/evaluation/score_official_predictions.py

# Translate one Chinese text and write outputs/demo_english.txt
python translate_one.py "你好"

# Generate one English speech file from the translated text
python english_tts_one.py
```

Training and data-preparation commands require their manifests, datasets, and often a CUDA GPU; inspect each script's constants and required input files before running. Use [the evaluation report](../docs/EVALUATION_RESULTS.md) for metrics and checkpoint caveats.

For the complete file map, see [../docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md).
