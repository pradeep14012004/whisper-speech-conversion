# Repository File Structure

This file explains where each part of the project belongs and which files are safe to move. The live app and training scripts currently use paths relative to the repository root, so keep the repository root as the working directory when running scripts.

## Active entry point (keep at root)

- `app.py` — Streamlit application. Keep this at the root because it locates model files relative to its own directory.
- `outputs/whisper_lora_small/` — local Whisper LoRA files required by the app (may be ignored by Git).
- `outputs/av_transcription_full.pt` — local AV adapter checkpoint required by the app (may be ignored by Git).
- `outputs/evaluation/` — saved evaluation logs and predictions.
- `README.md`, `ARCHITECTURE.md` — project overview and current architecture.

## Documentation

- `docs/architecture.mmd` — Mermaid diagram of the active pipeline.
- `examples/README.md` — example usage notes.

## Data preparation and feature extraction

These scripts prepare manifests, paired data, caches, or video features. They currently refer to paths such as `outputs/...` relative to the repository root:

- `build_manifest.py`
- `build_manifest_all.py`
- `build_av_manifest.py`
- `build_conversion_manifest.py`
- `build_pairs.py`
- `build_training_manifest.py`
- `prepare_training.py`
- `make_cache16k.py`
- `extract_video_features.py`
- `extract_train_video_features.py`
- `extract_test_video_features.py`

## Training

Training scripts are kept at the root for now because they refer to root-relative paths and may share local modules:

- `train_whisper_lora.py`
- `train_lora_large_v3.py`
- `train_baseline.py`
- `train_av_transcription.py`
- `train_av_transcription_full.py`
- `train_av_transcription_lowlr.py`
- `train_av_fusion.py`
- `train_av_to_mel.py`

## Evaluation and diagnostics

- `evaluate_*.py`
- `eval_*.py`
- `compare_*.py`
- `score_test.py`
- `rescore.py`, `rescore3.py`
- `smoke_test_av.py`, `test_adapter.py`, `test_av_decode.py`
- `validate_beam_search.py`, `validate_decoding.py`, `validate_conversion.py`

## Translation and speech utilities

- `translate_one.py`
- `translate_to_english.py`
- `english_tts.py`
- `english_tts_one.py`
- `run_chinese_tts.sh`
- `run_demo_pipeline.sh`
- `run_full_demo.sh`

## Archived conversion experiments

- `archive/conversion_experiments/` — inactive mel / HiFi-GAN / acoustic-conversion experiments. These are not part of the active Streamlit pipeline.
- The active app should remain: audio + video → Chinese text → English translation → English TTS.

## Legacy / large files

- `backup_parts/` — legacy backup fragments.
- `new-workspace.ipynb` — notebook.
- `nlp_project_results.tar.gz` — archive; consider storing large archives outside Git if they are not needed for source control.

## Important path rule

Until scripts are converted to use a shared project-root helper, run them from the repository root, for example:

```bash
cd /home/jovyan/whisper_speech_conversion
python train_av_transcription_full.py
```

Do not move scripts into subfolders without updating their path handling and testing their imports. The current organization is documented by role to avoid silently breaking relative paths.
