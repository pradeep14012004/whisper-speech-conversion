# Project Structure

Use the repository root as the working directory for existing commands. Root-level compatibility launchers remain so earlier commands continue to work, while most implementations are grouped by purpose under `scripts/`.

## Root files

| Path | Purpose |
|---|---|
| `app.py` | Active Streamlit application and inference entry point |
| `av_fusion.py` | Audio-visual model components |
| `project_paths.py` | Shared project path helpers |
| `README.md` | Setup and quick overview |
| `ARCHITECTURE.md` | Active pipeline explanation |
| `FINAL_RESULTS.txt` | Recorded project results |
| `outputs_README_NOTE.md` | Notes about model/output artifacts |
| root-level `.py` launchers | Backward-compatible command names that delegate into `scripts/`; archived conversion entry points now show a safe warning |

## Organized scripts

### `scripts/data_preparation/`

Manifest building, training-sample preparation, 16 kHz audio caches, and video feature extraction. Includes:
- `build_av_manifest.py`, `build_manifest.py`, `build_manifest_all.py`
- `build_pairs.py`, `build_training_manifest.py`
- `make_cache16k.py`, `prepare_training.py`
- `extract_video_features.py`, `extract_train_video_features.py`, `extract_test_video_features.py`

### `scripts/training/`

Model training implementations:
- `train_whisper_lora.py`
- `train_lora_large_v3.py`
- `train_baseline.py`
- `train_av_transcription.py`, `train_av_transcription_full.py`, `train_av_transcription_lowlr.py`

### `scripts/evaluation/`

Evaluation, official-test runs, rescoring, comparisons, smoke tests, adapter checks, and decoding validation. These scripts are primarily for development/evaluation, not part of normal app inference.

### `scripts/translation/`

- `translate_one.py`
- `translate_to_english.py`

### `scripts/tts/`

- `english_tts.py` — generate English speech for a translation CSV.
- `english_tts_one.py` — generate a single English speech output.

## Model outputs

- `outputs/whisper_lora_small/` — trained Whisper LoRA adapter and tokenizer metadata.
- `outputs/av_transcription_full.pt` — trained audio-visual transcription adapter checkpoint.
- `outputs/evaluation/` — saved evaluation logs and prediction files.

Keep datasets, caches, generated audio, and machine-specific artifacts out of source control unless they are deliberately needed and safe to publish.

## Archived and legacy material

- `archive/conversion_experiments/` — inactive mel, acoustic-conversion, and HiFi-GAN experiments, including `build_conversion_manifest.py` and `synthesize_deadline_preview.py`. These are not part of the active Streamlit pipeline.
- Root-level `build_conversion_manifest.py` and `synthesize_deadline_preview.py` now print an archive notice and stop; they do not execute the old conversion pipeline.
- `backup_parts/` — legacy backup fragments; not used by the app.
- `new-workspace.ipynb` — exploratory notebook.
- `nlp_project_results.tar.gz` — bundled archive; retain only if needed for reproducibility.

## Intended active app flow

```text
Whispered Chinese audio + matching video
                 |
                 v
Audio/video preprocessing
                 |
                 v
Whisper-small + LoRA + AV adapter
                 |
                 v
           Chinese text
                 |
                 v
           NLLB translation
                 |
                 v
           English text
                 |
                 v
              Edge TTS
                 |
                 v
            English audio
```

The intended active path bypasses archived acoustic-conversion experiments. A real audio/video inference test is still required to verify the full runtime path.

## Path and execution rule

Run from the repository root so existing relative data/output paths resolve as expected:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Root-level compatibility launchers delegate to the organized implementations. Keep them unless all existing commands and workflows have been updated and tested.
