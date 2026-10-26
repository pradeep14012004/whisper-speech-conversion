# Project Structure

This guide describes where project components live, which files are active, and which folders contain experiments. Run commands from the repository root unless the script explicitly documents otherwise.

## Root files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point and active inference pipeline |
| `av_fusion.py` | Audio-visual adapter/model components |
| `project_paths.py` | Shared path helpers |
| `README.md` | Setup and quick overview |
| `ARCHITECTURE.md` | Active pipeline explanation |
| `FINAL_RESULTS.txt` | Recorded project results |
| `*.py` launchers | Compatibility wrappers for scripts organized under `scripts/` |

## Organized scripts

### `scripts/data_preparation/`

Manifest building, paired-data preparation, and video feature extraction:
- `build_av_manifest.py`
- `build_manifest.py`, `build_manifest_all.py`, `build_training_manifest.py`
- `build_pairs.py`
- `extract_video_features.py`, `extract_train_video_features.py`, `extract_test_video_features.py`

### `scripts/training/`

Model training code:
- `train_whisper_lora.py`
- `train_lora_large_v3.py`
- `train_baseline.py`
- `train_av_transcription.py`, `train_av_transcription_full.py`, `train_av_transcription_lowlr.py`

### `scripts/evaluation/`

Evaluation, rescoring, official-test runs, and decoding validation. Files are grouped here by evaluation purpose; they are not necessarily part of the app's runtime.

### `scripts/translation/`

- `translate_one.py`
- `translate_to_english.py`

## Model outputs

- `outputs/whisper_lora_small/` — trained Whisper LoRA adapter and tokenizer metadata.
- `outputs/av_transcription_full.pt` — trained audio-visual transcription adapter checkpoint.
- `outputs/evaluation/` — saved evaluation logs and prediction files.

Keep large datasets, caches, generated audio, and other machine-specific artifacts out of source control unless they are deliberately needed and safe to publish.

## Archived and legacy material

- `archive/conversion_experiments/` — inactive mel/acoustic-conversion/HiFi-GAN experiments. These are not part of the intended active app pipeline.
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

The intended active path bypasses archived acoustic-conversion experiments. The complete path still needs a real audio/video inference test to verify runtime behavior.

## Path and execution rule

Run from the repo root so existing relative paths resolve as expected:

```bash
cd /home/jovyan/whisper_speech_conversion
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

The organized scripts may be invoked through root-level compatibility launchers to preserve older commands. Do not delete those launchers unless every reference and workflow has been updated and tested.
