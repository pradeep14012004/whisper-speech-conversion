---
library_name: peft
base_model: openai/whisper-small
pipeline_tag: automatic-speech-recognition
---

# Whisper-small LoRA adapter for whispered Chinese ASR

## Summary

This directory contains the project's PEFT/LoRA adapter and tokenizer/processor metadata for adapting `openai/whisper-small` to Chinese transcription. The adapter is used by the Streamlit application together with a separate audio-visual residual adapter.

This is an adapter, **not a complete standalone Whisper model**. The base model must be available from Hugging Face or its local cache.

## Intended use

- Transcribe Chinese speech from the project's whispered-audio/video pipeline.
- Serve as the audio ASR component before the AV adapter, translation, and English TTS stages.
- Run controlled comparisons against the unadapted Whisper-small base model.

## Model components

- Base model: `openai/whisper-small`
- Adapter format: PEFT LoRA
- Training script: `scripts/training/train_whisper_lora.py`
- Adapter directory: `outputs/whisper_lora_small/`
- Related AV checkpoint loaded by the app: `outputs/av_transcription_full.pt`

The training script configures LoRA rank 8, alpha 16, dropout 0.05, targeting attention `q_proj` and `v_proj`. These are script settings; exact training provenance should be taken from run logs and checkpoint metadata when available.

## Use

From the repository root, create an environment, install a compatible PyTorch build and project requirements, then launch the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application loads this adapter and the separate AV checkpoint automatically. See the top-level [README](../../README.md) and [architecture](../../ARCHITECTURE.md) for the complete flow.

## Evaluation notes

The tracked official-test predictions report normalized corpus CER of 40.45% for Whisper-small plus this LoRA adapter, versus 42.27% for the unadapted Whisper-small base, on 3,289 saved test clips. See [evaluation methodology and results](../../docs/EVALUATION_RESULTS.md) for normalization details and the separate Whisper-large-v3 comparison.

These are ASR results only. The active app's AV adapter is a separate checkpoint; historical AV pilot scores must not be attributed to this adapter or to the app's full AV checkpoint.

## Limitations, risks, and license

Performance can vary across speakers, recording conditions, and transcription domains. The current visual component uses full grayscale frames rather than mouth-region crops. Do not assume the LoRA adapter improves every utterance or generalizes to unseen speakers.

Check the license and usage terms of the base Whisper model and the training dataset before redistributing weights or deploying this adapter. The project repository does not assert a separate license for the training data.
