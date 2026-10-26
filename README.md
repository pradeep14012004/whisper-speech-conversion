# Audio-Visual Whispered Chinese Speech Conversion

## Project overview

This project aims to convert whispered Chinese speech into normal-sounding
Chinese speech using audio-visual features, then recognize the converted
Chinese speech, translate it into English, and synthesize English speech.

## Architecture

1. Input: whispered Chinese audio and synchronized video.
2. Audio features: Whisper-small with LoRA adapter.
3. Visual features: precomputed video-frame features.
4. Fusion: cross-attention combines audio and visual representations.
5. Speech generation: mel decoder and HiFi-GAN vocoder.
6. Chinese ASR: recognize the converted Chinese speech.
7. Machine translation: translate Chinese text into English.
8. English TTS: synthesize English speech.
9. Demo interface: display results and play generated audio.

## Current implementation status

The project contains an existing acoustic conversion model and an
audio-visual-to-mel training pipeline. The full inference pipeline must
be tested before claiming that every stage works end to end.

## Requirements

Python and compatible versions of PyTorch, Transformers, PEFT,
SpeechBrain, NumPy, pandas, SciPy, SoundFile, and other dependencies
used by the project.

Install dependencies using the project's verified requirements file
once it has been prepared.

## Model files

Model checkpoints and datasets may be too large for ordinary GitHub
uploads. Store large files using Git LFS or an approved external model
storage location. Do not commit private data, credentials, or tokens.

## Classroom demonstration

The intended demonstration accepts whispered Chinese audio and video,
generates converted Chinese speech, and then displays English translation
and English speech when those pipeline stages have been integrated
and tested.

## Reproducibility

Record the Python and library versions, model checkpoint locations,
input sample details, and validation results before presenting results.
