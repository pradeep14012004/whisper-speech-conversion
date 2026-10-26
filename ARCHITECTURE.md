# Whispered Chinese Speech Project Architecture

## Current demo pipeline

Chinese transcript
    |
    v
Pretrained Chinese neural TTS (Edge TTS)
    |
    v
outputs/chinese_tts.mp3
    |
    v
Audio playback and evaluation
    |
    v
Optional Chinese ASR -> English translation -> English TTS

## Speech-generation decision

The AV-to-mel model currently produces noisy audio when its
predicted mel-spectrogram is passed to HiFi-GAN.

For the demo speech-generation path, use pretrained Chinese TTS
directly from the Chinese transcript. Do not use the noisy AV
mel prediction as the demo audio source.

## Existing AV model

Keep the AV model, video features, LoRA adapter, training scripts,
and checkpoints for further research and debugging.

The TTS fallback is transcript-driven speech synthesis. It is not
whisper-to-normal speech conversion and does not demonstrate
preservation of the original speaker's voice.

## Optional English output

If a Chinese transcript is already available:
    Chinese transcript -> English translation -> English TTS

If only audio is available:
    Chinese audio -> Chinese ASR -> English translation -> English TTS

## Run the Chinese TTS demo

    ./run_chinese_tts.sh

The script requires internet access and a working Edge TTS service.
