# Project Architecture

## Active demo: audio/video → Chinese text → English speech

The active entry point is `app.py`. The intended runtime sequence is:

```text
Whispered Chinese audio + matching video
                    |
                    v
         Audio preparation (mono, 16 kHz)
                    |
                    +--------------------------+
                    |                          |
                    v                          v
          Whisper-small + LoRA         Sample video frames
                    |                  (16 grayscale 48x48)
                    |                          |
                    +------------+-------------+
                                 v
                    AVResidualAdapter fusion
                                 |
                                 v
                     Chinese transcription
                                 |
                                 v
                    NLLB Chinese → English
                                 |
                                 v
                      Edge TTS English audio
                                 |
                                 v
                Streamlit transcript/translation/
                         playable audio output
```

## Active model assets

- Whisper LoRA adapter: `outputs/whisper_lora_small/`
- AV transcription adapter: `outputs/av_transcription_full.pt`
- NLLB translation model: `facebook/nllb-200-distilled-600M`
- Edge TTS: remote speech synthesis service

## Inactive speech-conversion experiments

The older whisper-to-normal-speech approach (AV-to-mel prediction and HiFi-GAN vocoding) produced noisy audio and is **not part of the active demo**. Related scripts are kept under `archive/conversion_experiments/` for reference only. Do not call those scripts from the Streamlit app.

The active app generates English speech from the translated English text. This is text-driven TTS, not normal-sounding Chinese speech conversion and not voice preservation.

## Run

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

A hosting-provider port-forward/proxy may be required to open the interface from a local browser. `http://0.0.0.0:8501` is the server bind address, not necessarily the public URL.

## Verification

Syntax and required checkpoint existence have been checked. End-to-end runtime behavior remains unverified until a real matching audio/video pair successfully produces Chinese text, English translation, and playable English audio.
