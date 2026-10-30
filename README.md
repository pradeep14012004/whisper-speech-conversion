# Audio-Visual Chinese Whisper Transcription → English Speech

A Streamlit research demo that takes **whispered Chinese audio and a matching video**, generates Chinese transcript candidates with local Whisper/LoRA models and an audio-visual adapter, translates the selected Chinese text into English, and synthesizes English speech.

**Active application:** `app.py`  
**Repository:** https://github.com/pradeep14012004/whisper-speech-conversion

> **Current scope:** Chinese ASR → English text translation → English text-to-speech. This application does **not** convert whispered Chinese into normal-sounding Chinese speech and does not clone the original speaker's voice.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Models and components](#models-and-components)
- [Checkpoint requirements](#checkpoint-requirements)
- [Installation](#installation)
- [Run the application](#run-the-application)
- [Inputs and application behavior](#inputs-and-application-behavior)
- [Training and evaluation](#training-and-evaluation)
- [Recorded ASR results](#recorded-asr-results)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Integrations not currently in main](#integrations-not-currently-in-main)

---

## Pipeline overview

The current `app.py` runs two local Whisper-small paths and an experimental token-level fusion path. Groq is **not** part of the committed application on the `main` branch at the time this README was updated.

~~~mermaid
flowchart TD
    A["Whispered Chinese audio"] --> B["Mono audio, resample to 16 kHz"]
    V["Matching video"] --> C["Sample 16 frames"]
    C --> D["Grayscale, resize to 48 × 48"]

    B --> E["Whisper-small + original LoRA"]
    E --> F["Audio encoder hidden states"]
    D --> G["CNN visual encoder"]
    F --> H["AVResidualAdapter"]
    G --> H
    H --> I["Original LoRA + AV transcript"]

    B --> J["Whisper-small + 2k audio LoRA"]
    J --> K["New audio-only transcript"]

    H --> L["Experimental token-score fusion"]
    J --> L
    L --> M["Fused transcript"]

    I --> N["Local transcript selection"]
    K --> N
    M --> N
    N --> O["Chinese transcript"]
    O --> P["Helsinki-NLP/opus-mt-zh-en"]
    P --> Q["English translation"]
    Q --> R["Edge TTS"]
    R --> S["English speech"]
~~~

### Local transcription paths

1. **Original audio-visual path:** `openai/whisper-small` with the PEFT adapter in `outputs/whisper_lora_small/`. The adapter's audio encoder output is modified by the `AVResidualAdapter` checkpoint at `outputs/av_transcription_full.pt`.
2. **New 2k audio-only path:** a second `openai/whisper-small` model with a fine-tuned adapter expected at `outputs/whisper_lora_aishell6_2k/`.
3. **Experimental token-score fusion:** the app computes both paths' next-token log probabilities and combines them with initial weights of 0.35 for the original AV path and 0.65 for the new audio-only path. These weights are experimental, not validated optimal weights.

The app also generates independent transcripts from the original AV path and new 2k audio-only path. It selects the fused transcript when nonempty, otherwise falls back to the new 2k transcript and then the original AV transcript.

### Audio and video preprocessing

- Audio is read, mixed to mono when multichannel, and resampled to 16 kHz.
- Audio longer than **120 seconds** is rejected.
- The model samples **16 frames** distributed across the video.
- Each model frame is converted to grayscale, resized to **48 × 48**, and scaled to the range 0–1.
- If fewer than 16 frames are decoded, the last valid frame is repeated.
- The current adapter uses whole-frame grayscale inputs; it does not perform a dedicated mouth-region crop or explicit lip-landmark extraction.

### Translation and speech generation

The selected Chinese transcript is passed to `Helsinki-NLP/opus-mt-zh-en` for English translation. The English text is then synthesized with Edge TTS using the voice `en-US-AriaNeural`.

Translation and TTS operate on text; they do not modify the Whisper models or improve the acoustic transcription weights.

---

## Models and components

| Component | Model / path | Purpose |
|---|---|---|
| Base ASR | `openai/whisper-small` | Chinese speech recognition backbone |
| Original ASR adapter | `outputs/whisper_lora_small/` | PEFT LoRA adaptation |
| AV adapter | `outputs/av_transcription_full.pt` | CNN visual encoder, cross-attention, gated residual fusion |
| New audio-only adapter | `outputs/whisper_lora_aishell6_2k/` | Second Whisper-small LoRA prediction path |
| Local fusion | Weighted next-token log probabilities | Experimental combination of AV and audio-only paths |
| Translation | `Helsinki-NLP/opus-mt-zh-en` | Chinese-to-English text translation |
| Speech generation | Edge TTS, `en-US-AriaNeural` | English text-to-speech |
| Deep learning | PyTorch, Transformers, PEFT | Model loading and inference |
| Video | OpenCV | Video decoding and frame preprocessing |
| Audio | Librosa, SoundFile | Audio reading and resampling |
| Interface | Streamlit | Uploads, webcam preview/capture, predictions, playback and downloads |

### AVResidualAdapter

The adapter in `app.py` includes:
- Audio projection from Whisper's 768-dimensional hidden representation to a 256-dimensional fusion space.
- A small CNN that encodes each grayscale frame into 128-dimensional visual features.
- A four-head multi-head attention layer.
- A gated residual projection back into the Whisper audio hidden dimension.

The checkpoint must contain an `av_adapter` state dictionary matching the architecture in `app.py`.

---

## Checkpoint requirements

The active app expects these paths relative to the repository root:

~~~text
outputs/
├── whisper_lora_small/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer / processor files
├── whisper_lora_aishell6_2k/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer / processor files
└── av_transcription_full.pt
~~~

**Important repository state:** the inspected `main` branch contains `outputs/whisper_lora_small/` and `outputs/av_transcription_full.pt`, but the directory `outputs/whisper_lora_aishell6_2k/` was not present in the tracked repository when this README was updated. The current `app.py` checks for that directory and will raise a missing-adapter error until the 2k adapter is supplied locally.

The root-level script `finetune_whisper_lora_2k.py` is configured to create `outputs/whisper_lora_aishell6_2k/`. It requires:
- `data/aishell6_lora_2k/manifest.csv` with train/valid rows, audio paths and transcripts.
- An initial adapter at `outputs/whisper_lora_aishell6_pilot/`.

Those inputs and the generated 2k adapter are not included in the tracked `main` branch directory listing. Obtain or generate them separately before running the active app.

The base Whisper, translation and tokenizer assets are downloaded from Hugging Face on first use and cached locally.

---

## Installation

### 1. Clone the repository

~~~bash
git clone https://github.com/pradeep14012004/whisper-speech-conversion.git
cd whisper-speech-conversion
~~~

### 2. Create and activate an environment

~~~bash
python -m venv .venv
~~~

macOS / Linux:

~~~bash
source .venv/bin/activate
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

### 3. Install PyTorch and project requirements

Install a PyTorch build compatible with your machine first. See the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). The application selects CUDA when available, otherwise Apple MPS when available, and otherwise CPU.

Then install the repository requirements:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The requirements file is a minimal, unpinned runtime list; install a PyTorch version compatible with your Python and hardware. For a production or reproducible experiment, use a tested lock file and record the exact software versions.

### 4. Optional live webcam dependencies

The standard file-upload workflow does not require WebRTC. To enable the optional live webcam/microphone interface, install:

~~~bash
python -m pip install streamlit-webrtc av
~~~

If these optional packages are unavailable, the app can still be used with uploaded audio and video.

---

## Run the application

From the repository root:

~~~bash
python -m streamlit run app.py
~~~

Open the local URL printed by Streamlit, usually:

~~~text
http://localhost:8501
~~~

The first run may download the base Whisper-small and OPUS translation models. Ensure there is sufficient disk space and network access for these downloads.

---

## Inputs and application behavior

### Supported uploaded files

| Input | Formats accepted by the app |
|---|---|
| Audio | WAV, MP3, FLAC, OGG, M4A, AAC |
| Video | AVI, MP4, MOV, MKV, WEBM, MPEG, MPG |

The audio and video should be a matching pair from the same utterance. Unrelated or poorly synchronized video can provide misleading visual information.

### Typical workflow

1. Launch the Streamlit app.
2. Upload whispered Chinese audio and its matching video, or use the optional webcam capture when the WebRTC dependencies are installed.
3. Click **Run complete pipeline**.
4. Inspect the original AV transcript, new 2k audio-only transcript, and experimental fused transcript.
5. Review the selected Chinese text.
6. Review the English translation.
7. Play or download the generated English MP3.

The app can optionally detect local sample files named `hf_test_sample.wav` and `sample_video.mp4` in the repository root. These are not required; the normal workflow is to upload a matching pair.

The application automatically chooses the inference device:
- CUDA, if available.
- Apple MPS, if available.
- CPU otherwise.

Training scripts may have stricter hardware requirements than inference scripts; inspect the relevant script before running it.

---

## Training and evaluation

Run scripts from the repository root. Training and offline evaluation require the corresponding datasets, manifests, video features and checkpoints, which are not all included in the repository.

### Train the original Whisper LoRA

~~~bash
python scripts/training/train_whisper_lora.py
~~~

The script reads `outputs/training_manifest.csv` and writes the adapter to `outputs/whisper_lora_small/`. Review the script's sampling and training settings before using it for a full experiment.

### Train the full AV adapter

~~~bash
python scripts/training/train_av_transcription_full.py
~~~

This script expects CUDA and prepared training manifests/video features. It uses the original Whisper LoRA adapter and writes `outputs/av_transcription_full.pt`. Confirm the input paths and training configuration in the script before starting.

### Fine-tune the 2k audio LoRA

~~~bash
python finetune_whisper_lora_2k.py
~~~

This script requires the AISHELL-6 2k manifest and the initial pilot adapter described in [Checkpoint requirements](#checkpoint-requirements). It writes `outputs/whisper_lora_aishell6_2k/`, the second adapter expected by the active app.

### Recompute official-test ASR metrics

~~~bash
python scripts/evaluation/score_official_predictions.py
~~~

The scorer uses the tracked official-test prediction CSVs, checks the clip/reference alignment, and reports CER, exact matches, empty outputs and per-speaker CER.

A legacy scorer is also available:

~~~bash
python score_test.py
~~~

The legacy scorer may apply OpenCC Traditional-to-Simplified normalization. Do not mix its scores with the NFKC/alphanumeric-normalized scores below.

---

## Recorded ASR results

The repository contains three saved official-test prediction CSVs, each with **3,289 clips** and **65,134 normalized reference characters**. The reported figures were recomputed using:
1. Unicode NFKC normalization.
2. Lowercasing.
3. Retaining Unicode letters and numbers while removing punctuation, whitespace and symbols.
4. Corpus CER = total character edit distance divided by total normalized reference characters.

| ASR model / saved predictions | Test clips | Corpus CER ↓ | Exact matches |
|---|---:|---:|---:|
| Whisper-small base | 3,289 | 42.27% | 81 (2.46%) |
| Whisper-small + project LoRA | 3,289 | 40.45% | 92 (2.80%) |
| Whisper-large-v3 base comparison | 3,289 | 18.81% | 798 (24.26%) |

On these saved predictions, the small LoRA improves corpus CER by about 1.82 percentage points compared with unadapted Whisper-small. Whisper-large-v3 performs best in this saved comparison, but it is an **offline comparison model**, not the model loaded by the active Streamlit application.

### Per-speaker corpus CER range

| Model | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Whisper-small base | 28.80% | 39.24% | 80.34% |
| Whisper-small + LoRA | 27.57% | 38.93% | 70.07% |
| Whisper-large-v3 | 5.60% | 17.02% | 46.64% |

These values show variation across 28 speaker groups. They do not, by themselves, establish generalization to unseen speakers.

### Historical AV pilot results — separate checkpoint

The historical AV pilot report records the following results for `outputs/av_transcription_pilot.pt`, **not** the active app checkpoint `outputs/av_transcription_full.pt`:

| Split | Audio-only CER | AV CER | Coverage |
|---|---:|---:|---:|
| Validation | 34.3943% | 28.1161% | 2,239 clips |
| Official-test matched subset | 40.6707% | 35.5825% | 2,274 / 3,289 clips (69.14%) |

The pilot report records a 6.2782 percentage-point validation CER reduction (18.25% relative) and an approximately 5.0882 percentage-point reduction on the matched official-test subset (approximately 12.51% relative).

**Do not attribute these pilot scores to `outputs/av_transcription_full.pt`.** The matched official-test AV result excludes examples without available visual features and may have selection bias. A dedicated benchmark for the active full AV checkpoint is still needed.

See [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md) and [FINAL_RESULTS.txt](FINAL_RESULTS.txt) for additional details and caveats.

### Translation and TTS evaluation status

The repository does not currently provide validated BLEU, chrF, COMET, human translation scores, MOS/listener study results, objective speech-quality scores, or a completed end-to-end benchmark for transcription → translation → speech. ASR CER does not measure translation accuracy or generated-speech quality.

---

## Repository layout

~~~text
whisper-speech-conversion/
├── app.py
├── app_audio_only.py
├── app_av_2k_test.py
├── requirements.txt
├── requirements-server-lock.txt
├── ARCHITECTURE.md
├── FINAL_RESULTS.txt
├── av_fusion.py
├── finetune_whisper_lora_2k.py
├── train_av_aishell6_2k.py
├── transcribe.py
├── scripts/
│   ├── data_preparation/
│   ├── training/
│   ├── evaluation/
│   ├── translation/
│   └── tts/
├── docs/
│   ├── EVALUATION_RESULTS.md
│   ├── PROJECT_STRUCTURE.md
│   └── architecture.mmd
├── outputs/
│   ├── whisper_lora_small/
│   ├── av_transcription_full.pt
│   └── evaluation/
└── archive/
    └── conversion_experiments/
~~~

Other root-level scripts are retained as experiment entry points or compatibility launchers. The repository also contains `train_macbert_asr_corrector.py` as a separate/legacy experiment; the active `app.py` does not load MacBERT.

---

## Known limitations

- The active app requires the 2k LoRA adapter directory, which was not present in the inspected tracked `main` branch at the time this README was updated.
- The active full AV checkpoint has not yet been benchmarked in a committed, dedicated evaluation.
- The historical AV pilot results use a different checkpoint and incomplete official-test visual coverage.
- Whole-frame grayscale inputs are a coarse visual representation; the current app does not detect or crop the mouth region.
- The fusion weights (0.35 AV / 0.65 audio-only) are experimental and have not been established as optimal.
- Different models can disagree. A nonempty output is not necessarily a correct transcript; review predictions against the source audio.
- Translation and English TTS quality have not been quantitatively evaluated in the committed results.
- Edge TTS requires network access.
- The base Whisper-small and OPUS translation models may need to be downloaded on first use.
- Full datasets, manifests and intermediate video features are not all included. Obtain data separately and follow dataset licenses.
- The app's English speech is synthesized from translated text; it is not voice conversion and does not preserve the original speaker's identity or voice.

---

## Integrations not currently in main

The committed `app.py` and `requirements.txt` inspected for this README do **not** include Groq API calls, Groq Whisper transcription, or Groq text correction. No `GROQ_API_KEY` is required by the current committed app.

If you have a separate local Option A/Groq-enabled version, commit and push that implementation and its dependency changes before documenting Groq as an active repository feature.

---

## Acknowledgements and model usage

This project uses OpenAI Whisper, Hugging Face Transformers and PEFT, PyTorch, OpenCV, Librosa, SoundFile, Streamlit, Helsinki-NLP OPUS-MT and Edge TTS.

Review each model's license, usage terms and the training dataset's license before redistributing weights or data.
