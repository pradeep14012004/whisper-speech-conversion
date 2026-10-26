# Evaluation Results

This report separates **recomputed official-test ASR results** from **historical AV pilot results**. These results come from different checkpoints and should not be combined into a single benchmark.

## 1. Active checkpoint versus historical pilot

The current app in `app.py` loads `outputs/whisper_lora_small/` and `outputs/av_transcription_full.pt`.

The historical AV pilot report in `FINAL_RESULTS.txt` refers to `outputs/av_transcription_pilot.pt`, a different checkpoint that is not loaded by the current app. Pilot scores below must not be attributed to the full checkpoint.

## 2. Official-test ASR results recomputed from tracked CSVs

Three committed prediction artifacts contain the same 3,289 official-test clips. Metrics were recomputed from the saved CSVs with this consistent normalization:

1. Apply Unicode NFKC normalization.
2. Convert text to lowercase.
3. Keep Unicode letters and numbers; remove punctuation, whitespace, and symbols.
4. Calculate corpus CER as total Levenshtein character edits divided by total normalized reference characters.
5. Count normalized exact matches where prediction equals reference after normalization.

All three results use 65,134 normalized reference characters and have zero empty predictions.

| ASR model / artifact | Clips | Corpus CER (lower is better) | Exact matches |
|---|---:|---:|---:|
| Whisper-small base — `outputs/evaluation/official_test_small_predictions.csv` | 3,289 | 42.27% | 81 / 3,289 (2.46%) |
| Whisper-small + project LoRA — `outputs/evaluation/official_test_lora_predictions.csv` | 3,289 | 40.45% | 92 / 3,289 (2.80%) |
| Whisper-large-v3 base comparison — `outputs/evaluation/official_test_large_v3_predictions.csv` | 3,289 | 18.81% | 798 / 3,289 (24.26%) |

On these saved predictions, LoRA improves Whisper-small corpus CER by about 1.82 percentage points. Whisper-large-v3 is an **offline comparison model**, not the ASR backbone used by the Streamlit app.

### Per-speaker corpus CER range

Speaker groups were inferred from the clip path component immediately after `whisper/`; 28 speaker groups were present.

| Model | Minimum speaker CER | Median speaker CER | Maximum speaker CER |
|---|---:|---:|---:|
| Whisper-small base | 28.80% | 39.24% | 80.34% |
| Whisper-small + LoRA | 27.57% | 38.93% | 70.07% |
| Whisper-large-v3 | 5.60% | 17.02% | 46.64% |

These ranges demonstrate variation across speakers; they do not alone establish generalization to unseen speakers.

## 3. Historical audio-only versus AV pilot results

The existing `FINAL_RESULTS.txt` reports the following results for the historical pilot checkpoint `outputs/av_transcription_pilot.pt`.

| Split / experiment | Audio-only CER | AV CER | Coverage |
|---|---:|---:|---|
| Validation | 34.3943% | 28.1161% | 2,239 clips |
| Official-test matched subset | 40.6707% | 35.5825% | 2,274 / 3,289 (69.14%) |

Reported pilot reductions and comparisons:

- Validation: 6.2782 percentage points absolute CER reduction; 18.25% relative reduction.
- Matched official test: approximately 5.0882 percentage points absolute reduction; approximately 12.51% relative reduction.
- Validation utterances: AV improved 1,147 (51.2%), worsened 556 (24.8%), and was unchanged for 536 (23.9%).
- Alternative runs: low-learning-rate AV validation CER 33.2322%; expanded-training AV validation CER 37.9886%. The historical report selected the original pilot.

**Important limitations:** the AV official-test metric covers only clips with available video features, so its subset differs from the full 3,289-clip ASR table. Incomplete visual coverage may introduce selection bias. These legacy pilot scores are not benchmarks for `outputs/av_transcription_full.pt`.

## 4. Metrics implemented but without committed numeric summaries

Evaluation code includes:

- Character error rate (CER), at corpus and utterance levels.
- Word error rate (WER) in the Whisper base-versus-LoRA comparison code.
- Exact matches, empty predictions, per-speaker CER, and utterance-level AV improvement/worsening/ties.
- Greedy versus beam-search decoding comparisons on a small validation subset.

A numeric WER summary and a committed beam-search summary are not present in the result artifacts reviewed for this report. A saved benchmark for the active full AV checkpoint is also missing. Rerun those evaluations with the matching datasets, manifests, and checkpoints before reporting values.

## 5. Translation and generated speech

The app uses NLLB-200 distilled 600M for Chinese-to-English translation and Edge TTS (`en-US-AriaNeural`) to synthesize English speech. The repository does not currently contain validated BLEU, chrF, COMET, human translation scores, a MOS/listener study, objective generated-speech quality scores, or a completed end-to-end benchmark for the app's transcription → translation → speech path. ASR CER does not measure translation or TTS quality.

## 6. Recompute the official-test ASR table

From the repository root, with the three prediction CSVs present:

```bash
python scripts/evaluation/score_official_predictions.py
```

The scorer uses the normalization specified in section 2 and checks that all three files contain the same ordered clips and reference text. It prints CER, exact-match rate, empty-output count, and per-speaker CER summary.

The legacy `score_test.py` optionally uses OpenCC to convert Traditional Chinese to Simplified Chinese before scoring. Its results may differ from this NFKC/alphanumeric method; do not mix scores from the two normalizers in one table.
