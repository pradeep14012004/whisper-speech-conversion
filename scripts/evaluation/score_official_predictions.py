"""Recompute the documented official-test ASR comparison metrics.

Run from the repository root:
    python scripts/evaluation/score_official_predictions.py

Normalization is NFKC + lowercase + Unicode letters/numbers only. The metric is
corpus CER (total edit distance / total normalized reference characters), not
the mean of per-clip CERs. No third-party scoring package is required.
"""
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "Whisper-small base": ROOT / "outputs/evaluation/official_test_small_predictions.csv",
    "Whisper-small + LoRA": ROOT / "outputs/evaluation/official_test_lora_predictions.csv",
    "Whisper-large-v3": ROOT / "outputs/evaluation/official_test_large_v3_predictions.csv",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", "" if text is None else str(text)).lower()
    return "".join(ch for ch in text if ch.isalnum())


def edit_distance(ref, hyp):
    """Levenshtein distance using O(min(len(ref), len(hyp))) memory."""
    a, b = list(ref), list(hyp)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def load_rows(path):
    if not path.is_file():
        raise FileNotFoundError(
            "Missing prediction CSV: {}\nRun the corresponding evaluation first.".format(path)
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"whisper_audio", "reference", "prediction"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("{} must contain columns {}".format(path, sorted(required)))
        rows = []
        for row in reader:
            rows.append({
                "path": row["whisper_audio"],
                "reference": str(row["reference"] or ""),
                "prediction": str(row["prediction"] or ""),
            })
    return rows


def speaker_for(path):
    parts = Path(path).parts
    for index, part in enumerate(parts[:-1]):
        if part == "whisper":
            return parts[index + 1] if index + 1 < len(parts) - 1 else "unknown"
    return "unknown"


def score(rows):
    edits = 0
    reference_chars = 0
    exact = 0
    empty = 0
    speakers = defaultdict(lambda: {"edits": 0, "reference_chars": 0, "clips": 0})

    for row in rows:
        ref = normalize(row["reference"])
        hyp = normalize(row["prediction"])
        error_count = edit_distance(ref, hyp)
        edits += error_count
        reference_chars += len(ref)
        exact += int(ref == hyp)
        empty += int(not hyp)

        speaker = speaker_for(row["path"])
        speakers[speaker]["edits"] += error_count
        speakers[speaker]["reference_chars"] += len(ref)
        speakers[speaker]["clips"] += 1

    cer = edits / max(reference_chars, 1)
    speaker_cers = sorted(
        data["edits"] / max(data["reference_chars"], 1)
        for data in speakers.values()
    )
    median = None
    if speaker_cers:
        middle = len(speaker_cers) // 2
        median = speaker_cers[middle] if len(speaker_cers) % 2 else (
            speaker_cers[middle - 1] + speaker_cers[middle]
        ) / 2
    return {
        "clips": len(rows),
        "reference_chars": reference_chars,
        "edits": edits,
        "cer": cer,
        "exact": exact,
        "empty": empty,
        "speakers": len(speakers),
        "speaker_min": speaker_cers[0] if speaker_cers else None,
        "speaker_median": median,
        "speaker_max": speaker_cers[-1] if speaker_cers else None,
    }


def main():
    loaded = {name: load_rows(path) for name, path in FILES.items()}

    base_name = next(iter(loaded))
    base_rows = loaded[base_name]
    baseline_keys = [
        (row["path"], normalize(row["reference"])) for row in base_rows
    ]
    for name, rows in loaded.items():
        keys = [(row["path"], normalize(row["reference"])) for row in rows]
        if keys != baseline_keys:
            raise ValueError(
                "Clip order or normalized reference text differs for '{}'. "
                "Do not compare these files as the same test set.".format(name)
            )

    print("Official-test ASR scoring")
    print("Normalizer: Unicode NFKC, lowercase, letters/numbers only")
    print("CER: corpus edit count / normalized reference characters\n")
    print("{:<25s} {:>7s} {:>10s} {:>13s} {:>14s} {:>8s}".format(
        "Model", "Clips", "CER", "Exact matches", "Empty outputs", "Speakers"
    ))
    print("-" * 82)

    for name, rows in loaded.items():
        result = score(rows)
        print("{:<25s} {:>7d} {:>9.2f}% {:>6d} ({:5.2f}%) {:>14d} {:>8d}".format(
            name,
            result["clips"],
            result["cer"] * 100,
            result["exact"],
            100 * result["exact"] / max(result["clips"], 1),
            result["empty"],
            result["speakers"],
        ))
        print("  Reference characters: {}; edit count: {}".format(
            result["reference_chars"], result["edits"]
        ))
        print("  Per-speaker CER min / median / max: {}".format(
            " / ".join(
                "{:.2f}%".format(result[key] * 100)
                for key in ("speaker_min", "speaker_median", "speaker_max")
            )
        ))
    print("\nReminder: large-v3 is an offline comparison model; app.py uses Whisper-small + LoRA + AV adapter.")
    print("These CER scores evaluate Chinese transcription only, not translation or TTS quality.")


if __name__ == "__main__":
    main()
