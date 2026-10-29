from pathlib import Path
import csv
import random
import tarfile
import shutil

DATA = Path.home() / "Downloads" / "aishell6_whisper_training"
OUT = Path("data/aishell6_lora_pilot").resolve()
TRAIN_COUNT = 500
VALID_COUNT = 100
SEED = 42

def load_transcripts(path):
    result = {}
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result

def load_pairs(split, mapping, transcripts):
    rows = []
    for line in mapping.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        whispered, normal = parts
        if whispered.split("/", 1)[0] != split:
            continue

        key = Path(normal).stem
        if key not in transcripts:
            continue

        rows.append({
            "split": split,
            "source_audio": whispered,
            "normal_audio": normal,
            "transcript": transcripts[key],
            "speaker": Path(whispered).parent.name,
        })
    return rows

def extract_selected(split, selected, archive_path):
    expected = {r["source_audio"]: r for r in selected}
    found = set()
    print(f"\nScanning {archive_path.name} for {len(expected)} selected audio files...")

    # Streaming mode avoids loading the entire archive index into memory.
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            row = expected.get(member.name)
            if row is None or not member.isfile():
                continue

            dest = OUT / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)

            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read archive member: {member.name}")

            with source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)

            row["audio_path"] = str(dest)
            found.add(member.name)

            if len(found) % 100 == 0:
                print(f"Extracted {len(found)}/{len(expected)}", flush=True)

            if len(found) == len(expected):
                break

    missing = set(expected) - found
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected audio files were not found. "
            f"Example: {next(iter(missing))}"
        )
    print(f"Extracted {len(found)} files successfully.")

def main():
    for name in ("train.tar.gz", "valid.tar.gz", "w2n.txt", "text_sentence"):
        if not (DATA / name).is_file():
            raise FileNotFoundError(DATA / name)

    transcripts = load_transcripts(DATA / "text_sentence")
    mapping = DATA / "w2n.txt"

    train_rows = load_pairs("train", mapping, transcripts)
    valid_rows = load_pairs("valid", mapping, transcripts)
    print("Usable training pairs:", len(train_rows))
    print("Usable validation pairs:", len(valid_rows))

    if len(train_rows) < TRAIN_COUNT or len(valid_rows) < VALID_COUNT:
        raise RuntimeError("Not enough usable pairs for the requested subset.")

    rng = random.Random(SEED)
    train_selected = rng.sample(train_rows, TRAIN_COUNT)
    valid_selected = rng.sample(valid_rows, VALID_COUNT)

    # Verify the sample has unique source files and non-empty transcripts.
    selected = train_selected + valid_selected
    assert len({r["source_audio"] for r in selected}) == len(selected)
    assert all(r["transcript"].strip() for r in selected)

    OUT.mkdir(parents=True, exist_ok=True)
    extract_selected("train", train_selected, DATA / "train.tar.gz")
    extract_selected("valid", valid_selected, DATA / "valid.tar.gz")

    manifest = OUT / "manifest.csv"
    columns = [
        "split", "audio_path", "transcript",
        "source_audio", "normal_audio", "speaker",
    ]
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in train_selected + valid_selected:
            writer.writerow(row)

    print("\nPILOT SUBSET READY")
    print("Training examples:", len(train_selected))
    print("Validation examples:", len(valid_selected))
    print("Manifest:", manifest)
    print("Audio files stored in:", OUT)
    print("Existing model checkpoints were not modified.")

if __name__ == "__main__":
    main()
