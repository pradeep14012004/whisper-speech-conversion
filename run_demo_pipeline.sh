#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p outputs

if [[ $# -lt 1 ]]; then
    echo 'Usage: ./run_demo_pipeline.sh "Chinese transcript" [--english]'
    exit 1
fi

TEXT="$1"
ENGLISH="${2:-}"

echo "Step 1/2: Generating Chinese speech..."
./run_chinese_tts.sh "$TEXT"

if [[ "$ENGLISH" == "--english" ]]; then
    echo
    echo "Step 2/2: Translating transcript and generating English speech..."

    python - "$TEXT" <<'PY'
import sys
import pandas as pd

text = sys.argv[1]
pd.DataFrame([{"transcript": text}]).to_csv(
    "outputs/demo_transcript.csv", index=False
)
print("Saved Chinese transcript: outputs/demo_transcript.csv")
PY

    python - <<'PY'
import pandas as pd

df = pd.read_csv("outputs/demo_transcript.csv")
print("Chinese transcript:", df.loc[0, "transcript"])
print()
print("Note: The existing translate_to_english.py processes the")
print("whole training manifest, not this single demo transcript.")
print("Use a single-text translation script for the demo.")
PY

    echo "English stage is not yet connected to a single-transcript pipeline."
    exit 2
fi

echo
echo "Chinese TTS completed."
echo "Audio: outputs/chinese_tts.mp3"
