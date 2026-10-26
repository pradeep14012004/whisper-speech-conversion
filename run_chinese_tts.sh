#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TEXT="${1:-躺在草地上看着蓝天瞬间忘却了所有的烦恼}"
OUTPUT="outputs/chinese_tts.mp3"
VOICE="${VOICE:-zh-CN-XiaoxiaoNeural}"

mkdir -p outputs

if ! command -v edge-tts >/dev/null 2>&1; then
    echo "Error: edge-tts is not installed."
    echo "Run: python -m pip install edge-tts"
    exit 1
fi

echo "Generating Chinese speech..."
echo "Voice: $VOICE"
echo "Output: $OUTPUT"

edge-tts \
    --voice "$VOICE" \
    --rate="-5%" \
    --text "$TEXT" \
    --write-media "$OUTPUT"

test -s "$OUTPUT"

echo
echo "SUCCESS: Chinese speech generated."
echo "File: $(pwd)/$OUTPUT"
ls -lh "$OUTPUT"
