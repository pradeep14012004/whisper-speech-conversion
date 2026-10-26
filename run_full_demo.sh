#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
    echo 'Usage: ./run_full_demo.sh "Chinese transcript"'
    exit 1
fi

TEXT="$1"

echo "===== 1/3: Chinese speech generation ====="
./run_chinese_tts.sh "$TEXT"

echo
echo "===== 2/3: Chinese-to-English translation ====="
python translate_one.py "$TEXT"

echo
echo "===== 3/3: English speech generation ====="
python english_tts_one.py

echo
echo "===== DEMO COMPLETE ====="
echo "Chinese audio: outputs/chinese_tts.mp3"
echo "English text:  outputs/demo_english.txt"
echo "English audio: outputs/demo_english.mp3"
