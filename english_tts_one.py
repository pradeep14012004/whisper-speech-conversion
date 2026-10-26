import asyncio
from pathlib import Path
import sys
import edge_tts

if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:]).strip()
else:
    text_file = Path("outputs/demo_english.txt")
    if not text_file.exists():
        raise SystemExit(
            "No English text found. Run translate_one.py first."
        )
    text = text_file.read_text(encoding="utf-8").strip()

if not text:
    raise SystemExit("Error: empty English text")

output = Path("outputs/demo_english.mp3")
output.parent.mkdir(parents=True, exist_ok=True)

async def main():
    voice = edge_tts.Communicate(
        text,
        voice="en-US-AriaNeural"
    )
    await voice.save(str(output))

asyncio.run(main())

if not output.exists() or output.stat().st_size == 0:
    raise SystemExit("TTS failed: output audio is missing or empty")

print("English audio saved:", output.resolve())
print("File size:", output.stat().st_size, "bytes")
