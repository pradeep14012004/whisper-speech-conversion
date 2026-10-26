import asyncio
import os
import pandas as pd
import edge_tts

INPUT = "outputs/english_translations.csv"
OUT_DIR = "outputs/english_speech"
VOICE = "en-US-AriaNeural"

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT)

async def generate(index, text):
    if not isinstance(text, str) or not text.strip():
        return False

    path = os.path.join(OUT_DIR, "{:05d}.mp3".format(index))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True

    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(path)
        return True
    except Exception as e:
        print("TTS failed for row {}: {}".format(index, e), flush=True)
        return False

async def main():
    success = 0
    for start in range(0, len(df), 10):
        batch = df.iloc[start:start + 10]
        results = await asyncio.gather(*[
            generate(i, str(row["english_translation"]))
            for i, (_, row) in enumerate(batch.iterrows(), start=start)
        ])
        success += sum(results)
        print("Generated {}/{}".format(
            min(start + len(batch), len(df)), len(df)
        ), flush=True)

    print("Successful audio files:", success, "/", len(df))

asyncio.run(main())
