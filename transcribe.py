
import whisper
import os

audio_file = "audio.mp3"

if not os.path.exists(audio_file):
    print(f"Error: {audio_file} not found.")
    exit(1)

print("Loading Whisper large-v3...")
model = whisper.load_model("large-v3")

print("Transcribing audio...")
result = model.transcribe(audio_file, task="transcribe")

print("Detected language:", result["language"])
print("Transcript:", result["text"])

with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Saved transcript to transcript.txt")

