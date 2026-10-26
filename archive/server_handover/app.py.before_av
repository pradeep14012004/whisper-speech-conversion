from pathlib import Path
import subprocess
import tempfile

import pandas as pd
import streamlit as st
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs/verified_av_manifest.csv"

st.set_page_config(page_title="Chinese Speech Demo", layout="wide")
st.title("Dataset-Based Chinese Speech Demo")
st.caption("Original dataset video + whispered audio + generated Chinese and English speech")

if not MANIFEST.exists():
    st.error(f"Manifest not found: {MANIFEST}")
    st.stop()

df = pd.read_csv(MANIFEST).fillna("")
df = df[
    (df["whisper_audio_exists"].astype(str).str.lower() == "true")
    & (df["feature_exists"].astype(str).str.lower() == "true")
].reset_index(drop=True)

def resolve_path(value):
    p = Path(str(value))
    if p.is_absolute():
        return p
    return ROOT / p

if df.empty:
    st.error("No usable dataset rows found.")
    st.stop()

labels = [
    f"{r['speaker_id']} | {r['whisper_id']} | {str(r['transcript'])[:35]}"
    for _, r in df.iterrows()
]
selected = st.selectbox("Choose a dataset sample", range(len(df)),
                        format_func=lambda i: labels[i])
row = df.iloc[selected]

video_path = resolve_path(row["whisper_video"])
audio_path = resolve_path(row["whisper_audio"])
transcript = str(row["transcript"]).strip()

left, right = st.columns(2)

with left:
    st.subheader("Original dataset video")
    st.caption(str(row["whisper_id"]))
    if video_path.exists():
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            with tempfile.TemporaryDirectory() as tmp:
                mp4_path = Path(tmp) / "dataset_video.mp4"
                subprocess.run(
                    [ffmpeg, "-y", "-i", str(video_path),
                     "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", str(mp4_path)],
                    check=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                st.video(mp4_path.read_bytes(), format="video/mp4")
        except Exception as e:
            st.warning("Could not convert this AVI for browser playback.")
            st.code(str(e)[:1000])
    else:
        st.error(f"Video not found: {video_path}")

with right:
    st.subheader("Matching whispered audio")
    if audio_path.exists():
        st.audio(audio_path.read_bytes(), format="audio/wav")
    else:
        st.error(f"Audio not found: {audio_path}")

st.subheader("Chinese transcript")
edited_text = st.text_area("Edit transcript if needed", value=transcript, height=100)

st.caption(f"Speaker: {row['speaker_id']} | Split: {row['split']}")
st.caption("Video features: " + str(resolve_path(row["feature_path"])))

if st.button("Generate Chinese and English speech", type="primary"):
    text = edited_text.strip()
    if not text:
        st.error("Please enter a Chinese transcript.")
        st.stop()

    with st.spinner("Generating Chinese speech..."):
        result = subprocess.run(
            ["bash", str(ROOT / "run_chinese_tts.sh"), text],
            cwd=ROOT, capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("Chinese speech generation failed.")
            st.code(result.stderr[-3000:])
            st.stop()

    with st.spinner("Translating into English..."):
        result = subprocess.run(
            ["python", str(ROOT / "translate_one.py"), text],
            cwd=ROOT, capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("Translation failed.")
            st.code(result.stderr[-3000:])
            st.stop()

    with st.spinner("Generating English speech..."):
        result = subprocess.run(
            ["python", str(ROOT / "english_tts_one.py")],
            cwd=ROOT, capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("English speech generation failed.")
            st.code(result.stderr[-3000:])
            st.stop()

    st.session_state["demo_text"] = text
    st.session_state["demo_sample"] = str(row["whisper_id"])
    st.success("Both speech files were generated.")

if st.session_state.get("demo_sample") == str(row["whisper_id"]):
    st.subheader("Generated speech")
    st.markdown("**Chinese audio**")
    zh = ROOT / "outputs/chinese_tts.mp3"
    if zh.exists():
        st.audio(zh.read_bytes(), format="audio/mpeg")

    en_text = ROOT / "outputs/demo_english.txt"
    en_audio = ROOT / "outputs/demo_english.mp3"
    if en_text.exists():
        st.markdown("**English translation**")
        st.write(en_text.read_text(encoding="utf-8"))
    if en_audio.exists():
        st.markdown("**English audio**")
        st.audio(en_audio.read_bytes(), format="audio/mpeg")

    if zh.exists():
        st.download_button("Download Chinese audio", zh.read_bytes(),
                           file_name="chinese_tts.mp3", mime="audio/mpeg")
    if en_audio.exists():
        st.download_button("Download English audio", en_audio.read_bytes(),
                           file_name="english_tts.mp3", mime="audio/mpeg")
