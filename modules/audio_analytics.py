import numpy as np
import librosa
import plotly.graph_objects as go
import streamlit as st


def render_audio_analytics(audio_path):
    if not audio_path:
        st.info("Audio analytics will appear here after processing an input file.")
        return
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(np.square(y))))
    peak_amp = float(np.max(np.abs(y))) if len(y) else 0.0
    silence_mask = np.abs(y) < 0.01
    silent_ratio = float(np.mean(silence_mask)) * 100.0

    waveform_fig = go.Figure()
    waveform_fig.add_scatter(x=np.linspace(0.0, duration, len(y)), y=y, mode="lines", line=dict(color="#5ec7ff", width=1.0))
    waveform_fig.update_layout(title="Audio waveform", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(waveform_fig, use_container_width=True)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    spectrogram_fig = go.Figure(data=go.Heatmap(z=mel_db, x=np.linspace(0.0, duration, mel_db.shape[1]), y=np.linspace(0, sr / 2, mel_db.shape[0]), colorscale="Plasma"))
    spectrogram_fig.update_layout(title="Mel spectrogram", template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(spectrogram_fig, use_container_width=True)

    metrics = {
        "Audio duration (s)": round(duration, 3),
        "Sample rate (Hz)": sr,
        "Samples": len(y),
        "Peak amplitude": round(peak_amp, 5),
        "RMS energy": round(rms, 5),
        "Silence detection": f"{silent_ratio:.2f}% below 0.01 amplitude",
    }
    st.json(metrics)
