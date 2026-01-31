####################################################
#
# RUN (bash): streamlit run app.py --server.address 100.xxx.xx.xx (tailscale address) --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
#
####################################################


import os
import re
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from pydub import AudioSegment
import torch
from transformers import pipeline


# ---------------------------
# Text cleanup (your function)
# ---------------------------
def clean_ru_text(t: str) -> str:
    t = re.sub(r"\[(?:музыка|аплодисменты|смех|шум|тишина|music|noise|laughter).*?\]", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\b(?:\w+\s+){3,})\1+", r"\1", t, flags=re.I)
    return t


# -----------------------------------
# Audio extraction + chunking (pydub)
# -----------------------------------
def extract_audio_from_video(video_path: str, output_folder: str, chunk_length_ms: int = 30000):
    audio = AudioSegment.from_file(video_path)  # ffmpeg auto-detects format
    os.makedirs(output_folder, exist_ok=True)

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk.export(os.path.join(output_folder, f"chunk_{i//chunk_length_ms}.wav"), format="wav")


# ---------------------------
# Load Whisper once (cached)
# ---------------------------
@st.cache_resource
def get_asr_pipeline(model_name: str, use_gpu: bool):
    if use_gpu and torch.cuda.is_available():
        device = 0
        dtype = torch.float16
    else:
        device = -1
        dtype = torch.float32

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )
    return asr


# ---------------------------
# Transcription pipeline
# ---------------------------
def transcribe_chunks(asr, chunks_dir: Path, language: str = "ru", num_beams: int = 5):
    files = sorted([p for p in chunks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
    full_text = []

    progress = st.progress(0)
    status = st.empty()

    total = len(files)
    for idx, wav_path in enumerate(files, start=1):
        status.write(f"Transcribing chunk {idx}/{total}: {wav_path.name}")

        result = asr(
            str(wav_path),
            generate_kwargs={
                "task": "transcribe",
                "language": language,
                "temperature": 0.0,
                "num_beams": num_beams,
            },
        )
        full_text.append(clean_ru_text(result["text"]))
        progress.progress(int(idx / total * 100))

    status.write("Done.")
    return "\n".join(full_text)


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Minimalistic Transcription Tool", page_icon="📝", layout="centered")

st.title("📝 Video → Text (Whisper)")
st.caption("Upload a video, the server transcribes it, download the transcript as .txt")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Whisper model",
        options=["openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small"],
        index=0
    )
    use_gpu = st.checkbox("Use GPU (CUDA)", value=torch.cuda.is_available())
    language = st.selectbox("Language", options=["ru", "en", "de", "fr", "es"], index=0)
    chunk_seconds = st.slider("Chunk length (seconds)", min_value=10, max_value=30, value=30, step=5)
    num_beams = st.slider("Beam search (quality vs speed)", min_value=1, max_value=10, value=5)

    st.divider()
    if st.button("Clean CUDA cache"):
        cleanup_cuda()
        st.success("CUDA cache cleared.")

uploaded = st.file_uploader("Upload video (.mp4, .mov, .mkv)", type=["mp4", "mov", "mkv", "webm"])

if uploaded is not None:
    st.info(f"File received: {uploaded.name} ({uploaded.size / (1024**2):.1f} MB)")

    if st.button("🚀 Transcribe"):
        job_dir = tempfile.mkdtemp(prefix="transcribe_job_")
        try:
            # Save upload to disk
            video_path = os.path.join(job_dir, uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            chunks_dir = Path(job_dir) / "audio_chunks"
            st.write("Extracting audio and splitting into chunks...")
            extract_audio_from_video(
                video_path=video_path,
                output_folder=str(chunks_dir),
                chunk_length_ms=int(chunk_seconds * 1000),
            )

            st.write("Loading Whisper pipeline...")
            asr = get_asr_pipeline(model_name=model_name, use_gpu=use_gpu)

            st.write("Transcribing...")
            transcript = transcribe_chunks(asr, chunks_dir, language=language, num_beams=num_beams)

            # Save transcript to a file in memory + offer download
            out_name = Path(uploaded.name).stem + "_transcript.txt"
            transcript_bytes = transcript.encode("utf-8")

            st.success("Transcription completed ✅")
            st.text_area("Preview", transcript, height=300)

            st.download_button(
                label="⬇️ Download transcript (.txt)",
                data=transcript_bytes,
                file_name=out_name,
                mime="text/plain",
            )

            # Optional GPU cleanup after job
            cleanup_cuda()

        finally:
            # Clean up job folder
            shutil.rmtree(job_dir, ignore_errors=True)
