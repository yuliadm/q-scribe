# Q-Scribe 
## A Two-Person, Zero-Subscription Transcription “Service” Built in an Afternoon


<img width="612" height="396" alt="image" src="https://github.com/yuliadm/q-scribe/blob/main/assets/q-scribe.png" />



### Overview
                                                                                                     
<img width="612" height="396" alt="image" src="https://github.com/yuliadm/q-scribe/blob/main/assets/q-scribe-scheme.png" />

### A simple guide: from zero to working prototype
1. Create the environment

Set up a Python environment and install dependencies (typically listed in requirements.txt).
Key imports:
- `os`, `shutil`, `tempfile`, `Path`: file handling and temporary job folders
- `re`: text cleanup
- `streamlit`: UI
- `pydub.AudioSegment`: audio extraction + chunking (requires ffmpeg)
- `torch`: GPU detection + CUDA memory management
- `transformers.pipeline`: Whisper ASR pipeline

2. Clean up the transcript (clean_ru_text)

Whisper outputs can include things like [music], [laughter], or repeated phrases. The cleanup function:
- removes bracketed non-speech labels,
- collapses whitespace,
- removes repeated fragments (useful for some ASR hiccups).

This keeps the transcript readable and more suitable for summarization.

3. Extract audio and chunk it (extract_audio_from_video)
Additionally to loading the video/audio file and extracting the audio, the function slices the full audio into smaller parts and exports chunk_0.wav, chunk_1.wav, etc.:
- shorter inputs are more stable,
- GPU memory usage stays predictable,
- failures affect one chunk, not the whole job.

4. Load **Whisper** (https://huggingface.co/spaces/openai/whisper) once and cache it (get_asr_pipeline)
One can choose between several models depending on the task and compute resources:
- large (https://huggingface.co/openai/whisper-large-v3),
- medium (https://huggingface.co/openai/whisper-medium),
- small (https://huggingface.co/openai/whisper-small).
We cache the pipeline using Streamlit’s resource cache (for fast inference). If CUDA is available and enabled: use GPU (device=0) and float16. Otherwise: fall back to CPU (device=-1) and float32.

Once cached, future transcriptions reuse the same loaded model.

5. Transcribe each chunk (transcribe_chunks)

For each WAV chunk:
- the UI shows progress,
- Whisper runs with parameters like:
- `language` (ru/en/etc.)
- `temperature=0.0` (more deterministic output)
- `num_beams` (beam search: higher often improves quality, but slows down).

The text is cleaned, appended, and finally joined into one transcript.

## Running the app locally

Start Streamlit in terminal (from the dir where app.py lives):
```bash
streamlit run app.py
```
Additional flags (useful in some setups):

- `--server.address 100.xxx.xx.xx` for local-only access (here goes the Tailscale IP)
- `--server.port 8501` for the port
- disabling CORS/XSRF protections is convenient for quick experiments, do NOT do this when exposing the app publicly.
  

Medium post: 
https://


