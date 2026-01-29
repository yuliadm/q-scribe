# Q-Scribe 
## A Two-Person, Zero-Subscription Transcription “Service” Built in an Afternoon


<img width="1024" height="512" alt="image" src="https://github.com/yuliadm/q-scribe/blob/main/assets/q-scribe-main.png" />



### Overview
This is a single-file app: `app.py`, which combines the workflow (below) + the minimal Streamlit UI.
                                                                                                   
```bash
Upload video   ➡️   save to temp folder   ➡️   extract audio   ➡️   split into short WAV chunks  
                                                                             ⬇️
                                                          run Whisper on each chunk with progress UI 
                                                                             ⬇️
                                                                    clean up the text
                                                                             ⬇️
                 delete temp files (+ clear GPU cache)   ⬅️   show preview + allow user to download  
```      

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

### The practical “remote team” trick

How to run Streamlit on your laptop (the “server”) and let your colleague open it in their browser over Tailscale, without putting anything on the public internet?

(1) Install and log in to Tailscale (Host setup)

 Install Tailscale on your laptop, then authenticate and verify your laptop has a Tailscale IPv4:
```bash
sudo tailscale up
tailscale ip -4
```
Example output (your laptop’s tailnet IP): `100.32.105.55`

(2) Run your Streamlit app

Run Streamlit listening on all interfaces (this is required for device sharing NAT/IP aliasing to work reliably):
```
streamlit run app.py --server.address 0.0.0.0 --server.port 8080
```
Notes:

- You can use port 8501 too, but 8080 is often more “friendly”.
- Keep this terminal running. 

(3) Invite your colleague and share your device

In the Tailscale admin UI:

- Go to Users →Invite user → enter colleague’s email
- They must accept and log in
- Go to Machines/Devices →Select your laptop →Click Share… (or “Device sharing”)
- Enter colleague’s email and send, your colleague must accept the share.

(4) Install and log in to Tailscale (Colleague setup)

They install Tailscale and log in with the invited email. In your colleague’s Tailscale UI, your laptop appears as a shared device and it will typically show a different 100.x IP in their tailnet than the one you see, e. g. 100.89.107.55.

(5) Colleague opens the Streamlit URL in their browser

They must use the IP shown on their side and the port you chose: http://100.89.107.55:8080 


**LinkedIn article**: https://www.linkedin.com/pulse/q-scribe-two-person-zero-subscription-transcription-service-kulagina-4k0jc/?trackingId=s60fqzJwbEGVt9C2q2GPhA%3D%3D

### UI screenshot

<img width="1024" height="1696" alt="image" src="https://github.com/yuliadm/q-scribe/blob/main/assets/q-scribe-screenshot.png" />


