<div align="center">
  <img src="frontend/favicon.svg" alt="Dubcentral Logo" width="120" />

  <h1>dubcentral</h1>

  <p><b>Self-Hosted AI Video Dubbing Console</b></p>

  <!-- Status & Tech Stack Badges -->
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/TTS-CosyVoice-B48EAD?style=flat-square" alt="CosyVoice" />
  <img src="https://img.shields.io/badge/Audio-Demucs%20%7C%20FFmpeg-A3BE8C?style=flat-square" alt="Demucs & FFmpeg" />
  <img src="https://img.shields.io/badge/Status-In%20Development-EBCB8B?style=flat-square" alt="Status" />
</div>

<br>

Self-hosted, single-user web app for dubbing video using a pre-translated, timestamped subtitle (srt) file and the original video file, with manual speaker diarization.  

Pipeline: Upload subtitle → Assign speakers (diarization) → Demucs vocal/background separation → per-speaker reference clip extraction and pruning → TTS generation per line → time-fitting → final splice and mux.

In development; works best with one process at a time.

![Screenshot](images/screenshot.png)

## Prerequisites  
ffmpeg

## Run

```bash
git clone https://github.com/butterdori/dubcentral.git && cd dubcentral

python3.10 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# Clone and install CosyVoice
mkdir -p backend/vendor
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git backend/vendor/CosyVoice
cd backend/vendor/CosyVoice && pip install -r requirements.txt
cd -

# Download CosyVoice model (~1.5 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='backend/hf_cache/Fun-CosyVoice3-0.5B')"

# Run tests (124 expected)
python -m pytest backend/tests -q

# Start server
uvicorn backend.main:app --host 0.0.0.0 --port 8765

```

