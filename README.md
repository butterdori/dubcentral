# Dubbing Webapp

Self-hosted, single-user web app for dubbing video using a pre-translated, timestamped SRT file and the original video file, with manual speaker diarization. Pipeline: SRT → speaker assignment → Demucs vocal/background separation → per-speaker reference clip extraction (with manual pruning) → TTS generation per line → time-fitting → final splice and mux.

![Screenshot](images/screenshot.png)

## Run

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8765
python -m pytest backend/tests -q
```

Requires **ffmpeg/ffprobe on PATH**
