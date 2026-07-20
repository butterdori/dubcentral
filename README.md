# Dubbing Webapp

Single-user, self-hosted web UI over the Chatterbox dubbing pipeline
(spec: `dubbing_webapp_spec.md`; CLI reference: `dub_pipeline.py`).

## Run

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8765
python -m pytest backend/tests -q
```

Requires **ffmpeg/ffprobe on PATH**
