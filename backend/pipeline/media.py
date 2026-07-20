"""ffmpeg/ffprobe helpers for Phase 1: upload validation, duration probing,
and the Project Center thumbnail frame grab. No GPU involvement."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


class MediaError(ValueError):
    """Invalid/unreadable media upload — surfaced as HTTP 400."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise MediaError(f"{cmd[0]} not found on PATH — install ffmpeg") from None
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-3:]
        raise MediaError(f"{cmd[0]} failed: {' | '.join(tail) or e}") from None


def probe_video(path: Path) -> dict:
    """Validate that `path` is a real video (has a video stream) and return
    {duration_s, has_audio}. Raises MediaError on anything unreadable."""
    cp = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    info = json.loads(cp.stdout or "{}")
    streams = info.get("streams", [])
    kinds = {s.get("codec_type") for s in streams}
    if "video" not in kinds:
        raise MediaError("file has no video stream — not a usable video upload")
    duration = None
    fmt = info.get("format", {})
    if fmt.get("duration"):
        duration = float(fmt["duration"])
    else:
        for s in streams:
            if s.get("codec_type") == "video" and s.get("duration"):
                duration = float(s["duration"])
                break
    if not duration or duration <= 0:
        raise MediaError("could not determine video duration")
    return {"duration_s": duration, "has_audio": "audio" in kinds}


def grab_thumbnail(video: Path, out: Path, duration_s: float | None) -> None:
    """Single frame for the project card. Seek a quarter in (capped at 5s) so
    we skip black lead-ins; fall back to frame 0 for very short clips."""
    seek = 0.0
    if duration_s:
        seek = min(duration_s * 0.25, 5.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-ss", f"{seek:.3f}", "-i", str(video),
        "-frames:v", "1", "-vf", "scale=480:-2", str(out),
    ])
    if not out.exists():
        raise MediaError("thumbnail grab produced no output")
