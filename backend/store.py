"""Project storage registry: one directory per project under STORAGE_ROOT.

    storage/<key>/
        project.json      runtime source of truth
        source.mp4        uploaded video
        source.srt        uploaded subtitles
        segments.csv      interchange snapshot (export regenerates from state)
        thumb.jpg         Project Center thumbnail (ffmpeg frame grab)
        separated/        Demucs output            (Phase 2)
        speaker_clips/    extracted reference clips (Phase 2)
        refs/             concatenated references   (Phase 2/3)
        dub_work/         raw/ fit/ undo/           (Phase 3)

Single user, but FastAPI sync endpoints run on a threadpool — a per-project
lock serializes read-modify-write cycles on project.json.
"""
from __future__ import annotations

import re
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

from . import config
from .models.project_state import ProjectState, StateError

_KEY_RE = re.compile(r"[^A-Za-z0-9\-_]")
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

VIDEO_NAME = "source.mp4"
SRT_NAME = "source.srt"
THUMB_NAME = "thumb.jpg"

# Directories derived from line/speaker state — wiped on CSV/SRT reset.
# separated/ is per-VIDEO (Demucs on the full file), so it survives resets.
DERIVED_DIRS = ("speaker_clips", "refs", "dub_work", "test_speech")


def project_key(name: str) -> str:
    key = _KEY_RE.sub("_", name.strip())
    if not key:
        raise StateError("project name must contain at least one usable character")
    return key


def project_dir(key: str) -> Path:
    # key is always produced by project_key(); belt-and-braces against traversal:
    if _KEY_RE.search(key) or key in ("", ".", ".."):
        raise StateError(f"bad project key: {key!r}")
    return config.STORAGE_ROOT / key


def json_path(key: str) -> Path:
    return project_dir(key) / config.PROJECT_JSON


def exists(key: str) -> bool:
    return json_path(key).exists()


def list_keys() -> list[str]:
    if not config.STORAGE_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in config.STORAGE_ROOT.iterdir()
        if (p / config.PROJECT_JSON).exists()
    )


@contextmanager
def locked(key: str):
    """Serialize read-modify-write on one project's state."""
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        yield


def load(key: str) -> ProjectState:
    if not exists(key):
        raise StateError(f"no such project: {key!r}")
    return ProjectState.load(json_path(key))


def create(name: str) -> ProjectState:
    key = project_key(name)
    if exists(key):
        raise StateError(f"project already exists: {key!r}")
    ps = ProjectState.new(name, path=json_path(key))
    ps.data["key"] = key
    ps.data["video_duration_s"] = None
    ps.save()
    return ps


def delete(key: str) -> None:
    d = project_dir(key)
    if not (d / config.PROJECT_JSON).exists():
        raise StateError(f"no such project: {key!r}")
    shutil.rmtree(d)
    with _locks_guard:
        _locks.pop(key, None)


def wipe_derived(key: str) -> None:
    """Remove state-derived audio dirs after a line-table reset
    (SRT/CSV upload or CSV editor save). Keeps separated/ — Demucs output
    belongs to the video, not the line table."""
    d = project_dir(key)
    for sub in DERIVED_DIRS:
        shutil.rmtree(d / sub, ignore_errors=True)


def stats(ps: ProjectState) -> dict:
    """Project Center card numbers."""
    lines = ps.data["lines"]
    dubbable = [l for l in lines if l["speaker"] and l["text"]]
    dubbed = [l for l in dubbable
              if l["badge"] in ("clean", "needs_light", "needs_full")]
    return {
        "duration_s": ps.data.get("video_duration_s"),
        "n_speakers": len(ps.data["speakers"]),
        "n_lines": len(lines),
        "n_dubbable": len(dubbable),
        "n_dubbed": len(dubbed),
    }
