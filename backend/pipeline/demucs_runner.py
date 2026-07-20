"""Port of the CLI's demucs_paths / run_demucs against the webapp's fixed
per-project layout (video is always source.mp4, so the demucs output lands at
separated/<model>/source/{vocals,no_vocals}.wav).

Demucs runs as a SUBPROCESS (its own process acquires and releases VRAM), but
still inside the gpu.exclusive() slot with Chatterbox unloaded first — the
model-swap discipline the CLI did by hand.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from .. import config
from ..gpu_service import gpu


class DemucsError(RuntimeError):
    pass


def demucs_paths(project_dir: Path) -> tuple[Path, Path, Path]:
    sep_root = project_dir / "separated"
    stem_dir = sep_root / config.DEMUCS_MODEL / "source"   # source.mp4 stem
    return sep_root, stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"


def _run_cli(video: Path, sep_root: Path) -> None:
    """The actual demucs invocation — split out so tests can monkeypatch it
    with a fake separator. Always uses GPU when available: the app's Force
    CPU checkbox is scoped to TTS/dubbing only (see routers/dub.py)."""
    try:
        subprocess.run(
            ["demucs", "--two-stems=vocals", "-n", config.DEMUCS_MODEL,
             "-o", str(sep_root), str(video)],
            check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise DemucsError(
            "demucs not found on PATH — pip install demucs in the app venv"
        ) from None
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-4:]
        raise DemucsError("demucs failed: " + (" | ".join(tail) or str(e))) from None


def separate(project_dir: Path,
             on_message: Callable[[str], None] | None = None
             ) -> tuple[Path, Path]:
    """Run Demucs once on the full video (skipped if output already present —
    same caching as the CLI). Returns (vocals, no_vocals). Always uses GPU
    when available — not affected by the project's Force CPU checkbox,
    which is scoped to TTS/dubbing only (see routers/dub.py)."""
    video = project_dir / "source.mp4"
    if not video.exists():
        raise DemucsError("no video uploaded — upload one in the Upload Center")
    sep_root, vocals, no_vocals = demucs_paths(project_dir)
    if vocals.exists() and no_vocals.exists():
        return vocals, no_vocals

    if on_message:
        dev = gpu.device()
        note = "" if dev == "cuda" else " — this will be slow on CPU"
        on_message(f"separating vocals with Demucs ({dev}){note}…")

    gpu.unload_tts()                           # never co-resident with ANY TTS engine
    with gpu.exclusive("demucs separation"):
        _run_cli(video, sep_root)

    if not (vocals.exists() and no_vocals.exists()):
        raise DemucsError(
            f"demucs finished but expected outputs not found under {sep_root} — "
            "check the demucs model name in config")
    return vocals, no_vocals
