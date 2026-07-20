"""stretch_to_fit port + the Auto/Natural/Manual fit dispatch.

Extends the CLI in two ways the spec calls for:
  * Auto works in BOTH directions — compress when generated duration exceeds
    slot × tolerance (the CLI behavior) AND extend (slow down) when it falls
    under slot ÷ tolerance; the same tolerance governs both.
  * atempo chaining handles factors below ffmpeg's 0.5-per-pass floor as well
    as above the 2.0 ceiling.

Modes:
  auto    — slot comparison as above; within tolerance = untouched copy
  natural — no time adjustment ever; the clip plays at generated length
            (overlap into later windows is Phase 4 assembly's concern —
            mixed, never cut)
  manual  — a direct factor (>1 speeds up, <1 slows down, 1.0 untouched),
            applied instead of any slot comparison

Phase 4 adds the per-line effective-placement-window export that assembly.py
consumes.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .. import config


class TimefitError(RuntimeError):
    pass


MIN_ATEMPO = 0.5   # ffmpeg's per-pass floor (config.MAX_ATEMPO is the ceiling)


def probe_duration(path: Path) -> float:
    """Duration of an audio file in seconds (ffprobe — handles float32 wavs
    from torchaudio, which the stdlib wave module rejects)."""
    try:
        cp = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True)
        return float(cp.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise TimefitError(f"could not probe duration of {path.name}: {e}") from None


def effective_window(start_s: float, fit_path: Path) -> tuple[int, int]:
    """A dubbed line's effective placement window in ms: [start, start +
    fit duration). For auto/manual fits that's ~the subtitle slot; for
    Natural mode it's the generated length — possibly bleeding past the slot.
    assembly.py splices exactly these windows (merged when overlapping)."""
    start_ms = int(round(start_s * 1000))
    return start_ms, start_ms + int(round(probe_duration(fit_path) * 1000))


def atempo_filters(factor: float) -> list[str]:
    """Chain atempo passes so any positive factor stays within ffmpeg's
    per-filter [0.5, 2.0] range."""
    if factor <= 0:
        raise TimefitError(f"invalid stretch factor: {factor}")
    filters: list[str] = []
    remaining = factor
    while remaining > config.MAX_ATEMPO:
        filters.append(f"atempo={config.MAX_ATEMPO}")
        remaining /= config.MAX_ATEMPO
    while remaining < MIN_ATEMPO:
        filters.append(f"atempo={MIN_ATEMPO}")
        remaining /= MIN_ATEMPO
    filters.append(f"atempo={remaining:.5f}")
    return filters


def stretch(clip: Path, factor: float, out: Path) -> None:
    """Pitch-preserving tempo change: >1 speeds up, <1 slows down."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(clip),
             "-filter:a", ",".join(atempo_filters(factor)), str(out)],
            check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode(errors="replace").strip().splitlines()[-2:]
        raise TimefitError("ffmpeg atempo failed: " + " | ".join(tail)) from None


def _copy(raw: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw, out)


def fit_line(raw: Path, out: Path, slot_s: float, fields: dict) -> dict:
    """Produce the fit take from the (untouched) raw take, per the line's
    resolved fit mode. `fields` is the resolve() output for this line.
    Returns {mode, factor, raw_duration_s, fit_duration_s, hard_stretch}."""
    mode = fields["fit_mode"]["value"]
    raw_dur = probe_duration(raw)
    factor = 1.0

    if mode == "natural":
        pass                                   # play at generated length
    elif mode == "manual":
        factor = float(fields["manual_factor"]["value"])
    elif mode == "auto":
        tol = float(fields["tolerance"]["value"])
        if tol <= 0:
            raise TimefitError(f"invalid tolerance: {tol}")
        if slot_s > 0:
            if raw_dur > slot_s * tol:         # too long -> compress
                factor = raw_dur / slot_s
            elif raw_dur < slot_s / tol:       # too short -> extend
                factor = raw_dur / slot_s
    else:
        raise TimefitError(f"unknown fit mode: {mode!r}")

    if abs(factor - 1.0) < 1e-4:
        _copy(raw, out)
    else:
        stretch(raw, factor, out)

    return {
        "mode": mode,
        "factor": round(factor, 4),
        "raw_duration_s": round(raw_dur, 3),
        "fit_duration_s": round(probe_duration(out), 3),
        # beyond one full atempo pass: audible processing likely (CLI warned
        # at MAX_ATEMPO; manual-mode UI guidance range is 0.7–1.5)
        "hard_stretch": factor > config.MAX_ATEMPO or factor < MIN_ATEMPO,
    }
