"""Final Remix — the revised, simpler assembly design:

  The ORIGINAL MIXED AUDIO is the base everywhere. Only successfully dubbed
  windows are touched: within each window the base is spliced with Demucs'
  no_vocals (original speech removed, music/effects kept) and the dubbed take
  is overlaid on top. Undubbed and failed lines keep their original audio
  untouched — vocals.wav is never used here (it only feeds reference-clip
  extraction).

  A line's EFFECTIVE PLACEMENT WINDOW is [start, start + fit_duration]
  (timefit.effective_window): for auto/manual fits that is ~the subtitle
  slot; for Natural mode it is the generated length, which may bleed past
  the slot into the next window or into undubbed original audio. Overlaps
  are MIXED, never cut: overlapping windows are merged for the vocal-strip
  splice, and every take is overlaid at its own start — two takes overlapping
  simply sound simultaneously, and a take bleeding into undubbed audio plays
  over it.

  TIMESTAMP MUTE (project.timestamp_mute, off by default): the flip side of
  bleed — if the dub is SHORTER than the original speech, muting only for
  the dub's own duration leaves the tail of the original speech audible
  after the dub ends (subtitle timestamps are drawn from the original
  speech, so end-of-timestamp ≈ end-of-speech). When enabled, the muted span
  for each line is widened to cover at least the full subtitle timestamp
  [start, end], regardless of how long the dub actually is — the dub still
  plays for only its own length, but no original speech leaks in behind it.

  Output: <project>/final.mp4 — original video stream copied, remixed audio
  muxed in.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable

from ..models.project_state import ProjectState
from . import demucs_runner, dub_work, timefit

# badges whose current take is spliceable (stale takes are still the current
# audio; failed/never lines have no usable take)
SPLICEABLE_BADGES = {"clean", "needs_light", "needs_full"}


class AssemblyError(RuntimeError):
    pass


def final_path(project_dir: Path) -> Path:
    return project_dir / "final.mp4"


def original_audio_path(project_dir: Path) -> Path:
    # lives next to the Demucs output: both belong to the video and share its
    # lifecycle (separated/ is wiped when the video is replaced)
    return project_dir / "separated" / "original_mix.wav"


def extract_original_audio(project_dir: Path) -> Path:
    """Full mixed audio track of source.mp4, cached under separated/."""
    out = original_audio_path(project_dir)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(project_dir / "source.mp4"), "-vn",
             str(out)], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode(errors="replace").strip().splitlines()[-2:]
        raise AssemblyError("could not extract original audio: "
                            + " | ".join(tail)) from None
    return out


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent [start_ms, end_ms) spans."""
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def spliceable_lines(ps: ProjectState, project_dir: Path) -> list[dict]:
    out = []
    for l in ps.data["lines"]:
        if l["badge"] in SPLICEABLE_BADGES and \
                dub_work.fit_path(project_dir, l["line_no"]).exists():
            out.append(l)
    return out


def remix(ps: ProjectState, project_dir: Path,
          on_message: Callable[[str], None] | None = None) -> dict:
    """timestamp_mute is read from ps.data (project-level toggle, not a
    per-line field — see project_state.set_timestamp_mute)."""
    """Build final.mp4. Runs Demucs first if no_vocals is missing (cached
    otherwise — gpu discipline handled inside the runner)."""
    from pydub import AudioSegment

    t0 = time.time()
    msg = on_message or (lambda s: None)

    lines = spliceable_lines(ps, project_dir)
    if not lines:
        raise AssemblyError("no successfully dubbed lines to splice — "
                            "run a dub first")

    _, _, no_vocals_path = demucs_runner.demucs_paths(project_dir)
    if not no_vocals_path.exists():
        # same scoping as extraction: Force CPU applies to TTS only
        demucs_runner.separate(project_dir, on_message=msg)

    msg("loading original audio…")
    base = AudioSegment.from_file(extract_original_audio(project_dir))
    msg("loading instrumental (no_vocals) track…")
    nv = AudioSegment.from_file(no_vocals_path)
    nv = nv.set_frame_rate(base.frame_rate).set_channels(base.channels)

    # ---- effective placement windows ----
    timestamp_mute = bool(ps.data.get("timestamp_mute", False))
    takes = []       # (start_ms, clip AudioSegment) — dub plays for its own length
    spans = []       # vocal-strip (mute) spans — may be wider than the take
    for l in lines:
        fit = dub_work.fit_path(project_dir, l["line_no"])
        start_ms, end_ms = timefit.effective_window(l["start"], fit)
        if start_ms >= len(base):
            continue   # beyond the video — nothing to place
        clip = AudioSegment.from_file(fit)
        clip = clip.set_frame_rate(base.frame_rate).set_channels(base.channels)
        takes.append((start_ms, clip))
        mute_end_ms = end_ms
        if timestamp_mute:
            ts_end_ms = int(round(l["end"] * 1000))
            mute_end_ms = max(mute_end_ms, ts_end_ms)   # widen, never narrow
        spans.append((start_ms, min(mute_end_ms, len(base))))

    # ---- splice: strip original vocals inside every (merged) dubbed window ----
    msg(f"splicing {len(spans)} dubbed windows…")
    result = AudioSegment.empty()
    cursor = 0
    for s, e in merge_spans(spans):
        result += base[cursor:s] + nv[s:min(e, len(nv))]
        # (if no_vocals is a hair shorter than base, the tail of the window
        # falls back to original — clamp keeps concatenation consistent)
        if e > len(nv):
            result += base[len(nv):e]
        cursor = e
    result += base[cursor:]

    # ---- overlay every take at its own start — mix, never cut ----
    msg("overlaying dubbed takes…")
    for start_ms, clip in takes:
        result = result.overlay(clip, position=start_ms)

    # ---- mux ----
    msg("muxing final video…")
    tmp_audio = project_dir / "dub_work" / "final_audio.wav"
    tmp_audio.parent.mkdir(parents=True, exist_ok=True)
    result.export(tmp_audio, format="wav")
    out = final_path(project_dir)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(project_dir / "source.mp4"),
             "-i", str(tmp_audio), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", str(out)],
            check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode(errors="replace").strip().splitlines()[-3:]
        raise AssemblyError("mux failed: " + " | ".join(tail)) from None
    finally:
        tmp_audio.unlink(missing_ok=True)

    return {
        "n_lines_spliced": len(takes),
        "n_windows_merged": len(merge_spans(spans)),
        "timestamp_mute": timestamp_mute,
        "output_duration_s": round(timefit.probe_duration(out), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }
