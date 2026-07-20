"""Ports of the CLI's extract_reference_clip and build_concat_reference, plus
the project-level extraction pass that replaces the CLI's 'clips' command.

Webapp semantics on top of the CLI logic:
  * Pruning is IN-BROWSER clip deletion, remembered in project.json
    (speakers[key]["pruned"] = [line_no, ...]) — so re-running extraction
    never resurrects a clip the user deliberately deleted.
  * refs/ref_<spk>.wav (the concatenated Chatterbox reference) is DERIVED
    from Speaker Center state: invalidated by clip deletion and by
    re-extraction, not by dub runs. Phase 3 rebuilds it on demand.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable

from .. import config
from ..models.project_state import ProjectState

# lazy pydub import (needs audioop; keep module importable for path helpers)


def clips_dir(project_dir: Path, spk: str) -> Path:
    return project_dir / "speaker_clips" / spk


def clip_path(project_dir: Path, spk: str, line_no: int) -> Path:
    return clips_dir(project_dir, spk) / f"line_{line_no:04d}.wav"


def ref_path(project_dir: Path, spk: str) -> Path:
    return project_dir / "refs" / f"ref_{spk}.wav"


def invalidate_ref(project_dir: Path, spk: str) -> None:
    ref_path(project_dir, spk).unlink(missing_ok=True)


# ======================= CLI port: single-clip extraction =====================

def extract_reference_clip(vocals, start_ms: int, end_ms: int,
                           silence_thresh_db: int = config.TRIM_SILENCE_DB,
                           chunk_ms: int = 10, pad_ms: int = 60,
                           min_edge_pad_ms: int = config.MIN_EDGE_PAD_MS,
                           no_trim_under_ms: int = config.NO_TRIM_UNDER_MS):
    """Extract a speaker-reference clip for one subtitle's [start_ms, end_ms)
    window from the FULL vocals track (not a pre-sliced segment), so extra
    context can be pulled from surrounding audio. Ported verbatim from the
    CLI — see dub_pipeline.py for the full rule commentary."""
    from pydub import silence as psilence

    track_len = len(vocals)
    raw_start = max(0, start_ms)
    raw_end = min(track_len, end_ms)
    duration_ms = raw_end - raw_start

    if duration_ms < no_trim_under_ms:
        # too short to safely silence-trim — widen slightly into context
        wide_start = max(0, raw_start - min_edge_pad_ms)
        wide_end = min(track_len, raw_end + min_edge_pad_ms)
        return vocals[wide_start:wide_end]

    seg = vocals[raw_start:raw_end]
    lead = psilence.detect_leading_silence(
        seg, silence_threshold=silence_thresh_db, chunk_size=chunk_ms)
    tail = psilence.detect_leading_silence(
        seg.reverse(), silence_threshold=silence_thresh_db, chunk_size=chunk_ms)

    trim_start = max(0, lead - pad_ms)
    trim_end = max(trim_start + 1, len(seg) - max(0, tail - pad_ms))

    # hard floor of min_edge_pad_ms of real audio on each side; pull any
    # shortfall from the surrounding track instead of the slice itself
    if trim_start > min_edge_pad_ms:
        final_start_ms = raw_start + trim_start
    else:
        final_start_ms = max(0, raw_start - (min_edge_pad_ms - trim_start))

    tail_stripped = len(seg) - trim_end
    if tail_stripped > min_edge_pad_ms:
        final_end_ms = raw_start + trim_end
    else:
        final_end_ms = min(track_len, raw_end + (min_edge_pad_ms - tail_stripped))

    if final_end_ms - final_start_ms < min_edge_pad_ms * 2:
        return seg  # pathological (track edge) — raw slice beats something tiny
    return vocals[final_start_ms:final_end_ms]


# ===================== CLI port: concatenated reference =======================

def build_concat_reference(refs: list[Path], out_path: Path,
                           max_s: float = config.CHATTERBOX_MAX_REF_S) -> Path:
    """Chatterbox takes a single audio_prompt_path (no multi-reference list),
    so merge the surviving clips into one wav, capped to max_s with short
    gaps. Ported from the CLI."""
    from pydub import AudioSegment
    combined = AudioSegment.silent(duration=0)
    for p in refs:
        if len(combined) >= max_s * 1000:
            break
        combined += AudioSegment.from_wav(p) + AudioSegment.silent(duration=200)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(out_path, format="wav")
    return out_path


# ==================== project-level extraction (webapp) =======================

def extract_for_project(ps: ProjectState, project_dir: Path, vocals_path: Path,
                        on_message: Callable[[str], None] | None = None) -> dict:
    """Extract a reference clip per assigned, non-pruned line. Existing clip
    files are overwritten (line timings may have changed); pruned line_nos are
    skipped. Invalidates every touched speaker's concatenated ref."""
    from pydub import AudioSegment

    if on_message:
        on_message("loading isolated vocals track…")
    vocals = AudioSegment.from_wav(vocals_path)

    rows = [l for l in ps.data["lines"] if l["speaker"]]
    per_speaker: dict[str, int] = {}
    skipped_bad, skipped_pruned, warn_short = 0, 0, []

    for i, l in enumerate(rows, 1):
        spk = l["speaker"]
        pruned = set(ps.data["speakers"][spk].get("pruned", []))
        if l["line_no"] in pruned:
            skipped_pruned += 1
            continue
        if on_message and (i % 10 == 0 or i == len(rows)):
            on_message(f"extracting reference clips… {i}/{len(rows)}")
        start_ms, end_ms = int(l["start"] * 1000), int(l["end"] * 1000)
        if end_ms <= start_ms or start_ms >= len(vocals):
            skipped_bad += 1
            continue
        seg = extract_reference_clip(vocals, start_ms, end_ms)
        if len(seg) < 50:  # sanity floor, should be unreachable after padding
            skipped_bad += 1
            continue
        out = clip_path(project_dir, spk, l["line_no"])
        out.parent.mkdir(parents=True, exist_ok=True)
        seg.export(out, format="wav")
        per_speaker[spk] = per_speaker.get(spk, 0) + 1
        if len(seg) < config.MIN_CLIP_MS:
            warn_short.append(l["line_no"])

    for spk in per_speaker:
        invalidate_ref(project_dir, spk)   # new clips -> concatenated ref stale

    return {
        "per_speaker": per_speaker,
        "skipped_bad_timestamps": skipped_bad,
        "skipped_pruned": skipped_pruned,
        "short_clip_lines": warn_short,    # < MIN_CLIP_MS: probably weak references
    }


def list_clips(ps: ProjectState, project_dir: Path, spk: str) -> list[dict]:
    """Surviving clips for one speaker, with durations and the line's current
    subtitle text for the UI."""
    d = clips_dir(project_dir, spk)
    out = []
    if not d.is_dir():
        return out
    texts = {l["line_no"]: l["text"] for l in ps.data["lines"]}
    for p in sorted(d.glob("line_*.wav")):
        try:
            line_no = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            with wave.open(str(p), "rb") as w:
                dur = w.getnframes() / w.getframerate()
        except Exception:
            dur = None
        out.append({"line_no": line_no, "filename": p.name,
                    "text": texts.get(line_no, ""),
                    "duration_s": round(dur, 2) if dur else None,
                    "too_short": bool(dur is not None
                                      and dur * 1000 < config.MIN_CLIP_MS)})
    return out
