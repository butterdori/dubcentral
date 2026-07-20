"""Chatterbox Multilingual synthesis for the line-driven dub loop.

This module does exactly two things:
  * ensure_reference — build (or reuse) refs/ref_<spk>.wav from the speaker's
    surviving clips, with the CLI's MIN_REF_MS too-short filter. refs/ is
    derived Speaker Center state: clip deletion and re-extraction invalidate
    it; dub runs never do.
  * synthesize_line — one TTS generation to dub_work/raw/. The dub job owns
    the loop shape, OOM retry, and per-line Failed handling; keeping this
    function single-purpose is also what lets tests fake it wholesale.

torch/chatterbox imports are lazy; module import never requires them.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..gpu_service import gpu
from ..models.project_state import ProjectState
from . import reference_clips, timefit


class TTSError(RuntimeError):
    pass


def ensure_reference(ps: ProjectState, project_dir: Path, spk: str) -> Path:
    """Return refs/ref_<spk>.wav, building it from the speaker's surviving
    clips if missing. Clips shorter than MIN_REF_MS are skipped (CLI rule);
    a speaker with no usable clips is a TTSError — surfaced per line."""
    ref = reference_clips.ref_path(project_dir, spk)
    if ref.exists():
        return ref
    clips = reference_clips.list_clips(ps, project_dir, spk)
    usable = [reference_clips.clip_path(project_dir, spk, c["line_no"])
              for c in clips
              if c["duration_s"] and c["duration_s"] * 1000 >= config.MIN_REF_MS]
    if not usable:
        raise TTSError(
            f"speaker '{spk}' has no usable reference clips "
            f"(>= {config.MIN_REF_MS}ms) — extract clips in the Speaker "
            "Center, or check that pruning didn't remove everything")
    reference_clips.build_concat_reference(usable, ref)
    return ref


def synthesize_line(*, text: str, language: str, ref_path: Path,
                    exaggeration: float, cfg_weight: float, out_path: Path,
                    precision: str = config.PRECISION,
                    force_cpu: bool = False) -> float:
    """Generate one line to out_path (dub_work/raw/line_NNNN.wav). Returns the
    generated duration in seconds. Raises on failure — RuntimeError('...out of
    memory...') propagates so the dub job can empty_cache + retry once."""
    import torch
    import torchaudio as ta

    model = gpu.get_chatterbox(precision, force_cpu)
    with torch.inference_mode(), gpu.autocast(precision, force_cpu):
        try:
            wav = model.generate(
                text,
                language_id=language,
                audio_prompt_path=str(ref_path),
                exaggeration=float(exaggeration),
                cfg_weight=float(cfg_weight),
            )
        except RuntimeError as exc:
            m = str(exc).lower()
            if "half" in m or "dtype" in m:
                # this chatterbox build doesn't tolerate half-converted
                # weights — actionable message, per-line Failed (not a die())
                raise TTSError(
                    f"dtype mismatch inside Chatterbox at precision "
                    f"{precision} — this build doesn't tolerate half-converted "
                    "weights; set precision fp32 in backend/config.py (needs "
                    "more free VRAM)") from None
            raise  # OOM and everything else: the dub job decides

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(out_path), wav.float().cpu(), model.sr)
    return timefit.probe_duration(out_path)
