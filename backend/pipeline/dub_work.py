"""dub_work/ layout — the raw-vs-fit take storage contract:

    dub_work/raw/line_NNNN.wav   canonical synthesized take; only FULL regen
                                 (a new TTS generation) ever writes here
    dub_work/fit/line_NNNN.wav   stretched/placed version; LIGHT regen rewrites
                                 this from the untouched raw
    dub_work/undo/               raw+fit pairs displaced by the LAST dub action
                                 only (single-level undo)
"""
from __future__ import annotations

import shutil
from pathlib import Path


def raw_path(project_dir: Path, line_no: int) -> Path:
    return project_dir / "dub_work" / "raw" / f"line_{line_no:04d}.wav"


def fit_path(project_dir: Path, line_no: int) -> Path:
    return project_dir / "dub_work" / "fit" / f"line_{line_no:04d}.wav"


def undo_dir(project_dir: Path) -> Path:
    return project_dir / "dub_work" / "undo"


def undo_raw(project_dir: Path, line_no: int) -> Path:
    return undo_dir(project_dir) / f"raw_line_{line_no:04d}.wav"


def undo_fit(project_dir: Path, line_no: int) -> Path:
    return undo_dir(project_dir) / f"fit_line_{line_no:04d}.wav"


def clear_undo(project_dir: Path) -> None:
    shutil.rmtree(undo_dir(project_dir), ignore_errors=True)
    undo_dir(project_dir).mkdir(parents=True, exist_ok=True)


def displace_to_undo(project_dir: Path, line_no: int, tier: str) -> None:
    """Move the take a dub run is about to overwrite into undo/.
    FULL regen displaces raw+fit; LIGHT regen displaces only fit (raw is
    never touched by light regen, so there is nothing of it to restore)."""
    undo_dir(project_dir).mkdir(parents=True, exist_ok=True)
    if tier == "full":
        r = raw_path(project_dir, line_no)
        if r.exists():
            r.rename(undo_raw(project_dir, line_no))
    f = fit_path(project_dir, line_no)
    if f.exists():
        f.rename(undo_fit(project_dir, line_no))


def restore_from_undo(project_dir: Path, line_no: int, had_take: bool) -> None:
    """Undo one line: discard the current take, put the displaced one back.
    A line that had no prior take (had_take=False) reverts to never-dubbed —
    its current files are simply deleted."""
    raw, fit = raw_path(project_dir, line_no), fit_path(project_dir, line_no)
    ur, uf = undo_raw(project_dir, line_no), undo_fit(project_dir, line_no)
    if not had_take:
        raw.unlink(missing_ok=True)
        fit.unlink(missing_ok=True)
        return
    if ur.exists():                      # full regen was undone
        raw.unlink(missing_ok=True)
        ur.rename(raw)
    # (no undo raw + had_take) == light regen: current raw IS the prior raw
    if uf.exists():
        fit.unlink(missing_ok=True)
        uf.rename(fit)
