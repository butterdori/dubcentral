"""Upload Center API: video and SRT uploads with real validation (surfaced as
400s, never silent), the prep port (SRT -> line table), and the segments.csv
interchange endpoints (download / editor save / file re-upload).

Reset semantics (the confirmations live in the frontend; the API just does it):
  * SRT upload, CSV editor save, CSV re-upload all REBUILD the line table:
    overrides, badges, dub progress, and the undo slot are wiped, and the
    state-derived audio dirs (speaker_clips/, refs/, dub_work/) are removed.
  * separated/ (Demucs) survives — it belongs to the video, not the lines.
  * Video re-upload does NOT touch the line table, but it wipes separated/
    (stale for the new video) and regenerates the thumbnail/duration.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from .. import config, store
from ..pipeline import prep
from ..pipeline.media import MediaError, grab_thumbnail, probe_video
from .projects import get_state_or_404, _meta

router = APIRouter(prefix="/projects/{key}", tags=["upload"])


async def _save_upload(file: UploadFile, dest: Path) -> None:
    """Stream the upload to a temp file in dest's directory, then atomically
    replace — a failed/aborted upload never clobbers an existing good file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".upload.")
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _reset_lines(key: str, ps, rows) -> None:
    """Shared by SRT upload / CSV save / CSV re-upload: rebuild the line
    table (full reset) and wipe state-derived audio dirs."""
    ps.data["speakers"] = {}          # speakers rebuild from the rows
    ps.init_lines_from_segments(rows)
    store.wipe_derived(key)
    # refresh the interchange snapshot on disk
    (store.project_dir(key) / config.CSV_NAME).write_text(
        prep.state_to_csv_text(ps), encoding="utf-8")
    ps.save()


# --------------------------------- video -------------------------------------

@router.post("/upload/video")
async def upload_video(key: str, file: UploadFile):
    get_state_or_404(key)
    d = store.project_dir(key)
    dest = d / store.VIDEO_NAME
    with store.locked(key):
        ps = store.load(key)
        await _save_upload(file, dest)
        try:
            info = probe_video(dest)
        except MediaError as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, f"video rejected: {e}") from None
        if not info["has_audio"]:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "video rejected: no audio track — "
                                     "nothing to separate or dub over")
        shutil.rmtree(d / "separated", ignore_errors=True)  # stale for new video
        ps.data["video_duration_s"] = info["duration_s"]
        try:
            grab_thumbnail(dest, d / store.THUMB_NAME, info["duration_s"])
        except MediaError:
            pass  # thumbnail is cosmetic; never fail the upload over it
        ps.save()
        return _meta(ps)


@router.get("/video")
def get_video(key: str):
    get_state_or_404(key)
    p = store.project_dir(key) / store.VIDEO_NAME
    if not p.exists():
        raise HTTPException(404, "no video uploaded yet")
    # Starlette FileResponse handles Range requests -> the <video> player seeks
    return FileResponse(p, media_type="video/mp4")


# ---------------------------------- SRT --------------------------------------

@router.post("/upload/srt")
async def upload_srt(key: str, file: UploadFile):
    get_state_or_404(key)
    data = await file.read()
    try:
        rows = prep.parse_srt_bytes(data)
    except prep.PrepError as e:
        raise HTTPException(400, f"SRT rejected: {e}") from None
    with store.locked(key):
        ps = store.load(key)
        (store.project_dir(key) / store.SRT_NAME).write_bytes(data)
        _reset_lines(key, ps, rows)
        return {"meta": _meta(ps), "n_lines": len(rows)}


ORIGINAL_SRT_NAME = "original.srt"


@router.post("/upload/original_srt")
async def upload_original_srt(key: str, file: UploadFile):
    """Original-language SRT (Phase 5): populates per-line original_text —
    the transcript CosyVoice's zero-shot path needs for the reference clip.
    PURELY ADDITIVE: no line-table reset, no badge flips (references are
    selected at run time). Alignment: by SRT index first; unmatched source
    entries then fall back to the nearest-start line within 1.0s."""
    get_state_or_404(key)
    data = await file.read()
    try:
        rows = prep.parse_srt_bytes(data)
    except prep.PrepError as e:
        raise HTTPException(400, f"original SRT rejected: {e}") from None
    with store.locked(key):
        ps = store.load(key)
        if not ps.data["lines"]:
            raise HTTPException(400, "upload the translated SRT first — "
                                     "original_text attaches to its lines")
        (store.project_dir(key) / ORIGINAL_SRT_NAME).write_bytes(data)
        by_no = {l["line_no"]: l for l in ps.data["lines"]}
        matched_index = matched_ts = unmatched = 0
        taken: set[int] = set()
        leftovers = []
        for r in rows:
            line = by_no.get(r["line_no"])
            if line is not None:
                line["original_text"] = r["text"] or None
                taken.add(line["line_no"])
                matched_index += 1
            else:
                leftovers.append(r)
        for r in leftovers:
            best, best_d = None, 1.0
            for l in ps.data["lines"]:
                if l["line_no"] in taken:
                    continue
                d = abs(l["start"] - r["start"])
                if d <= best_d:
                    best, best_d = l, d
            if best is not None:
                best["original_text"] = r["text"] or None
                taken.add(best["line_no"])
                matched_ts += 1
            else:
                unmatched += 1
        ps._touch()
        ps.save()
        missing = sum(1 for l in ps.data["lines"]
                      if not (l.get("original_text") or "").strip())
        return {"matched_by_index": matched_index,
                "matched_by_timestamp": matched_ts,
                "unmatched_source_entries": unmatched,
                "lines_without_original_text": missing}


@router.get("/srt/export")
def export_srt(key: str):
    """Download the CURRENT line table as an SRT (timings/text as edited in
    the Sub/Dub Centers; speakers live in segments.csv)."""
    ps = get_state_or_404(key)
    if not ps.data["lines"]:
        raise HTTPException(404, "no segments yet — upload an SRT first")
    return PlainTextResponse(
        prep.state_to_srt_text(ps), media_type="application/x-subrip",
        headers={"Content-Disposition":
                 f'attachment; filename="{key}_edited.srt"'})


@router.post("/srt/revert")
def revert_srt(key: str):
    """Sub Center 'Revert': rebuild the line table from the ORIGINAL uploaded
    source.srt. Same full-reset semantics as re-uploading it."""
    get_state_or_404(key)
    src = store.project_dir(key) / store.SRT_NAME
    if not src.exists():
        raise HTTPException(400, "no uploaded SRT to revert to")
    try:
        rows = prep.parse_srt_bytes(src.read_bytes())
    except prep.PrepError as e:
        raise HTTPException(400, f"stored SRT unreadable: {e}") from None
    with store.locked(key):
        ps = store.load(key)
        _reset_lines(key, ps, rows)
        return {"meta": _meta(ps), "n_lines": len(rows)}


# ---------------------------------- CSV --------------------------------------

@router.get("/csv")
def download_csv(key: str):
    ps = get_state_or_404(key)
    if not ps.data["lines"]:
        raise HTTPException(404, "no segments yet — upload an SRT first")
    return PlainTextResponse(
        prep.state_to_csv_text(ps), media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="{key}_segments.csv"'})


@router.put("/csv")
def save_csv(key: str, text: str = Body(..., media_type="text/csv")):
    """CSV editor save. Full line-table reset (frontend confirms first)."""
    get_state_or_404(key)
    try:
        rows = prep.csv_text_to_rows(text)
    except prep.PrepError as e:
        raise HTTPException(400, f"CSV rejected: {e}") from None
    with store.locked(key):
        ps = store.load(key)
        _reset_lines(key, ps, rows)
        return {"meta": _meta(ps), "n_lines": len(rows)}


@router.post("/upload/csv")
async def upload_csv(key: str, file: UploadFile):
    """Bring-your-own segments.csv (e.g. hand-diarized from the CLI days).
    Same full-reset semantics as the editor save."""
    get_state_or_404(key)
    data = await file.read()
    try:
        rows = prep.csv_text_to_rows(data.decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise HTTPException(400, "CSV rejected: not UTF-8") from None
    except prep.PrepError as e:
        raise HTTPException(400, f"CSV rejected: {e}") from None
    with store.locked(key):
        ps = store.load(key)
        _reset_lines(key, ps, rows)
        return {"meta": _meta(ps), "n_lines": len(rows)}
