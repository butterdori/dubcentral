"""Final Remix API.

  POST /projects/{key}/remix           -> job (runs Demucs first if needed)
  GET  /projects/{key}/remix/download  -> final.mp4
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import store
from ..job_queue import Busy, jobs
from ..pipeline import assembly
from .projects import get_state_or_404

router = APIRouter(prefix="/projects/{key}", tags=["remix"])


@router.post("/remix", status_code=202)
def run_remix(key: str):
    ps0 = get_state_or_404(key)
    d = store.project_dir(key)
    if not (d / store.VIDEO_NAME).exists():
        raise HTTPException(400, "upload a video first")
    if not assembly.spliceable_lines(ps0, d):
        raise HTTPException(400, "no successfully dubbed lines to splice — "
                                 "run a dub first")

    def run(job):
        # a consistent read of the line table; dubbed takes on disk are the
        # audio truth and the single job slot prevents concurrent dub runs
        with store.locked(key):
            ps = store.load(key)
        result = assembly.remix(ps, d, on_message=job.set_message)
        job.set_message("done")
        return result

    try:
        job = jobs.submit("remix", key, run)
    except Busy as e:
        raise HTTPException(409, str(e)) from None
    return {"job": job.snapshot()}


@router.get("/remix/video")
def final_video(key: str):
    """Inline-playable final.mp4 for the Upload Center's Original/Dubbed
    toggle -- unlike /remix/download, no filename= is set, so the browser
    streams and plays it in a <video> tag instead of downloading (mirrors
    upload.py's get_video for the same reason)."""
    get_state_or_404(key)
    p = assembly.final_path(store.project_dir(key))
    if not p.exists():
        raise HTTPException(404, "no remixed output yet — run Final Remix first")
    return FileResponse(p, media_type="video/mp4")


@router.get("/remix/download")
def download(key: str):
    get_state_or_404(key)
    p = assembly.final_path(store.project_dir(key))
    if not p.exists():
        raise HTTPException(404, "no remixed output yet — run Final Remix first")
    return FileResponse(p, media_type="video/mp4",
                        filename=f"{key}_dubbed.mp4")
