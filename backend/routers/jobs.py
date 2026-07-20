"""GET /jobs/current — the single polling channel driving live status and
per-line badge flips in the UI. No SSE/websockets at this scale."""
from __future__ import annotations

from fastapi import APIRouter

from ..gpu_service import gpu
from ..job_queue import jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/current")
def current_job():
    job = jobs.current()
    return {
        "job": job.snapshot() if job else None,
        "busy": jobs.busy(),
        "device": gpu.device(),
    }
