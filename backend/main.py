"""FastAPI app skeleton (Phase 0).

Serves the static frontend and a health endpoint. Routers (projects, upload,
speakers, dub, jobs, remix) mount here as their phases land — each import
below is uncommented when its module exists.

Run from the repo root:
    uvicorn backend.main:app --host 0.0.0.0 --port 8765
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config

app = FastAPI(title="Dubbing Webapp", docs_url="/api/docs", openapi_url="/api/openapi.json")

# ---- routers (mounted as phases land) ----
from .routers import projects, upload              # Phase 1
from .routers import speakers, jobs                # Phase 2
from .routers import dub                           # Phase 3
from .routers import remix                         # Phase 4
app.include_router(projects.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(speakers.router, prefix="/api")
app.include_router(dub.router, prefix="/api")
app.include_router(remix.router, prefix="/api")


@app.get("/api/health")
def health():
    from .gpu_service import gpu
    return {
        "ok": True,
        "storage_root": str(config.STORAGE_ROOT),
        "device": gpu.device_info(),
    }


config.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# Static frontend last, at root, html=True so / serves index.html
app.mount("/", StaticFiles(directory=config.FRONTEND_ROOT, html=True), name="frontend")
