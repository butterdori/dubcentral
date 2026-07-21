"""Project Center API: list/create/delete projects, per-project meta,
thumbnail."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from typing import Any

from .. import store
from ..models.project_state import ProjectState, StateError

router = APIRouter(prefix="/projects", tags=["projects"])


class CreateProject(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SetProjectDefault(BaseModel):
    field: str
    value: Any


class SetEngine(BaseModel):
    engine: str


def _meta(ps: ProjectState) -> dict:
    key = ps.data["key"]
    d = store.project_dir(key)
    from ..pipeline import assembly
    return {
        "key": key,
        "name": ps.data["name"],
        "created_at": ps.data["created_at"],
        "modified_at": ps.data["modified_at"],
        "has_video": (d / store.VIDEO_NAME).exists(),
        "has_srt": (d / store.SRT_NAME).exists(),
        "has_thumbnail": (d / store.THUMB_NAME).exists(),
        "has_final": assembly.final_path(d).exists(),
        "stats": store.stats(ps),
    }


def get_state_or_404(key: str) -> ProjectState:
    try:
        return store.load(key)
    except StateError:
        raise HTTPException(404, f"no such project: {key}") from None


@router.get("")
def list_projects():
    metas = [_meta(store.load(k)) for k in store.list_keys()]
    metas.sort(key=lambda m: m["modified_at"], reverse=True)  # recency sort
    return {"projects": metas}


@router.post("", status_code=201)
def create_project(body: CreateProject):
    try:
        ps = store.create(body.name)
    except StateError as e:
        raise HTTPException(409, str(e)) from None
    return _meta(ps)


@router.get("/{key}")
def get_project(key: str):
    return _meta(get_state_or_404(key))


@router.delete("/{key}", status_code=204)
def delete_project(key: str):
    with store.locked(key):
        try:
            store.delete(key)
        except StateError as e:
            raise HTTPException(404, str(e)) from None


@router.put("/{key}/engine")
def set_engine(key: str, body: SetEngine):
    """Per-project synthesis-engine switch (chatterbox | cosyvoice3 | crispasr).
    Semantics live in project_state.set_engine: takes flip to needs_full,
    the inactive engine's overrides/speaker defaults are cleared. The
    confirmation dialog is the frontend's job; the API just does it."""
    from ..job_queue import jobs
    get_state_or_404(key)
    if jobs.busy():
        raise HTTPException(409, "a job is running — switch engines after "
                                 "it finishes")
    with store.locked(key):
        ps = store.load(key)
        try:
            out = ps.set_engine(body.engine)
        except StateError as e:
            raise HTTPException(400, str(e)) from None
        ps.save()
        out["engine"] = ps.data["engine"]
        return out


@router.put("/{key}/defaults")
def set_project_default(key: str, body: SetProjectDefault):
    """Change a project-wide default. Cascades (badge flips) to every line
    that inherits it all the way down — the table in project_state decides."""
    get_state_or_404(key)
    with store.locked(key):
        ps = store.load(key)
        try:
            deltas = ps.set_project_default(body.field, body.value)
        except StateError as e:
            raise HTTPException(400, str(e)) from None
        ps.save()
        return {"project_defaults": ps.data["project_defaults"],
                "deltas": deltas}


@router.get("/{key}/thumbnail")
def thumbnail(key: str):
    get_state_or_404(key)
    p = store.project_dir(key) / store.THUMB_NAME
    if not p.exists():
        raise HTTPException(404, "no thumbnail yet — upload a video first")
    return FileResponse(p, media_type="image/jpeg")
