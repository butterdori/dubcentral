"""Speaker Center API.

  GET    /projects/{key}/speakers                      list (defaults + clips)
  POST   /projects/{key}/speakers                      add speaker
  PATCH  /projects/{key}/speakers/{spk}                rename (display only)
  PUT    /projects/{key}/speakers/{spk}/defaults       set/clear one default
                                                       -> badge deltas (cascade)
  GET    /projects/{key}/speakers/{spk}/clips/{line}   serve clip wav (play)
  DELETE /projects/{key}/speakers/{spk}/clips/{line}   prune clip
                                                       -> refs/ invalidated
  POST   /projects/{key}/extract_clips                 job: Demucs (if needed)
                                                       + reference extraction

No speaker delete — intentionally out of scope, same as merge/split.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import store
from ..job_queue import Busy, jobs
from ..models.project_state import INHERITABLE_FIELDS, StateError
from ..pipeline import demucs_runner, reference_clips, tts_cosyvoice
from .projects import get_state_or_404

router = APIRouter(prefix="/projects/{key}", tags=["speakers"])


class AddSpeaker(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class Rename(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)


class SetDefault(BaseModel):
    field: str
    value: Any = None      # None clears the default (re-inherit from project)


def _speaker_payload(ps, project_dir, spk: str) -> dict:
    s = ps.data["speakers"][spk]
    effective = {}
    for f in INHERITABLE_FIELDS:
        d = s["defaults"]
        effective[f] = {
            "value": d[f] if f in d else ps.data["project_defaults"][f],
            "source": "speaker" if f in d else "project",
        }
    n_lines = sum(1 for l in ps.data["lines"] if l["speaker"] == spk)
    clips = reference_clips.list_clips(ps, project_dir, spk)

    # CosyVoice reference indicator — cheap/pure, harmless to compute even
    # under chatterbox (just unused by that engine's UI). ref_pins is the
    # STAGED selection; ref_pins_committed is what the last "Renew
    # reference" actually built into the concatenated file currently in
    # use (may differ if pins changed since, or some were skipped as
    # stale) — is_reference marks whichever clips are ACTUALLY driving
    # synthesis right now: the committed set if a concat file exists,
    # else the single auto-picked clip.
    ref_pins = sorted(s.get("ref_pins", []))
    ref_pins_committed = sorted(s.get("ref_pins_committed", []))
    has_committed_ref = tts_cosyvoice.cosy_ref_path(project_dir, spk).exists()
    if has_committed_ref:
        active = set(ref_pins_committed)
    else:
        auto = tts_cosyvoice.auto_pick_line(ps, project_dir, spk)
        active = {auto} if auto is not None else set()
    for c in clips:
        c["is_reference"] = c["line_no"] in active
    return {
        "key": spk,
        "display_name": s["display_name"],
        "defaults": dict(s["defaults"]),          # explicit only
        "effective": effective,                   # what non-overridden lines get
        "n_lines": n_lines,
        "pruned": sorted(s.get("pruned", [])),
        "clips": clips,
        "ref_pins": ref_pins,
        "ref_pins_committed": ref_pins_committed,
        "ref_dirty": ref_pins != ref_pins_committed,
    }


@router.get("/speakers")
def list_speakers(key: str):
    ps = get_state_or_404(key)
    d = store.project_dir(key)
    return {
        "engine": ps.data.get("engine", "chatterbox"),
        "project_defaults": ps.data["project_defaults"],
        "has_original_text": any((l.get("original_text") or "").strip()
                                 for l in ps.data["lines"]),
        "speakers": [_speaker_payload(ps, d, spk)
                     for spk in sorted(ps.data["speakers"])],
    }


@router.post("/speakers", status_code=201)
def add_speaker(key: str, body: AddSpeaker):
    get_state_or_404(key)
    with store.locked(key):
        ps = store.load(key)
        try:
            spk = ps.add_speaker(body.name)
        except StateError as e:
            raise HTTPException(409, str(e)) from None
        ps.save()
        return _speaker_payload(ps, store.project_dir(key), spk)


@router.patch("/speakers/{spk}")
def rename_speaker(key: str, spk: str, body: Rename):
    get_state_or_404(key)
    with store.locked(key):
        ps = store.load(key)
        try:
            ps.rename_speaker(spk, body.display_name)
        except StateError as e:
            raise HTTPException(404, str(e)) from None
        ps.save()
        return _speaker_payload(ps, store.project_dir(key), spk)


@router.put("/speakers/{spk}/defaults")
def set_default(key: str, spk: str, body: SetDefault):
    """Set/clear a speaker default. The cascade (project_state owns it) flips
    every non-overridden line of this speaker to the field's regen tier."""
    get_state_or_404(key)
    with store.locked(key):
        ps = store.load(key)
        try:
            deltas = ps.set_speaker_default(spk, body.field, body.value)
        except StateError as e:
            code = 404 if "no such speaker" in str(e) else 400
            raise HTTPException(code, str(e)) from None
        ps.save()
        return {"speaker": _speaker_payload(ps, store.project_dir(key), spk),
                "deltas": deltas}


@router.get("/speakers/{spk}/clips/{line_no}")
def get_clip(key: str, spk: str, line_no: int):
    get_state_or_404(key)
    p = reference_clips.clip_path(store.project_dir(key), spk, line_no)
    if not p.exists():
        raise HTTPException(404, "no such clip")
    return FileResponse(p, media_type="audio/wav")


@router.delete("/speakers/{spk}/clips/{line_no}")
def delete_clip(key: str, spk: str, line_no: int):
    """Manual pruning, in-browser instead of a file manager. Remembered in
    project.json so re-extraction never resurrects it; invalidates the
    speaker's chatterbox concatenated reference (derived state). Also
    unstages this clip from the CosyVoice ref_pins set if present -- a
    staged pin naming a file that no longer exists isn't useful to keep
    checked (this does NOT touch an already-committed/renewed reference
    file, which is self-contained; it just means the staged set no longer
    matches what was committed, surfaced as ref_dirty)."""
    get_state_or_404(key)
    d = store.project_dir(key)
    with store.locked(key):
        ps = store.load(key)
        try:
            s = ps.speaker(spk)
        except StateError as e:
            raise HTTPException(404, str(e)) from None
        p = reference_clips.clip_path(d, spk, line_no)
        if not p.exists():
            raise HTTPException(404, "no such clip")
        p.unlink()
        pruned = set(s.setdefault("pruned", []))
        pruned.add(line_no)
        s["pruned"] = sorted(pruned)
        pins = set(s.get("ref_pins", []))
        if line_no in pins:
            pins.discard(line_no)
            s["ref_pins"] = sorted(pins)
        reference_clips.invalidate_ref(d, spk)
        ps.save()
        return _speaker_payload(ps, d, spk)


class SetRefPins(BaseModel):
    line_nos: list[int] = Field(default_factory=list)  # full replace; [] = unpin everything


@router.put("/speakers/{spk}/reference_pins")
def set_reference_pins(key: str, spk: str, body: SetRefPins):
    """Stage which clips will be concatenated into this speaker's CosyVoice
    reference on the next 'Renew reference'. Pure staging -- no badge
    cascade here (see project_state.set_speaker_ref_pins); nothing about
    synthesis changes until Renew actually commits a new build."""
    get_state_or_404(key)
    d = store.project_dir(key)
    with store.locked(key):
        ps = store.load(key)
        for n in body.line_nos:
            if not reference_clips.clip_path(d, spk, n).exists():
                raise HTTPException(404, f"no such clip for line {n}")
        try:
            ps.set_speaker_ref_pins(spk, body.line_nos)
        except StateError as e:
            code = 404 if "no such speaker" in str(e) else 400
            raise HTTPException(code, str(e)) from None
        ps.save()
        return _speaker_payload(ps, d, spk)


@router.post("/renew_reference", status_code=202)
def renew_reference(key: str):
    """Project-wide job: for every speaker, (re)build the concatenated
    CosyVoice reference from their currently staged ref_pins (pins that
    have gone stale since staging -- pruned, or lost original_text -- are
    skipped, not a hard failure). A speaker with no pins staged reverts to
    single-clip auto-pick. Badge cascades to needs_full happen HERE, per
    speaker, only for speakers whose committed set actually changed --
    not at pin-staging time. Pure file/audio work, no GPU, but still
    routed through the job queue for progress reporting and to avoid
    racing a concurrent dub/extract run touching the same files."""
    ps0 = get_state_or_404(key)
    d = store.project_dir(key)

    def run(job):
        results = {}
        all_deltas = {}
        for spk in sorted(ps0.data["speakers"]):
            job.set_message(f"renewing reference for {spk}…")
            with store.locked(key):
                ps = store.load(key)
                out = tts_cosyvoice.renew_reference(ps, d, spk)
                prev_committed = sorted(ps.data["speakers"][spk].get(
                    "ref_pins_committed", []))
                if sorted(out["committed"]) != prev_committed:
                    deltas = ps.mark_speaker_reference_renewed(
                        spk, out["committed"])
                    all_deltas.update(deltas)
                    job.add_deltas(deltas)
                else:
                    ps.data["speakers"][spk]["ref_pins_committed"] = \
                        sorted(out["committed"])
                    ps._touch()
                ps.save()
            results[spk] = out
        job.set_message("done")
        return {"speakers": results, "deltas": all_deltas}

    try:
        job = jobs.submit("renew_reference", key, run)
    except Busy as e:
        raise HTTPException(409, str(e)) from None
    return {"job": job.snapshot()}


@router.post("/extract_clips", status_code=202)
def extract_clips(key: str):
    """One job: Demucs separation (skipped if cached) then per-line reference
    extraction. 409 while any job is active (single GPU slot)."""
    ps = get_state_or_404(key)
    d = store.project_dir(key)
    if not (d / store.VIDEO_NAME).exists():
        raise HTTPException(400, "upload a video first")
    if not any(l["speaker"] for l in ps.data["lines"]):
        raise HTTPException(400, "no lines have a speaker assigned — "
                                 "diarize segments.csv first")

    def run(job):
        # Demucs always uses GPU when available — the Force CPU checkbox
        # (project.cuda_enabled) is scoped to TTS/dubbing only, not extraction
        vocals, _ = demucs_runner.separate(d, on_message=job.set_message)
        with store.locked(key):
            ps2 = store.load(key)
            summary = reference_clips.extract_for_project(
                ps2, d, vocals, on_message=job.set_message)
            ps2.save()   # touch modified_at; pruned lists unchanged
        job.set_message("done")
        return summary

    try:
        job = jobs.submit("extract_clips", key, run)
    except Busy as e:
        raise HTTPException(409, str(e)) from None
    return {"job": job.snapshot()}
