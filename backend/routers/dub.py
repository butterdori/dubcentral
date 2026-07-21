"""Dub Center API.

  GET   /projects/{key}/dub/grid            grid rows (effective values +
                                            override sources), speakers,
                                            has_undo, video duration
  PATCH /projects/{key}/dub/lines           LIST-accepting edits (bulk UI in
                                            Phase 5 is pure fan-out onto this)
                                            -> badge deltas + updated rows
  POST  /projects/{key}/dub/run             Dub All / Dub Selected -> job
  POST  /projects/{key}/dub/undo            single-level undo (audio actions)
  GET   /projects/{key}/dub/lines/{n}/audio play a line's current fit take

The dub job is LINE-DRIVEN (a rewrite of cmd_dub's loop shape, not a lift):
each target line resolves its own knobs — two lines of one speaker can differ
in language/exaggeration/cfg. Concatenated speaker references still amortize
across lines. "Is this line dubbed" lives in project.json badges, not the
filesystem. CUDA OOM -> empty_cache -> one retry -> per-line Failed; the run
continues and the service never dies.
"""
from __future__ import annotations

import time
import traceback
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import store
from ..gpu_service import gpu
from ..job_queue import Busy, jobs
from ..models.project_state import ENGINE_FIELDS, Badge, StateError
from ..models.schema import LineEditRequest
from ..pipeline import (assembly, dub_work, timefit, tts_chatterbox,
                        tts_cosyvoice, tts_crispasr)
from .projects import get_state_or_404

router = APIRouter(prefix="/projects/{key}/dub", tags=["dub"])

# badges that Dub All considers "needs work"
_NEEDS_WORK = {Badge.NEVER.value, Badge.NEEDS_LIGHT.value,
               Badge.NEEDS_FULL.value, Badge.FAILED.value}


class SetTimestampMute(BaseModel):
    value: bool


class SetCudaEnabled(BaseModel):
    value: bool


class RunRequest(BaseModel):
    mode: Literal["all", "selected", "changed"]
    line_nos: list[int] = Field(default_factory=list)
    # for mode="changed": restrict to one regen tier (None = both)
    tier: Literal["light", "full"] | None = None


def _grid_payload(ps) -> dict:
    engine = ps.data.get("engine", "chatterbox")
    return {
        "engine": engine,
        "engine_fields": list(ENGINE_FIELDS[engine]),
        "rows": ps.grid(),
        "speakers": {k: v["display_name"]
                     for k, v in sorted(ps.data["speakers"].items())},
        "project_defaults": ps.data["project_defaults"],
        "has_undo": bool(ps.data.get("undo")),
        "has_final": assembly.final_path(
            store.project_dir(ps.data["key"])).exists(),
        "timestamp_mute": bool(ps.data.get("timestamp_mute", False)),
        "cuda_enabled": bool(ps.data.get("cuda_enabled", True)),
        "video_duration_s": ps.data.get("video_duration_s"),
    }


@router.get("/grid")
def grid(key: str):
    return _grid_payload(get_state_or_404(key))


@router.put("/timestamp_mute")
def set_timestamp_mute(key: str, body: SetTimestampMute):
    """Remix option: mute the original audio for the full subtitle
    timestamp rather than just the dub's own duration — see assembly.py."""
    get_state_or_404(key)
    with store.locked(key):
        ps = store.load(key)
        ps.set_timestamp_mute(body.value)
        ps.save()
        return {"timestamp_mute": ps.data["timestamp_mute"]}


@router.put("/cuda")
def set_cuda_enabled(key: str, body: SetCudaEnabled):
    """Troubleshooting override: disable to force every GPU op (TTS +
    Demucs) for this project onto CPU regardless of CUDA availability. Read
    once at the start of each dub/extract run — never changes mid-run."""
    get_state_or_404(key)
    with store.locked(key):
        ps = store.load(key)
        ps.set_cuda_enabled(body.value)
        ps.save()
        return {"cuda_enabled": ps.data["cuda_enabled"]}


@router.patch("/lines")
def edit_lines(key: str, body: LineEditRequest):
    """Apply a batch of line edits. Badge transitions come from the Phase-0
    table inside project_state — never recomputed here."""
    get_state_or_404(key)
    with store.locked(key):
        ps = store.load(key)
        try:
            deltas = ps.apply_line_edits([e.model_dump() for e in body.edits])
        except StateError as e:
            code = 409 if "generating" in str(e) else 400
            raise HTTPException(code, str(e)) from None
        ps.save()
        touched = {e.line_no for e in body.edits}
        rows = [r for r in ps.grid() if r["line_no"] in touched]
        return {"deltas": deltas, "rows": rows, "has_undo": bool(ps.data.get("undo"))}


@router.get("/lines/{line_no}/audio")
def line_audio(key: str, line_no: int):
    get_state_or_404(key)
    p = dub_work.fit_path(store.project_dir(key), line_no)
    if not p.exists():
        raise HTTPException(404, "no dubbed audio for this line yet")
    return FileResponse(p, media_type="audio/wav")


# ------------------------------- the dub run ----------------------------------

def _tier_for(line: dict) -> str:
    """needs_light -> refit only; everything else (never / needs_full /
    failed / explicitly re-selected clean line) -> full TTS generation.
    Re-dubbing a clean selected line is a deliberate re-roll — Chatterbox
    generation is stochastic, so that's a real use case."""
    return "light" if line["badge"] == Badge.NEEDS_LIGHT.value else "full"


@router.post("/run", status_code=202)
def run_dub(key: str, body: RunRequest):
    ps0 = get_state_or_404(key)
    d = store.project_dir(key)

    dubbable = {l["line_no"] for l in ps0.data["lines"]
                if l["speaker"] and l["text"]}
    if body.mode == "all":
        targets = [n for n in sorted(dubbable)
                   if ps0.line(n)["badge"] in _NEEDS_WORK]
    elif body.mode == "changed":
        # only lines whose EXISTING take went stale (edits after a dub) —
        # never-dubbed and failed lines are left for Dub All
        want = {Badge.NEEDS_LIGHT.value, Badge.NEEDS_FULL.value}
        if body.tier == "light":
            want = {Badge.NEEDS_LIGHT.value}
        elif body.tier == "full":
            want = {Badge.NEEDS_FULL.value}
        targets = [n for n in sorted(dubbable) if ps0.line(n)["badge"] in want]
    else:
        if not body.line_nos:
            raise HTTPException(400, "no lines selected")
        missing = [n for n in body.line_nos if n not in dubbable]
        if missing:
            raise HTTPException(
                400, f"lines not dubbable (no speaker/text): {sorted(missing)}")
        targets = sorted(set(body.line_nos))
    if not targets:
        if body.mode == "changed":
            raise HTTPException(400, "nothing to dub — no lines are flagged "
                                     "as changed" +
                                     (f" ({body.tier} tier)" if body.tier else ""))
        if body.mode == "all" and not dubbable:
            raise HTTPException(400, "nothing to dub — no lines have both a "
                                     "speaker and text (diarize in the Sub "
                                     "Center first)")
        raise HTTPException(400, "nothing to dub — every line is clean")

    def run(job):
        t0 = time.time()
        # ---- snapshot + mark: one locked pass ----
        with store.locked(key):
            ps = store.load(key)
            engine = ps.data.get("engine", "chatterbox")
            force_cpu = not ps.data.get("cuda_enabled", True)
            plan = []   # per-line work orders w/ resolved knobs
            dub_work.clear_undo(d)
            entries = {}
            for n in targets:
                line = ps.line(n)
                tier = _tier_for(line)
                entries[n] = {
                    "had_take": dub_work.raw_path(d, n).exists(),
                    "prev_badge": line["badge"],
                    "prev_raw_duration_s": line["raw_duration_s"],
                    "prev_fit_duration_s": line.get("fit_duration_s"),
                }
                dub_work.displace_to_undo(d, n, tier)
                plan.append({
                    "line_no": n, "tier": tier,
                    "fields": ps.resolve(line),
                    "slot_s": max(0.0, line["end"] - line["start"]),
                    "text": line["text"], "speaker": line["speaker"],
                })
            job.result["engine"] = engine
            ps.record_undo(entries)
            job.add_deltas(ps.mark_generating(targets))
            ps.save()
            ps_ro = ps   # read-only uses below (references)

        # ---- per-line loop (no lock held during GPU work) ----
        ref_cache: dict[str, object] = {}   # spk -> Path | TTSError
        n_ok = n_failed = 0
        hard_stretched: list[int] = []
        errors: dict[int, str] = {}
        total_raw = total_slot = 0.0
        print(f"[dub] {key}: starting run [{engine}"
              f"{'/cpu-forced' if force_cpu else ''}] on {len(plan)} lines "
              f"({sum(1 for x in plan if x['tier'] == 'full')} tts, "
              f"{sum(1 for x in plan if x['tier'] == 'light')} refit)",
              flush=True)

        for i, p in enumerate(plan, 1):
            n = p["line_no"]
            job.set_message(f"line {n} ({i}/{len(plan)}, "
                            f"{'refit' if p['tier'] == 'light' else 'tts'})…")
            try:
                raw = dub_work.raw_path(d, n)
                if p["tier"] == "full":
                    spk = p["speaker"]
                    f = p["fields"]
                    if spk not in ref_cache:
                        try:
                            if engine in ("cosyvoice3", "crispasr"):
                                # same zero-shot single-clip+transcript
                                # pattern for both -- crispasr's cosyvoice3
                                # backend needs identical reference material
                                ref_cache[spk] = tts_cosyvoice.select_reference(
                                    ps_ro, d, spk)   # (clip_path, original_text)
                            else:
                                ref_cache[spk] = tts_chatterbox.ensure_reference(
                                    ps_ro, d, spk)
                        except (tts_chatterbox.TTSError,
                                tts_cosyvoice.CosyVoiceError) as e:
                            ref_cache[spk] = e
                    if isinstance(ref_cache[spk], Exception):
                        raise ref_cache[spk]
                    if engine == "cosyvoice3":
                        ref_wav, prompt_text = ref_cache[spk]
                        synth = tts_cosyvoice.synthesize_line
                        kwargs = dict(
                            text=p["text"],
                            prompt_text=prompt_text,
                            ref_path=ref_wav,
                            speed=f["speed"]["value"],
                            instruct_text=f["instruct_text"]["value"],
                            out_path=raw,
                            force_cpu=force_cpu,
                        )
                    elif engine == "crispasr":
                        ref_wav, prompt_text = ref_cache[spk]
                        synth = tts_crispasr.synthesize_line
                        kwargs = dict(
                            text=p["text"],
                            prompt_text=prompt_text,
                            ref_path=ref_wav,
                            backend=f["crispasr_backend"]["value"],
                            out_path=raw,
                            force_cpu=force_cpu,
                        )
                    else:
                        synth = tts_chatterbox.synthesize_line
                        kwargs = dict(
                            text=p["text"],
                            language=f["dub_language"]["value"],
                            ref_path=ref_cache[spk],
                            exaggeration=f["exaggeration"]["value"],
                            cfg_weight=f["cfg_weight"]["value"],
                            out_path=raw,
                            force_cpu=force_cpu,
                        )
                    try:
                        dur = synth(**kwargs)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        # OOM: empty_cache + ONE retry, then per-line Failed
                        job.set_message(f"line {n}: CUDA OOM — retrying once…")
                        gpu.empty_cache()
                        dur = synth(**kwargs)
                else:
                    if not raw.exists():
                        raise RuntimeError(
                            "raw take missing for light refit — run a full "
                            "regen on this line")
                    dur = timefit.probe_duration(raw)

                fit = timefit.fit_line(raw, dub_work.fit_path(d, n),
                                       p["slot_s"], p["fields"])
                if fit["hard_stretch"]:
                    hard_stretched.append(n)
                total_raw += dur
                total_slot += p["slot_s"]
                with store.locked(key):
                    ps = store.load(key)
                    job.add_deltas(ps.mark_success(
                        n, raw_duration_s=dur,
                        fit_duration_s=fit["fit_duration_s"]))
                    ps.save()
                n_ok += 1
                print(f"[dub] {key}: line {n} ok "
                      f"({p['tier']}, {dur:.2f}s raw, factor "
                      f"{fit['factor']})", flush=True)
            except Exception as e:  # noqa: BLE001 — per-line Failed, run continues
                print(f"[dub] {key}: line {n} FAILED: {e}", flush=True)
                traceback.print_exc()
                errors[n] = str(e)
                with store.locked(key):
                    ps = store.load(key)
                    job.add_deltas(ps.mark_failed(n, str(e)))
                    ps.save()
                n_failed += 1

        job.set_message("done")
        print(f"[dub] {key}: run finished — ok {n_ok}, failed {n_failed}, "
              f"{time.time() - t0:.1f}s", flush=True)
        return {
            "n_ok": n_ok,
            "n_failed": n_failed,
            # first few per-line errors for the UI/API (full text also lives
            # on each line: hover the ⚠ badge, or GET /dub/grid -> rows[].error)
            "errors": {n: msg for n, msg in list(errors.items())[:5]},
            "elapsed_s": round(time.time() - t0, 1),
            # "dub audio is 1.21x the original runtime" for the lines processed
            "raw_vs_slot_ratio": round(total_raw / total_slot, 3)
                                 if total_slot else None,
            "hard_stretched_lines": hard_stretched,
        }

    try:
        job = jobs.submit("dub", key, run)
    except Busy as e:
        raise HTTPException(409, str(e)) from None
    return {"job": job.snapshot(), "n_targets": len(targets)}


@router.post("/undo")
def undo(key: str):
    get_state_or_404(key)
    if jobs.busy():
        raise HTTPException(409, "a job is running — undo after it finishes")
    d = store.project_dir(key)
    with store.locked(key):
        ps = store.load(key)
        try:
            entries = ps.consume_undo()
        except StateError as e:
            raise HTTPException(400, str(e)) from None
        for n, meta in entries.items():
            dub_work.restore_from_undo(d, n, meta.get("had_take", False))
        dub_work.clear_undo(d)
        ps.save()
        deltas = {n: ps.line(n)["badge"] for n in entries}
        return {"restored": sorted(entries), "deltas": deltas,
                "has_undo": False}
