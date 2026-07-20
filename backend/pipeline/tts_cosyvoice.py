"""Fun-CosyVoice3 engine (Phase 5).

Mirrors tts_chatterbox's contract — reference selection + synthesize_line —
so the dub loop dispatches per-project without caring which engine runs.

Key differences from the chatterbox engine, by design:
  * REFERENCE = ONE clean clip PLUS ITS TRANSCRIPT. CosyVoice's zero-shot
    path aligns the prompt audio with prompt_text (the transcript of what's
    SAID in the reference — i.e. the ORIGINAL-language line, which is why
    the app ingests the original-language SRT). Chatterbox's concatenated
    multi-clip reference is useless here: a concat clip has no coherent
    single transcript. select_reference() picks the best surviving clip in
    the 3-15s sweet spot whose line has original_text.
  * `speed` is a GENERATION-time knob (the model speaks faster/slower), so
    it's full-regen tier — unlike the manual atempo factor, which remains a
    light post-fit. timefit.py still runs on top, unchanged.
  * instruct_text non-empty routes through inference_instruct2 (style by
    natural language); empty = plain inference_zero_shot.

Packaging: the CosyVoice repo is vendored (config.COSYVOICE_REPO), not pip-
installed; _ensure_paths() puts the repo + its Matcha-TTS submodule on
sys.path at first use. All heavy imports are lazy — this module imports
cleanly on machines without CosyVoice (tests fake synthesize_line wholesale,
same pattern as chatterbox).
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import config
from ..gpu_service import gpu
from ..models.project_state import ProjectState
from . import reference_clips, timefit


class CosyVoiceError(RuntimeError):
    pass


import contextlib


@contextlib.contextmanager
def _force_cpu_visibility():
    """CosyVoice3's constructor has NO device parameter (confirmed against
    upstream source: `__init__(self, model_dir, load_jit=False,
    load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1)`) — it
    always auto-detects via torch.cuda.is_available() internally and uses
    CUDA if that's True, with no override. The app's force_cpu flag can't
    reach that decision through any argument, so this scopes a monkeypatch
    of torch.cuda.is_available() around JUST the construction call (the
    exact signal CosyVoice's own code branches on) and restores it
    immediately after — reversible, and doesn't affect CUDA-enabled loads
    for other projects in the same process."""
    import torch
    orig = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = orig


def _wrap_prompt_text(prompt_text: str, instruct_text: str = "") -> str:
    """CosyVoice3 requires a literal <|endofprompt|> marker in prompt_text
    (see config.COSYVOICE_ZERO_SHOT_PREFIX for why). The text before it is
    a style-instruction slot; use instruct_text there when set even on the
    plain zero-shot path (inference_instruct2 is reserved for cases where
    no reference transcript is available at all -- see synthesize_line)."""
    if "<|endofprompt|>" in prompt_text:
        return prompt_text
    prefix = (instruct_text or "").strip() or config.COSYVOICE_ZERO_SHOT_PREFIX
    return f"{prefix}<|endofprompt|>{prompt_text}"


def _wrap_instruct_text(instruct_text: str) -> str:
    """CosyVoice3's inference_instruct2 ALSO requires the literal
    <|endofprompt|> marker (same underlying LLM-side assertion as
    zero-shot's prompt_text -- see _wrap_prompt_text), but in a different
    shape: the marker TERMINATES the instruction with nothing after it,
    per the model card's own instruct2 example (a dialect-conversion
    instruction):
        'You are a helpful assistant.<actual instruction><|endofprompt|>'
    Contrast with zero-shot, where content comes AFTER the marker (the
    reference transcript) -- instruct2 has no separate transcript slot, so
    there's nothing to put after it."""
    if "<|endofprompt|>" in instruct_text:
        return instruct_text
    return f"{config.COSYVOICE_ZERO_SHOT_PREFIX}{instruct_text}<|endofprompt|>"


def _ensure_paths() -> None:
    repo = config.COSYVOICE_REPO
    if not (repo / "cosyvoice").is_dir():
        raise CosyVoiceError(
            f"CosyVoice repo not found at {repo} — "
            "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "
            "there (or set COSYVOICE_REPO), then pip install -r its "
            "requirements.txt into this venv")
    for p in (repo, repo / "third_party" / "Matcha-TTS"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def load_model(precision: str, device: str = "cuda"):
    """Construct the CosyVoice model (called by gpu_service.get_tts inside
    the resident-engine bookkeeping — never call this directly from the dub
    loop). `device` is passed in explicitly (not re-derived from gpu.device())
    so the CUDA-disable checkbox's force_cpu override is respected exactly
    once, at the call site in get_tts(). Returns the model object; sample
    rate is model.sample_rate."""
    _ensure_paths()
    if not config.COSYVOICE_MODEL_DIR.is_dir():
        raise CosyVoiceError(
            f"CosyVoice model dir not found: {config.COSYVOICE_MODEL_DIR} — "
            "snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', "
            f"local_dir='{config.COSYVOICE_MODEL_DIR}')")
    if device == "cpu":
        print("CosyVoice3 running on CPU (CUDA disabled or unavailable) — "
              "usable for single-line testing, slow for full runs")
    fp16 = precision != "fp32" and device == "cuda"
    last_err = None
    # AutoModel is the CosyVoice3-era entry point; CosyVoice2 covers older
    # checkouts. fp16 kwarg first, plain fallback if the signature lacks it.
    for kwargs in ([{"fp16": True}] if fp16 else []) + [{}]:
        # fresh context manager EACH iteration — contextlib.contextmanager
        # objects are single-use generators, can't be re-entered
        cpu_ctx = _force_cpu_visibility() if device == "cpu" else contextlib.nullcontext()
        try:
            with cpu_ctx:
                try:
                    from cosyvoice.cli.cosyvoice import AutoModel
                    return AutoModel(model_dir=str(config.COSYVOICE_MODEL_DIR),
                                     **kwargs)
                except ImportError:
                    from cosyvoice.cli.cosyvoice import CosyVoice2
                    return CosyVoice2(str(config.COSYVOICE_MODEL_DIR),
                                      load_jit=False, load_trt=False, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    raise CosyVoiceError(f"could not construct CosyVoice model: {last_err}")


def cosy_ref_path(project_dir: Path, spk: str) -> Path:
    """The concatenated multi-pin reference audio, built by renew_reference()
    -- distinct from Chatterbox's refs/ref_<spk>.wav (different format/use,
    same refs/ directory)."""
    return project_dir / "refs" / f"ref_cosy_{spk}.wav"


def cosy_ref_text_path(project_dir: Path, spk: str) -> Path:
    """The transcript paired with cosy_ref_path — CosyVoice's zero-shot
    prompt_text needs to correspond to what's actually in the reference
    audio; this is the concatenation (in the same order/cap) of each
    committed clip's original_text."""
    return project_dir / "refs" / f"ref_cosy_{spk}.txt"


def auto_pick_line(ps: ProjectState, project_dir: Path, spk: str) -> int | None:
    """Best single surviving clip for zero-shot prompting when nothing has
    been pinned+renewed: not pruned, its line has original_text, duration
    preferably inside [REF_MIN_S, REF_MAX_S] (closest to the sweet spot
    wins; longest as tiebreak). Pure file/metadata math — safe for a UI
    indicator, not just the synthesis fallback path."""
    clips = reference_clips.list_clips(ps, project_dir, spk)
    texts = {l["line_no"]: l.get("original_text") for l in ps.data["lines"]}
    usable = []
    for c in clips:
        ot = (texts.get(c["line_no"]) or "").strip()
        d = c["duration_s"] or 0.0
        if not ot or d <= 0:
            continue
        lo, hi = config.COSYVOICE_REF_MIN_S, config.COSYVOICE_REF_MAX_S
        penalty = 0.0 if lo <= d <= hi else min(abs(d - lo), abs(d - hi))
        usable.append((penalty, -d, c["line_no"]))
    if not usable:
        return None
    usable.sort()
    return usable[0][2]


def renew_reference(ps: ProjectState, project_dir: Path, spk: str) -> dict:
    """(Re)build spk's concatenated CosyVoice reference from its currently
    STAGED pins (speakers[spk]["ref_pins"]). Pure file/audio work, no GPU —
    safe to call from a fast job. Pins whose clip no longer exists, or whose
    line has lost its original_text, are skipped (not a hard failure) and
    reported back. An empty (or entirely-stale) pin set clears any existing
    concatenated reference, reverting that speaker to auto_pick_line() at
    synthesis time. Returns {"mode": "pinned"|"auto", "committed": [...],
    "skipped": [...]} — the router uses "committed" to call
    project_state.mark_speaker_reference_renewed for the badge cascade."""
    from pydub import AudioSegment

    pins = sorted(ps.data["speakers"].get(spk, {}).get("ref_pins", []))
    wav_out, txt_out = cosy_ref_path(project_dir, spk), cosy_ref_text_path(project_dir, spk)

    committed, skipped = [], []
    if pins:
        texts = {l["line_no"]: l.get("original_text") for l in ps.data["lines"]}
        combined = AudioSegment.silent(duration=0)
        parts_text = []
        for n in pins:
            p = reference_clips.clip_path(project_dir, spk, n)
            ot = (texts.get(n) or "").strip()
            if not p.exists() or not ot:
                skipped.append(n)
                continue
            if len(combined) >= config.COSYVOICE_PINNED_REF_MAX_S * 1000:
                skipped.append(n)   # cap reached: not included, not an error
                continue
            combined += AudioSegment.from_wav(p) + AudioSegment.silent(duration=200)
            parts_text.append(ot)
            committed.append(n)

    if committed:
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        combined.export(wav_out, format="wav")
        txt_out.write_text(" ".join(parts_text), encoding="utf-8")
        return {"mode": "pinned", "committed": committed, "skipped": skipped}

    # nothing usable was committed (no pins, or every pin was stale/capped):
    # clear any previously-built concatenation so synthesis falls back to
    # auto_pick_line() below
    wav_out.unlink(missing_ok=True)
    txt_out.unlink(missing_ok=True)
    return {"mode": "auto", "committed": [], "skipped": skipped}


def select_reference(ps: ProjectState, project_dir: Path, spk: str
                     ) -> tuple[Path, str]:
    """Resolve this speaker's effective reference for synthesis: the
    committed multi-pin concatenation if one has been renewed (cheap file
    existence check — the actual build happened earlier, in
    renew_reference), else the single best surviving clip
    (auto_pick_line). Raises with an actionable message when nothing is
    usable either way."""
    wav, txt = cosy_ref_path(project_dir, spk), cosy_ref_text_path(project_dir, spk)
    if wav.exists() and txt.exists():
        return wav, txt.read_text(encoding="utf-8")

    line_no = auto_pick_line(ps, project_dir, spk)
    if line_no is None:
        raise CosyVoiceError(
            f"speaker '{spk}' has no usable CosyVoice reference — needs at "
            "least one extracted clip whose line has original_text (upload "
            "the original-language SRT in the Upload Center)")
    texts = {l["line_no"]: l.get("original_text") for l in ps.data["lines"]}
    return reference_clips.clip_path(project_dir, spk, line_no), texts[line_no]


def synthesize_line(*, text: str, prompt_text: str, ref_path: Path,
                    speed: float, instruct_text: str, out_path: Path,
                    precision: str = config.PRECISION,
                    force_cpu: bool = False) -> float:
    """One generation to dub_work/raw/. Returns the generated duration in
    seconds. RuntimeError('...out of memory...') propagates for the dub
    job's empty_cache + single-retry, same contract as chatterbox."""
    import torch
    import torchaudio

    model = gpu.get_tts("cosyvoice3", precision, force_cpu)
    _ensure_paths()
    # PASS THE RAW PATH, not a pre-loaded tensor: this checkout's
    # frontend_zero_shot reloads the prompt internally via its own
    # load_wav(prompt_wav, 24000) call on whatever object it's given --
    # a pre-loaded tensor makes that crash (`TypeError: Invalid file:
    # tensor(...)` inside torchaudio/soundfile). Confirmed against
    # Fun-CosyVoice3-0.5B-2512 via cosyvoice_probe.py; if a future
    # checkout changes this back, that probe script's auto-detection
    # will catch it again before it reaches here.
    prompt_arg = str(ref_path)

    instruct = (instruct_text or "").strip()
    # zero-shot needs the marker mid-string (content follows); instruct2
    # needs it as a suffix (nothing follows) -- confirmed against the model
    # card's own examples for both, see _wrap_prompt_text / _wrap_instruct_text
    wrapped_prompt = _wrap_prompt_text(prompt_text, instruct)
    wrapped_instruct = _wrap_instruct_text(instruct) if instruct else ""
    with torch.inference_mode():
        try:
            if instruct:
                gen = model.inference_instruct2(
                    text, wrapped_instruct, prompt_arg, stream=False, speed=speed)
            else:
                gen = model.inference_zero_shot(
                    text, wrapped_prompt, prompt_arg, stream=False, speed=speed)
        except TypeError:
            # older signatures without the speed kwarg
            if instruct:
                gen = model.inference_instruct2(
                    text, wrapped_instruct, prompt_arg, stream=False)
            else:
                gen = model.inference_zero_shot(
                    text, wrapped_prompt, prompt_arg, stream=False)
        chunks = [j["tts_speech"] for j in gen]
    if not chunks:
        raise CosyVoiceError("CosyVoice returned no audio")
    speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), speech.float().cpu(), model.sample_rate)
    return timefit.probe_duration(out_path)
