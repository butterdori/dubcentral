"""Project-creation defaults and infrastructure config.

Values here seed *project defaults* when a new project is created; after
that, the runtime source of truth is each project's project.json (see
models/project_state.py).
"""
import os
from pathlib import Path

# ---- storage ----------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent
STORAGE_ROOT = BACKEND_ROOT / "storage"          # per-project working dirs
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"

# ---- Hugging Face cache ------------------------------------------------------
# App-local model cache so Chatterbox weights download ONCE and persist across
# service restarts/users, instead of hitting the Hub every load. setdefault:
# an explicitly exported HF_HOME still wins. NOTE: if you see 401/unauthorized
# from the Hub for these public models, the usual culprit is a stale/invalid
# HF_TOKEN in the environment — `unset HF_TOKEN` (and HUGGING_FACE_HUB_TOKEN)
# or fix it with `huggingface-cli login`.
HF_CACHE = BACKEND_ROOT / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))

CSV_NAME = "segments.csv"        # import/export interchange only, never runtime state
PROJECT_JSON = "project.json"

# ---- dub defaults (seed values for a new project's project_defaults) --------
DEFAULT_DUB_LANGUAGE = "ko"      # Chatterbox language code for the dub
SLOT_TOLERANCE = 1.15            # Auto mode: allowed overrun (and 1/x underrun) before stretching
DEFAULT_FIT_MODE = "natural"     # auto | natural | manual (natural: play at generated length)
DEFAULT_MANUAL_FACTOR = 1.0      # Manual mode: >1 speeds up, <1 slows down
MANUAL_FACTOR_RECOMMENDED = (0.7, 1.5)  # UI guidance only — outside values accepted, not blocked

CHATTERBOX_EXAGGERATION = 0.5    # emotion intensity (0=flat .. ~1=dramatic)
CHATTERBOX_CFG_WEIGHT = 0.5      # reference-voice adherence
CHATTERBOX_MAX_REF_S = 20        # cap for the concatenated per-speaker reference

# ---- CosyVoice3 engine (Phase 5) ------------------------------------------
# The CosyVoice repo is NOT pip-installable: git clone --recursive it into
# backend/vendor/CosyVoice (or point COSYVOICE_REPO elsewhere). Model weights
# go under the HF cache:
#   python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
#     local_dir='backend/hf_cache/Fun-CosyVoice3-0.5B')"
COSYVOICE_REPO = Path(os.environ.get(
    "COSYVOICE_REPO", BACKEND_ROOT / "vendor" / "CosyVoice"))
COSYVOICE_MODEL_DIR = Path(os.environ.get(
    "COSYVOICE_MODEL_DIR", HF_CACHE / "Fun-CosyVoice3-0.5B"))
COSYVOICE_SPEED = 1.0            # native generation-speed knob (1.0 neutral)
COSYVOICE_INSTRUCT_TEXT = ""     # free-text style instruction; "" = plain zero-shot
# CosyVoice3's zero-shot path REQUIRES a literal "<|endofprompt|>" marker
# somewhere in prompt_text -- its LLM stage asserts on that special token
# (inside a background thread; the failure otherwise surfaces as an
# unrelated-looking crash a few lines later, not the real assertion). The
# text before the marker is a style-instruction slot, not inert boilerplate
# -- this is the neutral default when the line has no instruct_text set.
COSYVOICE_ZERO_SHOT_PREFIX = "You are a helpful assistant."
COSYVOICE_REF_MIN_S = 3.0        # zero-shot prompt clip sweet spot
COSYVOICE_REF_MAX_S = 15.0
COSYVOICE_PINNED_REF_MAX_S = 20.0  # cap for the concatenated multi-pin reference

# ---- CrispASR engine ---------------------------------------------
# CrispASR runs as a SEPARATE, always-running server process (ggml/C++),
# not something this app loads/unloads in-process. It exposes an OpenAI-
# compatible /v1/audio/speech endpoint. Set these to wherever you're
# actually running it:
#   CRISPASR_URL      - the normal (GPU) instance
#   CRISPASR_CPU_URL   - a SEPARATE instance you've started CPU-only (e.g.
#                        with the relevant CRISPASR_<BACKEND>_*_CPU=1 env
#                        vars at ITS startup). Defaults to CRISPASR_URL --
#                        meaning the Force CPU checkbox is a NO-OP for this
#                        engine unless you actually stand up a second
#                        instance. CrispASR's device selection is fixed at
#                        ITS process startup (env vars), not a per-request
#                        HTTP parameter, so there is no way to force CPU
#                        for a single call against a GPU-configured instance.
# IMPORTANT (VRAM): this app's "never load Chatterbox/CosyVoice3 in-process
# while Demucs runs" discipline (gpu_service.exclusive()) has NO visibility
# into CrispASR's own GPU usage -- it's a separate process. If CrispASR's
# model is resident on the same physical GPU while Demucs runs here, they
# can contend for VRAM with no coordination. Worth knowing on a 4GB card.
CRISPASR_URL = os.environ.get("CRISPASR_URL", "http://127.0.0.1:8090")
CRISPASR_CPU_URL = os.environ.get("CRISPASR_CPU_URL", CRISPASR_URL)
CRISPASR_DEFAULT_BACKEND = "cosyvoice3-tts"   # confirmed via `crispasr --list-backends`
                                              # (NOT "cosyvoice3" -- corrected after
                                              # verifying against a real build; the
                                              # web docs disagreed with each other)
CRISPASR_TIMEOUT_S = 120

CRISPASR_CONSENT_ATTESTATION = (
    "Operator-attested: reference audio is used with the speaker's consent, "
    "for personal non-commercial dubbing."
)
CRISPASR_SPOKEN_DISCLAIMER = False


# ---- audio / extraction constants (unchanged from the CLI) -------------------
MAX_ATEMPO = 2.0                 # ffmpeg atempo hard limit per filter pass
DEMUCS_MODEL = "htdemucs"
TRIM_SILENCE_DB = -40            # dBFS threshold for tightening clip boundaries
MIN_EDGE_PAD_MS = 100            # never leave less than this much real audio per side
NO_TRIM_UNDER_MS = 1000          # slots shorter than this are extracted raw, untrimmed
MIN_REF_MS = 300                 # reference clips shorter than this are rejected
MIN_CLIP_MS = 700                # warn (not skip) when an extracted ref clip is shorter

# ---- GPU --------------------------------------------------------------------

PRECISION = "fp16"
