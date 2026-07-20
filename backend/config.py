"""Ported CONFIG block from dub_pipeline.py, trimmed to the Chatterbox-only webapp.

Values here seed *project defaults* when a new project is created; after that,
the runtime source of truth is each project's project.json (see
models/project_state.py). XTTS / easy_xtts_trainer knobs from the CLI are gone.
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

# ---- audio / extraction constants (unchanged from the CLI) -------------------
MAX_ATEMPO = 2.0                 # ffmpeg atempo hard limit per filter pass
DEMUCS_MODEL = "htdemucs"
TRIM_SILENCE_DB = -40            # dBFS threshold for tightening clip boundaries
MIN_EDGE_PAD_MS = 100            # never leave less than this much real audio per side
NO_TRIM_UNDER_MS = 1000          # slots shorter than this are extracted raw, untrimmed
MIN_REF_MS = 300                 # reference clips shorter than this are rejected
MIN_CLIP_MS = 700                # warn (not skip) when an extracted ref clip is shorter

# ---- GPU --------------------------------------------------------------------
# fp16 is the default here (the CLI defaulted to fp32): the webapp's assumed
# path is the 4GB GTX 1650, where fp16 weight conversion + autocast is required
# for Chatterbox to fit. CPU fallback with a visible warning if CUDA is absent.
PRECISION = "fp16"               # fp32 | fp16 | bf16 (bf16 falls back to fp16 pre-Ampere)
