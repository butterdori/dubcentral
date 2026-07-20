"""Persistent model manager: the automated version of the CLI's manual
sequencing on the 4GB GTX 1650.

Two invariants, enforced here and nowhere else:

  1. ONE GPU OPERATION AT A TIME — the `exclusive()` slot. The job queue is
     already single-slot, so this lock should never be contended; holding it
     anyway means a coding mistake surfaces as a loud RuntimeError instead of
     a silent CUDA OOM.
  2. DEMUCS AND CHATTERBOX ARE NEVER RESIDENT TOGETHER — anything that runs
     Demucs calls `unload_chatterbox()` first. Chatterbox stays warm in VRAM
     across TTS requests otherwise (Phase 3 fills in load_chatterbox with the
     fp16 weight-conversion path ported from the CLI).

torch is imported lazily and optionally: Phase 2's only GPU work (Demucs) runs
as a subprocess with its own torch, so the web service itself works on a
machine without torch installed — device just reports "cpu".
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


class GPUService:
    def __init__(self) -> None:
        self._slot = threading.Lock()
        # one resident TTS engine at a time (P5): "chatterbox" | "cosyvoice3"
        self._tts_engine: str | None = None
        self._tts_model = None
        self._tts_precision: str | None = None
        self._tts_device: str | None = None   # tracked so a force_cpu
                                               # toggle invalidates the cache

    # ------------------------------ the slot ---------------------------------

    @contextmanager
    def exclusive(self, purpose: str):
        """Hold the single GPU slot. Non-blocking acquire: contention means a
        sequencing bug, and we want to hear about it immediately."""
        if not self._slot.acquire(blocking=False):
            raise RuntimeError(
                f"GPU slot already held — refusing to start {purpose!r} "
                "(one GPU operation at a time)")
        try:
            yield
        finally:
            self._slot.release()

    # ----------------------------- chatterbox --------------------------------

    def unload_tts(self) -> None:
        """Free whichever TTS engine is resident (called before any Demucs
        run, and on engine switch)."""
        if self._tts_model is None:
            return
        self._tts_model = None
        self._tts_engine = None
        self._tts_precision = None
        torch = self._torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # back-compat alias (demucs_runner and older code call this name)
    def unload_chatterbox(self) -> None:
        self.unload_tts()

    @property
    def chatterbox_loaded(self) -> bool:
        return self._tts_engine == "chatterbox"

    @property
    def tts_engine_loaded(self) -> str | None:
        return self._tts_engine

    def get_tts(self, engine: str, precision: str, force_cpu: bool = False):
        """One resident engine of N kinds. Same engine+precision+device =>
        the warm model; anything else => unload the current occupant first.
        force_cpu is the Dub Center's CUDA-disable checkbox: True pins this
        load to CPU regardless of CUDA availability (troubleshooting
        override — see device()). Since engine/device choice is per-project,
        a swap only ever happens when the user switches projects or flips
        the checkbox — never mid-run."""
        device = self.device(force_cpu)
        if self._tts_model is not None and self._tts_engine == engine \
                and self._tts_precision == precision \
                and self._tts_device == device:
            return self._tts_model
        self.unload_tts()
        if engine == "chatterbox":
            model = self._load_chatterbox(precision, device)
        elif engine == "cosyvoice3":
            from .pipeline import tts_cosyvoice   # lazy: vendored repo
            model = tts_cosyvoice.load_model(precision, device)
        else:
            raise ValueError(f"unknown TTS engine: {engine!r}")
        self._tts_model = model
        self._tts_engine = engine
        self._tts_precision = precision
        self._tts_device = device
        return model

    def get_chatterbox(self, precision: str, force_cpu: bool = False):
        return self.get_tts("chatterbox", precision, force_cpu)

    def _load_chatterbox(self, precision: str, device: str):
        """Warm-load Chatterbox Multilingual (kept resident in VRAM across
        requests). fp16/bf16 converts the main submodules' WEIGHTS to half —
        autocast alone keeps ~2GB of fp32 weights resident, which doesn't fit
        a 4GB card. Ported from the CLI's cmd_dub. Pure loader — residency
        bookkeeping lives in get_tts(); `device` is passed in rather than
        re-derived so a force_cpu override is respected exactly once, here."""
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if device == "cpu":
            print("Chatterbox running on CPU (CUDA disabled or unavailable) "
                  "— usable for single-line testing, slow for full runs")
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)

        if precision != "fp32" and device == "cuda":
            dtype = torch.float16
            if precision == "bf16":
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    print("WARNING: bf16 needs Ampere or newer — this GPU "
                          "doesn't support it, using fp16 instead")
            converted = []
            for name in ("t3", "s3gen", "ve"):
                sub = getattr(model, name, None)
                if sub is not None and hasattr(sub, "to"):
                    try:
                        sub.to(dtype=dtype)
                        converted.append(name)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  note: couldn't convert {name} to "
                              f"{precision}: {exc}")
            print(f"Chatterbox weights in {precision}: "
                  f"{', '.join(converted) or 'none converted'}")

        return model

    def autocast(self, precision: str, force_cpu: bool = False):
        """Mixed-precision inference context (CLI's autocast_ctx port).
        fp32 or CPU -> no-op; fp16/bf16 -> torch.autocast."""
        import contextlib
        if self.device(force_cpu) != "cuda" or precision == "fp32":
            return contextlib.nullcontext()
        import torch
        dtype = torch.float16
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def empty_cache(self) -> None:
        """Best-effort CUDA cache flush (the OOM-retry path)."""
        torch = self._torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------- device ----------------------------------

    @staticmethod
    def _torch():
        try:
            import torch
            return torch
        except ImportError:
            return None

    def device(self, force_cpu: bool = False) -> str:
        """force_cpu is the per-project CUDA-disable checkbox — a
        troubleshooting override, not a capability check. Independent of
        actual CUDA availability: if force_cpu is True this always returns
        "cpu", even on a machine with a perfectly healthy GPU."""
        if force_cpu:
            return "cpu"
        torch = self._torch()
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def device_info(self, force_cpu: bool = False) -> dict:
        """For /api/health and UI warnings (CPU fallback is visible, never
        silent — same pattern as the CLI)."""
        if force_cpu:
            return {"device": "cpu", "note": "CUDA disabled (project override)",
                    "tts_engine_loaded": self._tts_engine}
        torch = self._torch()
        if torch is None:
            return {"device": "cpu", "torch": None,
                    "note": "torch not installed in the web service venv"}
        if not torch.cuda.is_available():
            return {"device": "cpu", "torch": torch.__version__,
                    "note": "CUDA not available — GPU jobs will run on CPU "
                            "(slow, but visible as a warning, not a failure)"}
        props = torch.cuda.get_device_properties(0)
        return {
            "device": "cuda",
            "torch": torch.__version__,
            "gpu": props.name,
            "vram_total_mb": props.total_memory // (1 << 20),
            "tts_engine_loaded": self._tts_engine,
        }


# process-wide singleton
gpu = GPUService()
