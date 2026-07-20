"""project.json schema + atomic writes. The single owner of:

  * the inheritance cascade — project defaults -> speaker defaults -> line
    overrides — via one resolve(); API responses carry effective value +
    override source per field, and the frontend only renders (the cascade is
    never reimplemented in JS)
  * the light/full BADGE-TRANSITION TABLE: which edited field triggers which
    regen tier, and how each edit moves a line's badge. Routers call
    apply_line_edits / set_speaker_default / set_project_default and get badge
    deltas back — the table is not smeared across routers.
  * badge lifecycle for job runs (mark_queued / mark_generating /
    mark_success / mark_failed), used by the Phase-2/3 job worker
  * the single-level undo slot metadata (the displaced raw+fit audio files
    themselves live in dub_work/undo/ — this module only tracks which lines
    the last dub action touched and what badge/take state they had before)

Everything in this module is pure state: no GPU, no audio, no FastAPI.
project.json is the runtime source of truth; segments.csv is import/export
interchange only.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


# ============================== enums / tables ===============================

class Badge(str, Enum):
    NEVER = "never"            # never dubbed
    CLEAN = "clean"            # dubbed, up to date
    NEEDS_LIGHT = "needs_light"  # timing-only change: re-stretch/re-place raw, no TTS
    NEEDS_FULL = "needs_full"    # text/speaker/language/knob change: new TTS required
    GENERATING = "generating"  # in progress
    FAILED = "failed"          # last synthesis attempt errored (message in line.error)


class FitMode(str, Enum):
    AUTO = "auto"        # compress if > slot*tol, extend if < slot/tol
    NATURAL = "natural"  # play at generated length; may overlap (mixed, never cut)
    MANUAL = "manual"    # direct factor (>1 speed up, <1 slow down)


# Synthesis engines (per-project choice). Engine-SPECIFIC inheritable
# fields are namespaced here: the cascade mechanism is one and the same,
# but only the active engine's fields are surfaced/used, and switching
# engines clears the now-inactive engine's overrides + speaker defaults.
ENGINES = ("chatterbox", "cosyvoice3")
ENGINE_FIELDS = {
    "chatterbox": ("exaggeration", "cfg_weight"),
    "cosyvoice3": ("speed", "instruct_text"),
}

# Fields that flow through the inheritance cascade
# (project defaults -> speaker defaults -> line overrides):
INHERITABLE_FIELDS = (
    "dub_language",
    "exaggeration",     # chatterbox
    "cfg_weight",       # chatterbox
    "speed",            # cosyvoice3: native generation-speed knob
    "instruct_text",    # cosyvoice3: free-text style instruction ("" = plain zero-shot)
    "tolerance",
    "fit_mode",
    "manual_factor",
)

# Line-local fields stored directly on the line (never inherited):
# original_text = the ORIGINAL-language line (cosyvoice3 uses the reference
# clip's original_text as the zero-shot prompt transcript)
LINE_LOCAL_FIELDS = ("speaker", "start", "end", "text", "original_text")

# ---- the badge-transition table (single source of truth) --------------------
# Which regen tier an edit to each field demands:
#   full  = the synthesized audio itself is stale -> new TTS generation
#   light = only stretching/placement is stale -> refit from untouched raw/
FIELD_REGEN_TIER: dict[str, str] = {
    # full: different text / voice / language / synthesis knobs
    "text": "full",
    "speaker": "full",         # different reference voice = resynthesize
    "dub_language": "full",
    "exaggeration": "full",
    "cfg_weight": "full",
    "speed": "full",           # cosyvoice generates AT this speed (not post atempo)
    "instruct_text": "full",
    # engine-conditional: full under cosyvoice3 (it feeds the reference
    # prompt), NO badge effect under chatterbox — special-cased in
    # apply_line_edits, the one deliberate exception to the static table
    "original_text": "full",
    # light: placement/stretch only — raw take stays valid
    "start": "light",
    "end": "light",
    "tolerance": "light",      # only affects re-stretch threshold
    "fit_mode": "light",
    "manual_factor": "light",
}

# (current badge, edit tier) -> new badge.
# GENERATING is intentionally absent: edits to a line mid-generation are
# rejected by apply_line_edits (routers surface that as a 409-style error).
EDIT_BADGE_TRANSITION: dict[tuple[Badge, str], Badge] = {
    # never dubbed: edits change what WILL be generated, badge stays "never"
    (Badge.NEVER, "light"): Badge.NEVER,
    (Badge.NEVER, "full"): Badge.NEVER,
    # clean take: degrade by tier
    (Badge.CLEAN, "light"): Badge.NEEDS_LIGHT,
    (Badge.CLEAN, "full"): Badge.NEEDS_FULL,
    # already needs light: full supersedes, light accumulates
    (Badge.NEEDS_LIGHT, "light"): Badge.NEEDS_LIGHT,
    (Badge.NEEDS_LIGHT, "full"): Badge.NEEDS_FULL,
    # already needs full: nothing downgrades it
    (Badge.NEEDS_FULL, "light"): Badge.NEEDS_FULL,
    (Badge.NEEDS_FULL, "full"): Badge.NEEDS_FULL,
    # failed: there is no usable take to refit, so any edit leaves the line
    # failed — the retry (re-running Dub All/Selected on it) is a full TTS
    # attempt regardless of what was edited
    (Badge.FAILED, "light"): Badge.FAILED,
    (Badge.FAILED, "full"): Badge.FAILED,
}

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9\-_]")


def sanitize_speaker(name: str) -> str:
    """Same rule as the CLI: filesystem-safe speaker key."""
    key = _SANITIZE_RE.sub("_", name.strip())
    return key or "UNKNOWN"


# ================================ exceptions =================================

class StateError(Exception):
    """Invalid state operation (unknown line/speaker/field, edit while generating)."""


# ================================ ProjectState ===============================

SCHEMA_VERSION = 1


class ProjectState:
    """In-memory handle on one project.json. Load, mutate through the methods
    below (which enforce the badge table + cascade), then save() atomically.
    """

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self.data = data
        self.path = path

    # ---------------------------- construction -------------------------------

    @classmethod
    def new(cls, name: str, project_defaults: dict[str, Any] | None = None,
            path: Path | None = None) -> "ProjectState":
        from .. import config  # late import: keep module importable standalone
        defaults = {
            "dub_language": config.DEFAULT_DUB_LANGUAGE,
            "exaggeration": config.CHATTERBOX_EXAGGERATION,
            "cfg_weight": config.CHATTERBOX_CFG_WEIGHT,
            "speed": config.COSYVOICE_SPEED,
            "instruct_text": config.COSYVOICE_INSTRUCT_TEXT,
            "tolerance": config.SLOT_TOLERANCE,
            "fit_mode": config.DEFAULT_FIT_MODE,
            "manual_factor": config.DEFAULT_MANUAL_FACTOR,
        }
        if project_defaults:
            unknown = set(project_defaults) - set(INHERITABLE_FIELDS)
            if unknown:
                raise StateError(f"unknown project default fields: {sorted(unknown)}")
            defaults.update(project_defaults)
        now = time.time()
        return cls({
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "created_at": now,
            "modified_at": now,
            "video_filename": None,
            "srt_filename": None,
            # remix option: mute the original audio for the full subtitle
            # timestamp, not just the dub's own (post-fit) duration. Off by
            # default (matches the original CLI/webapp behavior); needed when
            # a shorter dub would otherwise let the tail of the original
            # speech bleed through — see set_timestamp_mute.
            "timestamp_mute": True,
            "cuda_enabled": False,  # troubleshooting override: Force CPU is ON by default
            "engine": "cosyvoice3",
            "project_defaults": defaults,
            # speakers: key -> {"display_name": str, "defaults": {field: value}}
            # defaults holds ONLY explicitly-set fields (absence == inherit project)
            "speakers": {},
            "lines": [],
            # single-level undo slot: metadata for the lines displaced by the
            # LAST dub action only (audio pairs live in dub_work/undo/)
            "undo": None,
        }, path=path)

    def init_lines_from_segments(self, rows: Iterable[dict[str, Any]]) -> None:
        """(Re)build the line table from prep/CSV rows
        ({line_no, speaker, start, end, text}). Full reset: wipes overrides,
        badges, and the undo slot — matches CSV re-upload semantics."""
        lines = []
        for r in rows:
            spk = (r.get("speaker") or "").strip()
            key = sanitize_speaker(spk) if spk else None
            if key and key not in self.data["speakers"]:
                self.data["speakers"][key] = {"display_name": spk, "defaults": {}}
            lines.append({
                "line_no": int(r["line_no"]),
                "speaker": key,                    # None => skipped at dub time
                "start": float(r["start"]),
                "end": float(r["end"]),
                "text": (r.get("text") or "").strip(),
                "overrides": {},                   # field -> explicit value
                "badge": Badge.NEVER.value,
                "error": None,
                "raw_duration_s": None,            # generated length (set after synthesis)
                "fit_duration_s": None,            # placed length after time-fit
                "original_text": None,             # original-language line (P5)
            })
        lines.sort(key=lambda l: l["line_no"])
        self.data["lines"] = lines
        self.data["undo"] = None
        self._touch()

    # ------------------------------ accessors --------------------------------

    def line(self, line_no: int) -> dict[str, Any]:
        for l in self.data["lines"]:
            if l["line_no"] == line_no:
                return l
        raise StateError(f"no such line: {line_no}")

    def speaker(self, key: str) -> dict[str, Any]:
        try:
            return self.data["speakers"][key]
        except KeyError:
            raise StateError(f"no such speaker: {key!r}") from None

    def add_speaker(self, display_name: str) -> str:
        key = sanitize_speaker(display_name)
        if key in self.data["speakers"]:
            raise StateError(f"speaker already exists: {key!r}")
        self.data["speakers"][key] = {"display_name": display_name, "defaults": {}}
        self._touch()
        return key

    # ------------------------------- resolve ---------------------------------

    def resolve(self, line: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """THE cascade. For each inheritable field return
        {"value": effective, "source": "line" | "speaker" | "project"}.
        Line overrides beat speaker defaults beat project defaults; a speaker
        default exists only if explicitly set (absence is not None-valued)."""
        spk_defaults = {}
        if line["speaker"]:
            spk_defaults = self.data["speakers"].get(
                line["speaker"], {}).get("defaults", {})
        out = {}
        for f in INHERITABLE_FIELDS:
            if f in line["overrides"]:
                out[f] = {"value": line["overrides"][f], "source": "line"}
            elif f in spk_defaults:
                out[f] = {"value": spk_defaults[f], "source": "speaker"}
            else:
                out[f] = {"value": self.data["project_defaults"][f],
                          "source": "project"}
        return out

    def grid(self) -> list[dict[str, Any]]:
        """Dub Center rows: line-local fields + per-field effective value and
        override source. This is the payload shape the frontend renders as-is."""
        rows = []
        for l in self.data["lines"]:
            rows.append({
                "line_no": l["line_no"],
                "speaker": l["speaker"],
                "start": l["start"],
                "end": l["end"],
                "text": l["text"],
                "original_text": l.get("original_text"),
                "badge": l["badge"],
                "error": l["error"],
                "raw_duration_s": l["raw_duration_s"],
                "fit_duration_s": l.get("fit_duration_s"),
                "slot_s": max(0.0, l["end"] - l["start"]),
                "fields": self.resolve(l),
            })
        return rows

    # ------------------------------- editing ---------------------------------
    # All mutators return badge deltas: {line_no: new_badge_value} for every
    # line whose badge changed — exactly what the polling/edit endpoints relay.

    def apply_line_edits(self, edits: list[dict[str, Any]]) -> dict[int, str]:
        """Apply a LIST of line edits (the Phase-3 endpoints are list-accepting
        from day one; bulk UI later is pure fan-out onto this).

        Each edit: {"line_no": int, "field": str, "value": Any}.
          * line-local fields (speaker/start/end/text) are set directly
          * inheritable fields are written to line.overrides
          * value=None on an inheritable field CLEARS the override
            (back to inherited)
        Validates everything first — an invalid edit anywhere rejects the whole
        batch, so a bulk apply never lands half-way.
        """
        # -- validate pass --
        staged: list[tuple[dict, str, Any]] = []
        for e in edits:
            line = self.line(int(e["line_no"]))
            field, value = e["field"], e.get("value")
            if field not in FIELD_REGEN_TIER:
                raise StateError(f"unknown editable field: {field!r}")
            if line["badge"] == Badge.GENERATING.value:
                raise StateError(
                    f"line {line['line_no']} is generating — edits are blocked "
                    "until the run finishes")
            if field == "speaker":
                if value is not None and value not in self.data["speakers"]:
                    raise StateError(f"no such speaker: {value!r}")
            elif field in ("start", "end"):
                value = float(value)
            elif field == "fit_mode" and value is not None:
                FitMode(value)  # raises ValueError on garbage
            elif field in ("exaggeration", "cfg_weight", "tolerance",
                           "manual_factor", "speed") and value is not None:
                value = float(value)
            elif field == "instruct_text" and value is not None:
                value = str(value)
            elif field == "original_text" and value is not None:
                value = str(value).strip() or None
            staged.append((line, field, value))

        # -- apply pass --
        deltas: dict[int, str] = {}
        for line, field, value in staged:
            if field in LINE_LOCAL_FIELDS:
                if line[field] == value:
                    continue  # no-op edit: no badge change
                line[field] = value
                changed = True
            else:
                changed = self._set_override(line, field, value)
            if changed:
                if field == "original_text" and \
                        self.data.get("engine") != "cosyvoice3":
                    pass   # irrelevant to chatterbox synthesis: no badge flip
                else:
                    self._degrade(line, FIELD_REGEN_TIER[field], deltas)
        if deltas or staged:
            self._touch()
        return deltas

    def set_speaker_default(self, key: str, field: str, value: Any) -> dict[int, str]:
        """Set (or clear, with value=None) a speaker default. Cascades: every
        line of that speaker WITHOUT its own override on `field` flips by the
        field's tier."""
        if field not in INHERITABLE_FIELDS:
            raise StateError(f"not an inheritable field: {field!r}")
        spk = self.speaker(key)
        old_effective = self._speaker_effective(key, field)
        if value is None:
            spk["defaults"].pop(field, None)
        else:
            spk["defaults"][field] = value
        deltas: dict[int, str] = {}
        if self._speaker_effective(key, field) != old_effective:
            for line in self.data["lines"]:
                if line["speaker"] == key and field not in line["overrides"]:
                    self._degrade(line, FIELD_REGEN_TIER[field], deltas)
        self._touch()
        return deltas

    def set_engine(self, engine: str) -> dict[str, Any]:
        """Per-project engine switch. Different engine => different audio, so
        every line that HAS a take flips to needs_full (never/failed/
        generating are untouched — generating is rejected upstream by the
        router's busy check). The now-INACTIVE engine's line overrides and
        speaker defaults are cleared so the new engine's grid starts clean;
        shared fields (timing/text/speaker/lang/tolerance/fit/factor) and
        original_text are untouched."""
        if engine not in ENGINES:
            raise StateError(f"unknown engine: {engine!r} (choose from {ENGINES})")
        old = self.data.get("engine", "chatterbox")
        if engine == old:
            return {"deltas": {}, "cleared_line_overrides": 0,
                    "cleared_speaker_defaults": 0}
        old_fields = ENGINE_FIELDS[old]
        cleared_ovr = cleared_spk = 0
        for line in self.data["lines"]:
            for f in old_fields:
                if line["overrides"].pop(f, None) is not None:
                    cleared_ovr += 1
        for spk in self.data["speakers"].values():
            for f in old_fields:
                if spk["defaults"].pop(f, None) is not None:
                    cleared_spk += 1
        deltas: dict[int, str] = {}
        with_take = {Badge.CLEAN.value, Badge.NEEDS_LIGHT.value,
                     Badge.NEEDS_FULL.value}
        for line in self.data["lines"]:
            if line["badge"] in with_take and \
                    line["badge"] != Badge.NEEDS_FULL.value:
                line["badge"] = Badge.NEEDS_FULL.value
                deltas[line["line_no"]] = line["badge"]
        self.data["engine"] = engine
        self._touch()
        return {"deltas": deltas, "cleared_line_overrides": cleared_ovr,
                "cleared_speaker_defaults": cleared_spk}

    def set_timestamp_mute(self, value: bool) -> None:
        self.data["timestamp_mute"] = bool(value)
        self._touch()

    def set_cuda_enabled(self, value: bool) -> None:
        """Troubleshooting override, read at the start of each dub run only:
        False forces TTS generation onto CPU regardless of CUDA
        availability. Demucs (extraction, and remix's implicit separation)
        always uses GPU when available — this checkbox does not touch it.
        Does not affect anything mid-run."""
        self.data["cuda_enabled"] = bool(value)
        self._touch()

    def set_speaker_ref_pins(self, key: str, line_nos: list[int]) -> None:
        """CosyVoice reference-clip pins (plural): the set of clips staged
        to be concatenated into this speaker's reference on the NEXT
        "Renew reference" action. Purely a staging change — no badge
        cascade here. The actually-used reference (and therefore whether
        anything is stale) only changes when renew_speaker_reference()
        commits a new concatenation; see that method for the cascade."""
        self.speaker(key)   # raises StateError if unknown
        for n in line_nos:
            if not any(l["line_no"] == n for l in self.data["lines"]):
                raise StateError(f"no such line: {n}")
        self.speaker(key)["ref_pins"] = sorted(set(line_nos))
        self._touch()

    def mark_speaker_reference_renewed(self, key: str,
                                       committed_line_nos: list[int]
                                       ) -> dict[int, str]:
        """Called by the router right after successfully (re)building this
        speaker's concatenated CosyVoice reference file. Records which
        lines actually made it into the build (may be a subset of the
        staged pins, if some had gone stale — e.g. pruned since staged),
        and, when engine == cosyvoice3, flags that speaker's already-
        dubbed lines as stale (same consequence class as a speaker default
        change) since the actual reference audio just changed."""
        spk = self.speaker(key)
        spk["ref_pins_committed"] = sorted(committed_line_nos)
        deltas: dict[int, str] = {}
        if self.data.get("engine") == "cosyvoice3":
            flippable = {Badge.CLEAN.value, Badge.NEEDS_LIGHT.value}
            for line in self.data["lines"]:
                if line["speaker"] == key and line["badge"] in flippable:
                    line["badge"] = Badge.NEEDS_FULL.value
                    deltas[line["line_no"]] = line["badge"]
        self._touch()
        return deltas

    def rename_speaker(self, key: str, display_name: str) -> None:
        """Display-name only — the key (and clip folders, refs) are untouched,
        so no regen is triggered."""
        self.speaker(key)["display_name"] = display_name
        self._touch()

    def set_project_default(self, field: str, value: Any) -> dict[int, str]:
        """Set a project default. Cascades to every line that inherits it all
        the way down (no line override AND no speaker default on that field)."""
        if field not in INHERITABLE_FIELDS:
            raise StateError(f"not an inheritable field: {field!r}")
        if value is None:
            raise StateError("project defaults cannot be cleared, only changed")
        if self.data["project_defaults"][field] == value:
            return {}
        self.data["project_defaults"][field] = value
        deltas: dict[int, str] = {}
        for line in self.data["lines"]:
            if field in line["overrides"]:
                continue
            spk = line["speaker"]
            if spk and field in self.data["speakers"].get(spk, {}).get("defaults", {}):
                continue
            self._degrade(line, FIELD_REGEN_TIER[field], deltas)
        self._touch()
        return deltas

    # --------------------------- job-run lifecycle ---------------------------
    # Used by the Phase-2/3 worker. Kept here so every badge write goes through
    # this module.

    def mark_generating(self, line_nos: Iterable[int]) -> dict[int, str]:
        deltas = {}
        for n in line_nos:
            line = self.line(n)
            line["badge"] = Badge.GENERATING.value
            line["error"] = None
            deltas[n] = line["badge"]
        self._touch()
        return deltas

    def mark_success(self, line_no: int, raw_duration_s: float | None = None,
                     fit_duration_s: float | None = None) -> dict[int, str]:
        line = self.line(line_no)
        line["badge"] = Badge.CLEAN.value
        line["error"] = None
        if raw_duration_s is not None:
            line["raw_duration_s"] = float(raw_duration_s)
        if fit_duration_s is not None:
            line["fit_duration_s"] = float(fit_duration_s)
        self._touch()
        return {line_no: line["badge"]}

    def mark_failed(self, line_no: int, error: str) -> dict[int, str]:
        line = self.line(line_no)
        line["badge"] = Badge.FAILED.value
        line["error"] = str(error)
        self._touch()
        return {line_no: line["badge"]}

    # -------------------------------- undo -----------------------------------

    def record_undo(self, entries: dict[int, dict[str, Any]]) -> None:
        """Called by the dub worker right before a run overwrites takes.
        `entries`: line_no -> {"had_take": bool, "prev_badge": str,
        "prev_raw_duration_s": float|None}. Single-level: replaces whatever the
        slot held (dub_work/undo/ is likewise cleared by the worker)."""
        self.data["undo"] = {
            "timestamp": time.time(),
            "lines": {str(n): meta for n, meta in entries.items()},
        }
        self._touch()

    def consume_undo(self) -> dict[int, dict[str, Any]]:
        """Pop the undo slot and restore each touched line's badge metadata.
        (The worker swaps the audio files back from dub_work/undo/; lines with
        had_take=False revert to never-dubbed.) Returns the entries so the
        caller knows which files to swap."""
        slot = self.data.get("undo")
        if not slot:
            raise StateError("nothing to undo")
        entries = {int(n): meta for n, meta in slot["lines"].items()}
        for n, meta in entries.items():
            line = self.line(n)
            if meta.get("had_take"):
                line["badge"] = meta.get("prev_badge", Badge.CLEAN.value)
                line["raw_duration_s"] = meta.get("prev_raw_duration_s")
                line["fit_duration_s"] = meta.get("prev_fit_duration_s")
            else:
                line["badge"] = Badge.NEVER.value
                line["raw_duration_s"] = None
                line["fit_duration_s"] = None
            line["error"] = None
        self.data["undo"] = None
        self._touch()
        return entries

    # ------------------------------ persistence ------------------------------

    def save(self, path: Path | None = None) -> None:
        """Atomic write: temp file in the same directory + os.replace."""
        path = path or self.path
        if path is None:
            raise StateError("no path set for project.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".project_json.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "ProjectState":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("schema_version")
        if v != SCHEMA_VERSION:
            raise StateError(f"unsupported project.json schema_version: {v!r}")
        data.setdefault("timestamp_mute", False)   # projects saved pre-4.1
        data.setdefault("engine", "chatterbox")    # pre-P5
        data.setdefault("cuda_enabled", True)      # pre-P5.1
        from .. import config
        data["project_defaults"].setdefault("speed", config.COSYVOICE_SPEED)
        data["project_defaults"].setdefault(
            "instruct_text", config.COSYVOICE_INSTRUCT_TEXT)
        for l in data.get("lines", []):
            l.setdefault("original_text", None)
        return cls(data, path=path)

    # ------------------------------- internals -------------------------------

    def _speaker_effective(self, key: str, field: str) -> Any:
        """What a non-overridden line of this speaker currently inherits."""
        d = self.data["speakers"][key]["defaults"]
        return d[field] if field in d else self.data["project_defaults"][field]

    def _set_override(self, line: dict, field: str, value: Any) -> bool:
        """Write/clear a line override. Returns True iff the EFFECTIVE value
        changed (setting an override equal to the inherited value, or clearing
        one that matched it, is a no-op for regen purposes — but the override
        flag itself is still recorded/cleared)."""
        before = self.resolve(line)[field]["value"]
        if value is None:
            line["overrides"].pop(field, None)
        else:
            line["overrides"][field] = value
        after = self.resolve(line)[field]["value"]
        return before != after

    def _degrade(self, line: dict, tier: str, deltas: dict[int, str]) -> None:
        cur = Badge(line["badge"])
        new = EDIT_BADGE_TRANSITION[(cur, tier)]
        if new != cur:
            line["badge"] = new.value
            deltas[line["line_no"]] = new.value

    def _touch(self) -> None:
        self.data["modified_at"] = time.time()
