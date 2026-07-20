"""Pydantic models for the API layer.

Only the shapes that are stable from Phase 0 onward live here now: the dub-grid
row (effective value + override source per field — the frontend renders these
verbatim and never recomputes the cascade) and the list-accepting line-edit
request. Router-specific request/response models arrive with their routers.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

InheritSource = Literal["line", "speaker", "project"]
BadgeName = Literal["never", "clean", "needs_light", "needs_full",
                    "generating", "failed"]


class EffectiveField(BaseModel):
    value: Any
    source: InheritSource   # "line" == explicitly overridden on this line


class GridRow(BaseModel):
    line_no: int
    speaker: str | None
    start: float
    end: float
    text: str
    badge: BadgeName
    error: str | None = None
    raw_duration_s: float | None = None   # generated length
    fit_duration_s: float | None = None   # placed length after time-fit — the
                                          # ratio column prefers this
    slot_s: float
    # dub_language / exaggeration / cfg_weight / tolerance / fit_mode /
    # manual_factor — each with its effective value + where it came from
    fields: dict[str, EffectiveField]


class LineEdit(BaseModel):
    line_no: int
    field: str
    value: Any = None       # None clears an inheritable override (re-inherit)


class LineEditRequest(BaseModel):
    """List-accepting from day one — Phase-5 bulk controls are pure fan-out."""
    edits: list[LineEdit] = Field(min_length=1)


class BadgeDeltas(BaseModel):
    """Returned by every mutating call and by GET /jobs/current."""
    deltas: dict[int, BadgeName]
