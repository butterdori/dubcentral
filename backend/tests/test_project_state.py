"""Phase 0 tests: the state model is the risky piece — pin its behavior down
before any GPU or audio code exists.

Covers: cascade precedence + override sources, the light/full badge-transition
table (per-field tiers, full-supersedes-light, failed-stays-failed,
never-stays-never), speaker/project default cascades skipping overridden
lines, no-op edits, generating-blocks-edits, batch atomicity, the single-level
undo slot, and atomic JSON round-trips.
"""
import json

import pytest

from backend.models.project_state import (
    Badge, ProjectState, StateError, FIELD_REGEN_TIER,
)

SEGS = [
    {"line_no": 1, "speaker": "Alice", "start": 0.0, "end": 2.0, "text": "hi"},
    {"line_no": 2, "speaker": "Alice", "start": 2.5, "end": 4.0, "text": "yo"},
    {"line_no": 3, "speaker": "Bob",   "start": 5.0, "end": 8.0, "text": "hm"},
    {"line_no": 4, "speaker": "",      "start": 9.0, "end": 9.5, "text": "(music)"},
]


@pytest.fixture
def ps(tmp_path):
    ps = ProjectState.new("t", path=tmp_path / "project.json")
    ps.init_lines_from_segments(SEGS)
    return ps


def all_clean(ps, nos=(1, 2, 3)):
    for n in nos:
        ps.mark_generating([n])
        ps.mark_success(n, raw_duration_s=1.0)


# ------------------------------- cascade -------------------------------------

def test_resolve_project_default(ps):
    r = ps.resolve(ps.line(1))
    assert r["exaggeration"] == {"value": 0.5, "source": "project"}
    assert r["fit_mode"]["source"] == "project"


def test_resolve_speaker_beats_project(ps):
    ps.set_speaker_default("Alice", "exaggeration", 0.8)
    assert ps.resolve(ps.line(1))["exaggeration"] == {"value": 0.8, "source": "speaker"}
    # Bob untouched
    assert ps.resolve(ps.line(3))["exaggeration"]["source"] == "project"


def test_resolve_line_beats_speaker(ps):
    ps.set_speaker_default("Alice", "exaggeration", 0.8)
    ps.apply_line_edits([{"line_no": 1, "field": "exaggeration", "value": 0.3}])
    assert ps.resolve(ps.line(1))["exaggeration"] == {"value": 0.3, "source": "line"}
    assert ps.resolve(ps.line(2))["exaggeration"] == {"value": 0.8, "source": "speaker"}


def test_clearing_override_reinherits(ps):
    ps.apply_line_edits([{"line_no": 1, "field": "tolerance", "value": 1.5}])
    assert ps.resolve(ps.line(1))["tolerance"]["source"] == "line"
    ps.apply_line_edits([{"line_no": 1, "field": "tolerance", "value": None}])
    assert ps.resolve(ps.line(1))["tolerance"] == {"value": 1.15, "source": "project"}


def test_override_flag_distinct_even_when_value_matches_inherited(ps):
    """'inherited' vs 'explicitly set to the same value' must be distinguishable."""
    ps.apply_line_edits([{"line_no": 1, "field": "exaggeration", "value": 0.5}])
    assert ps.resolve(ps.line(1))["exaggeration"]["source"] == "line"


# --------------------------- badge tier table --------------------------------

@pytest.mark.parametrize("field,value,expected", [
    ("text", "new words", Badge.NEEDS_FULL),
    ("speaker", "Bob", Badge.NEEDS_FULL),
    ("dub_language", "ja", Badge.NEEDS_FULL),
    ("exaggeration", 0.9, Badge.NEEDS_FULL),
    ("cfg_weight", 0.2, Badge.NEEDS_FULL),
    ("start", 0.5, Badge.NEEDS_LIGHT),
    ("end", 2.5, Badge.NEEDS_LIGHT),
    ("tolerance", 1.3, Badge.NEEDS_LIGHT),
    ("fit_mode", "manual", Badge.NEEDS_LIGHT),   # non-default value
    ("manual_factor", 1.2, Badge.NEEDS_LIGHT),
])
def test_edit_tier_from_clean(ps, field, value, expected):
    all_clean(ps)
    deltas = ps.apply_line_edits([{"line_no": 1, "field": field, "value": value}])
    assert deltas == {1: expected.value}


def test_full_supersedes_light_and_nothing_downgrades(ps):
    all_clean(ps)
    ps.apply_line_edits([{"line_no": 1, "field": "start", "value": 0.2}])
    assert ps.line(1)["badge"] == Badge.NEEDS_LIGHT.value
    ps.apply_line_edits([{"line_no": 1, "field": "text", "value": "x"}])
    assert ps.line(1)["badge"] == Badge.NEEDS_FULL.value
    # a later light edit must NOT downgrade full -> light
    ps.apply_line_edits([{"line_no": 1, "field": "end", "value": 2.2}])
    assert ps.line(1)["badge"] == Badge.NEEDS_FULL.value


def test_never_dubbed_stays_never_on_edits(ps):
    deltas = ps.apply_line_edits([{"line_no": 1, "field": "text", "value": "x"}])
    assert deltas == {}
    assert ps.line(1)["badge"] == Badge.NEVER.value


def test_failed_stays_failed_on_edits(ps):
    ps.mark_generating([1])
    ps.mark_failed(1, "CUDA OOM")
    ps.apply_line_edits([{"line_no": 1, "field": "start", "value": 0.1}])
    assert ps.line(1)["badge"] == Badge.FAILED.value
    assert ps.line(1)["error"] == "CUDA OOM"


def test_edit_blocked_while_generating(ps):
    ps.mark_generating([1])
    with pytest.raises(StateError, match="generating"):
        ps.apply_line_edits([{"line_no": 1, "field": "text", "value": "x"}])


def test_noop_edits_do_not_degrade(ps):
    all_clean(ps)
    # same text
    assert ps.apply_line_edits(
        [{"line_no": 1, "field": "text", "value": "hi"}]) == {}
    # override set equal to inherited value: flag recorded, but no regen
    assert ps.apply_line_edits(
        [{"line_no": 1, "field": "tolerance", "value": 1.15}]) == {}
    assert ps.resolve(ps.line(1))["tolerance"]["source"] == "line"
    assert ps.line(1)["badge"] == Badge.CLEAN.value


def test_batch_is_atomic_on_validation_error(ps):
    all_clean(ps)
    with pytest.raises(StateError):
        ps.apply_line_edits([
            {"line_no": 1, "field": "text", "value": "changed"},
            {"line_no": 2, "field": "nonsense_field", "value": 1},
        ])
    # first edit must not have landed
    assert ps.line(1)["text"] == "hi"
    assert ps.line(1)["badge"] == Badge.CLEAN.value


def test_every_editable_field_has_a_tier():
    assert set(FIELD_REGEN_TIER.values()) == {"light", "full"}


# ------------------------- default cascades ----------------------------------

def test_speaker_default_flips_only_non_overridden_lines(ps):
    all_clean(ps)
    ps.apply_line_edits([{"line_no": 2, "field": "cfg_weight", "value": 0.9}])
    assert ps.line(2)["badge"] == Badge.NEEDS_FULL.value  # from the override itself
    ps.mark_generating([2]); ps.mark_success(2, 1.0)      # regen it -> clean again
    deltas = ps.set_speaker_default("Alice", "cfg_weight", 0.1)
    # line 1 inherits -> flips full; line 2 overridden -> untouched; Bob untouched
    assert deltas == {1: Badge.NEEDS_FULL.value}
    assert ps.line(2)["badge"] == Badge.CLEAN.value
    assert ps.line(3)["badge"] == Badge.CLEAN.value


def test_speaker_default_light_field_flips_light(ps):
    all_clean(ps)
    deltas = ps.set_speaker_default("Alice", "fit_mode", "auto")  # non-default
    assert deltas == {1: Badge.NEEDS_LIGHT.value, 2: Badge.NEEDS_LIGHT.value}
    # setting the PROJECT default value on a speaker is a no-op for badges
    assert ps.set_speaker_default("Bob", "fit_mode", "natural") == {}


def test_project_default_skips_speaker_covered_and_overridden_lines(ps):
    all_clean(ps)
    ps.set_speaker_default("Alice", "dub_language", "ja")   # covers lines 1,2
    for n in (1, 2):
        ps.mark_generating([n]); ps.mark_success(n, 1.0)
    deltas = ps.set_project_default("dub_language", "en")
    # only Bob's line 3 inherits from project level
    assert deltas == {3: Badge.NEEDS_FULL.value}


def test_clearing_speaker_default_cascades_if_effective_changes(ps):
    all_clean(ps)
    ps.set_speaker_default("Alice", "exaggeration", 0.9)
    for n in (1, 2):
        ps.mark_generating([n]); ps.mark_success(n, 1.0)
    deltas = ps.set_speaker_default("Alice", "exaggeration", None)  # back to 0.5
    assert deltas == {1: Badge.NEEDS_FULL.value, 2: Badge.NEEDS_FULL.value}


def test_setting_speaker_default_equal_to_project_is_noop_for_badges(ps):
    all_clean(ps)
    assert ps.set_speaker_default("Alice", "exaggeration", 0.5) == {}


# ------------------------------ speakers -------------------------------------

def test_add_speaker_and_reassign(ps):
    key = ps.add_speaker("New Guy!")
    assert key == "New_Guy_"
    all_clean(ps)
    deltas = ps.apply_line_edits([{"line_no": 3, "field": "speaker", "value": key}])
    assert deltas == {3: Badge.NEEDS_FULL.value}


def test_reassign_to_unknown_speaker_rejected(ps):
    with pytest.raises(StateError, match="no such speaker"):
        ps.apply_line_edits([{"line_no": 1, "field": "speaker", "value": "Ghost"}])


def test_rename_speaker_is_display_only_no_regen(ps):
    all_clean(ps)
    ps.rename_speaker("Alice", "Alice (main)")
    assert ps.speaker("Alice")["display_name"] == "Alice (main)"
    assert all(ps.line(n)["badge"] == Badge.CLEAN.value for n in (1, 2))


# -------------------------------- undo ---------------------------------------

def test_undo_restores_prior_state_and_is_single_level(ps):
    all_clean(ps, nos=(1,))
    # a dub run is about to overwrite line 1 (had a take) and dub line 3 (never)
    ps.record_undo({
        1: {"had_take": True, "prev_badge": "clean", "prev_raw_duration_s": 1.0},
        3: {"had_take": False, "prev_badge": "never", "prev_raw_duration_s": None},
    })
    ps.mark_generating([1, 3])
    ps.mark_success(1, 2.0)
    ps.mark_success(3, 3.0)
    entries = ps.consume_undo()
    assert set(entries) == {1, 3}
    assert ps.line(1)["badge"] == Badge.CLEAN.value
    assert ps.line(1)["raw_duration_s"] == 1.0
    assert ps.line(3)["badge"] == Badge.NEVER.value
    assert ps.line(3)["raw_duration_s"] is None
    # single-level: slot is now empty
    with pytest.raises(StateError, match="nothing to undo"):
        ps.consume_undo()


def test_new_undo_replaces_old_slot(ps):
    ps.record_undo({1: {"had_take": False, "prev_badge": "never",
                        "prev_raw_duration_s": None}})
    ps.record_undo({2: {"had_take": False, "prev_badge": "never",
                        "prev_raw_duration_s": None}})
    assert set(ps.consume_undo()) == {2}


# ----------------------------- persistence -----------------------------------

def test_atomic_save_load_roundtrip(ps, tmp_path):
    ps.set_speaker_default("Alice", "exaggeration", 0.8)
    ps.apply_line_edits([{"line_no": 3, "field": "fit_mode", "value": "manual"},
                         {"line_no": 3, "field": "manual_factor", "value": 1.3}])
    ps.save()
    loaded = ProjectState.load(ps.path)
    assert loaded.data == ps.data
    r = loaded.resolve(loaded.line(3))
    assert r["fit_mode"] == {"value": "manual", "source": "line"}
    assert r["manual_factor"] == {"value": 1.3, "source": "line"}
    # no stray temp files left behind
    assert [p.name for p in ps.path.parent.iterdir()] == ["project.json"]


def test_load_rejects_unknown_schema_version(tmp_path):
    p = tmp_path / "project.json"
    p.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(StateError, match="schema_version"):
        ProjectState.load(p)


def test_grid_shape(ps):
    rows = ps.grid()
    assert [r["line_no"] for r in rows] == [1, 2, 3, 4]
    r1 = rows[0]
    assert r1["slot_s"] == 2.0
    assert set(r1["fields"]) == {"dub_language", "exaggeration", "cfg_weight",
                                 "speed", "instruct_text",
                                 "tolerance", "fit_mode", "manual_factor"}
    assert r1["fields"]["dub_language"]["source"] == "project"
    # speakerless line resolves entirely from project defaults, never dubbed
    r4 = rows[3]
    assert r4["speaker"] is None and r4["badge"] == "never"


def test_reinit_lines_resets_everything(ps):
    all_clean(ps)
    ps.record_undo({1: {"had_take": True, "prev_badge": "clean",
                        "prev_raw_duration_s": 1.0}})
    ps.init_lines_from_segments(SEGS)  # CSV re-upload semantics
    assert all(l["badge"] == Badge.NEVER.value for l in ps.data["lines"])
    assert ps.data["undo"] is None
