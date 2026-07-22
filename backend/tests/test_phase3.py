"""Phase 3 tests. TTS is faked (synthesize_line writes real wavs via ffmpeg
with test-controlled durations) so timefit, badges, undo file-swaps, and the
job loop all run for real. Demucs faked as in Phase 2 so references are built
from real extracted clips.

Covers: timefit auto both directions + chaining + manual/natural; the dub run
end-to-end (badge lifecycle via polled deltas, raw/fit files, ratio stats);
light refit not touching raw; OOM empty_cache+retry; per-line failure with
the run continuing; no-reference-clips failure; single-level undo restoring
prior takes and never-dubbed state; run/undo guards; SRT export + revert.
"""
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from backend.job_queue import jobs as job_queue

SRT = """\
1
00:00:00,200 --> 00:00:02,200
Hello there.

2
00:00:02,600 --> 00:00:04,600
General Kenobi!

3
00:00:04,800 --> 00:00:05,300
Hmm.
"""

CSV = ("speaker,line_no,start,end,text\n"
       "Obi Wan,1,0.200,2.200,Hello there.\n"
       "Grievous,2,2.600,4.600,General Kenobi!\n"
       "Obi Wan,3,4.800,5.300,Hmm.\n")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "STORAGE_ROOT", tmp_path / "storage")
    job_queue.wait_idle()
    job_queue.reset_for_tests()
    from backend.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=6:size=160x120:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
         "-af", "volume='if(lt(mod(t,1.0),0.6),1.0,0.0)':eval=frame",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


@pytest.fixture()
def fake_demucs(monkeypatch):
    from backend.pipeline import demucs_runner

    def fake(video, sep_root, force_cpu=False):
        from backend import config
        stem = sep_root / config.DEMUCS_MODEL / "source"
        stem.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn",
                        str(stem / "vocals.wav")], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(stem / "vocals.wav"),
                        "-af", "volume=0", str(stem / "no_vocals.wav")],
                       check=True, capture_output=True)
    monkeypatch.setattr(demucs_runner, "_run_cli", fake)


@pytest.fixture()
def fake_tts(monkeypatch):
    """Fake synthesize_line: writes a sine wav of a test-controlled duration.
    cfg.durations[line_no] sets the generated length (default 1.0s);
    cfg.fail / cfg.oom_once trigger the failure paths."""
    from backend.pipeline import tts_chatterbox

    class Cfg:
        durations: dict[int, float] = {}
        fail: set[int] = set()
        oom_once: set[int] = set()
        calls: list[int] = []
    cfg = Cfg()

    def synth(*, text, language, ref_path, exaggeration, cfg_weight,
              out_path, precision="fp16", force_cpu=False):
        n = int(out_path.stem.split("_")[1])
        cfg.calls.append(n)
        assert ref_path.exists(), "reference must be built before synthesis"
        if n in cfg.oom_once:
            cfg.oom_once.discard(n)
            raise RuntimeError("CUDA error: out of memory")
        if n in cfg.fail:
            raise RuntimeError("synthetic tts failure")
        dur = cfg.durations.get(n, 1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        f"sine=frequency=440:duration={dur}",
                        str(out_path)], check=True, capture_output=True)
        return dur

    monkeypatch.setattr(tts_chatterbox, "synthesize_line", synth)
    return cfg


def make_ready(client, video):
    """Project with media, diarized CSV, and extracted clips. Phase 3 tests
    exercise the chatterbox pipeline specifically -- pin the engine
    explicitly rather than relying on the app's default (which is
    cosyvoice3 as of Phase 5.3)."""
    key = client.post("/api/projects", json={"name": "P3"}).json()["key"]
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    client.put(f"/api/projects/{key}/csv", content=CSV,
               headers={"content-type": "text/csv"})
    client.put(f"/api/projects/{key}/engine", json={"engine": "chatterbox"})
    client.post(f"/api/projects/{key}/extract_clips")
    assert job_queue.wait_idle(30)
    # the app default fit is 'natural'; these e2e tests exercise auto fitting
    r = client.put(f"/api/projects/{key}/defaults",
                   json={"field": "fit_mode", "value": "auto"})
    assert r.status_code == 200, r.text
    return key


def run_dub(client, key, mode="all", line_nos=None):
    r = client.post(f"/api/projects/{key}/dub/run",
                    json={"mode": mode, "line_nos": line_nos or []})
    assert r.status_code == 202, r.text
    assert job_queue.wait_idle(60)
    snap = client.get("/api/jobs/current").json()["job"]
    assert snap["status"] == "done", snap
    return snap


def grid(client, key):
    return client.get(f"/api/projects/{key}/dub/grid").json()


# -------------------------------- timefit -------------------------------------

def _wav(tmp_path, dur, name="in.wav"):
    p = tmp_path / name
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={dur}", str(p)],
                   check=True, capture_output=True)
    return p


def _fields(mode="auto", tol=1.15, factor=1.0):
    return {"fit_mode": {"value": mode, "source": "project"},
            "tolerance": {"value": tol, "source": "project"},
            "manual_factor": {"value": factor, "source": "project"}}


def test_auto_compress(tmp_path):
    from backend.pipeline import timefit
    out = tmp_path / "out.wav"
    res = timefit.fit_line(_wav(tmp_path, 3.0), out, 2.0, _fields())
    assert res["factor"] == pytest.approx(1.5, abs=0.01)
    assert res["fit_duration_s"] == pytest.approx(2.0, abs=0.15)


def test_auto_extend(tmp_path):
    from backend.pipeline import timefit
    out = tmp_path / "out.wav"
    res = timefit.fit_line(_wav(tmp_path, 1.0), out, 2.0, _fields())
    assert res["factor"] == pytest.approx(0.5, abs=0.01)
    assert res["fit_duration_s"] == pytest.approx(2.0, abs=0.15)


def test_auto_within_tolerance_untouched(tmp_path):
    from backend.pipeline import timefit
    raw = _wav(tmp_path, 2.1)
    out = tmp_path / "out.wav"
    res = timefit.fit_line(raw, out, 2.0, _fields())     # 2.1 < 2.0*1.15
    assert res["factor"] == 1.0
    assert out.read_bytes() == raw.read_bytes()          # exact copy


def test_manual_and_natural(tmp_path):
    from backend.pipeline import timefit
    raw = _wav(tmp_path, 2.0)
    res = timefit.fit_line(raw, tmp_path / "m.wav", 5.0,
                           _fields(mode="manual", factor=2.0))
    assert res["fit_duration_s"] == pytest.approx(1.0, abs=0.12)
    res = timefit.fit_line(raw, tmp_path / "n.wav", 0.5, _fields(mode="natural"))
    assert res["factor"] == 1.0 and res["fit_duration_s"] == pytest.approx(2.0, abs=0.1)


def test_atempo_chaining_extremes(tmp_path):
    from backend.pipeline import timefit
    assert len(timefit.atempo_filters(5.0)) == 3      # 2 * 2 * 1.25
    assert len(timefit.atempo_filters(0.3)) == 2      # 0.5 * 0.6
    raw = _wav(tmp_path, 4.0)
    res = timefit.fit_line(raw, tmp_path / "x.wav", 1.0, _fields())
    assert res["factor"] == pytest.approx(4.0, abs=0.02)
    assert res["hard_stretch"] is True
    assert res["fit_duration_s"] == pytest.approx(1.0, abs=0.15)


# ----------------------------- dub run e2e ------------------------------------

def test_dub_all_end_to_end(client, video, fake_demucs, fake_tts):
    from backend import store
    from backend.pipeline import dub_work, timefit
    key = make_ready(client, video)
    fake_tts.durations = {1: 3.0, 2: 2.0, 3: 0.5}   # slots: 2.0, 2.0, 0.5

    job = run_dub(client, key)
    assert job["result"]["n_ok"] == 3 and job["result"]["n_failed"] == 0
    # deltas streamed the full lifecycle; final state clean
    assert set(job["deltas"].items()) >= {("1", "clean"), ("2", "clean"),
                                          ("3", "clean")}
    g = grid(client, key)
    assert all(r["badge"] == "clean" for r in g["rows"])
    assert g["has_undo"] is True
    d = store.project_dir(key)
    for n in (1, 2, 3):
        assert dub_work.raw_path(d, n).exists()
        assert dub_work.fit_path(d, n).exists()
    # line 1: 3.0s into a 2.0 slot -> compressed fit
    assert timefit.probe_duration(dub_work.fit_path(d, 1)) == pytest.approx(2.0, abs=0.2)
    # raw_duration recorded for the ratio column
    assert grid(client, key)["rows"][0]["raw_duration_s"] == pytest.approx(3.0, abs=0.05)
    # ratio stat: (3+2+0.5)/(2+2+0.5)
    assert job["result"]["raw_vs_slot_ratio"] == pytest.approx(5.5 / 4.5, abs=0.01)
    # audio endpoint serves the fit take
    r = client.get(f"/api/projects/{key}/dub/lines/1/audio")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    # dub all again with everything clean -> 400
    assert client.post(f"/api/projects/{key}/dub/run",
                       json={"mode": "all"}).status_code == 400


def test_light_refit_never_touches_raw(client, video, fake_demucs, fake_tts):
    from backend import store
    from backend.pipeline import dub_work, timefit
    key = make_ready(client, video)
    fake_tts.durations = {1: 2.0}
    run_dub(client, key)
    n_tts_calls = len(fake_tts.calls)
    d = store.project_dir(key)
    raw_bytes = dub_work.raw_path(d, 1).read_bytes()

    # widen line 1's slot -> needs_light
    r = client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "end", "value": 4.2}]})
    assert r.json()["deltas"] == {"1": "needs_light"}

    job = run_dub(client, key)                    # only line 1 needs work
    assert job["result"]["n_ok"] == 1
    assert len(fake_tts.calls) == n_tts_calls     # NO new TTS generation
    assert dub_work.raw_path(d, 1).read_bytes() == raw_bytes  # raw untouched
    # 2.0s into a 4.0 slot at tol 1.15 -> extended to ~4.0
    assert timefit.probe_duration(dub_work.fit_path(d, 1)) == pytest.approx(4.0, abs=0.3)
    # the badge returns to the clean checkmark after light regen — in the
    # final state AND in the deltas the polling UI receives
    assert job["deltas"]["1"] == "clean"
    row = grid(client, key)["rows"][0]
    assert row["badge"] == "clean"
    # ratio column data: placed (fit) duration recorded alongside raw
    assert row["fit_duration_s"] == pytest.approx(4.0, abs=0.3)
    assert row["raw_duration_s"] == pytest.approx(2.0, abs=0.05)


def test_full_regen_after_text_edit(client, video, fake_demucs, fake_tts):
    key = make_ready(client, video)
    fake_tts.durations = {2: 1.0}
    run_dub(client, key)
    n = len(fake_tts.calls)
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 2, "field": "text", "value": "New line reading"}]})
    assert grid(client, key)["rows"][1]["badge"] == "needs_full"
    run_dub(client, key)
    assert fake_tts.calls[n:] == [2]              # exactly one regeneration


def test_oom_retry_then_success(client, video, fake_demucs, fake_tts):
    key = make_ready(client, video)
    fake_tts.oom_once = {2}
    job = run_dub(client, key)
    assert job["result"]["n_ok"] == 3 and job["result"]["n_failed"] == 0
    assert fake_tts.calls.count(2) == 2           # first OOM, then retry


def test_per_line_failure_run_continues(client, video, fake_demucs, fake_tts):
    key = make_ready(client, video)
    fake_tts.fail = {2}
    job = run_dub(client, key)
    assert job["result"]["n_ok"] == 2 and job["result"]["n_failed"] == 1
    assert "synthetic tts failure" in job["result"]["errors"]["2"]
    rows = {r["line_no"]: r for r in grid(client, key)["rows"]}
    assert rows[2]["badge"] == "failed"
    assert "synthetic tts failure" in rows[2]["error"]
    assert rows[1]["badge"] == rows[3]["badge"] == "clean"
    # failed line is retried by the next Dub All
    fake_tts.fail = set()
    job = run_dub(client, key)
    assert job["result"]["n_ok"] == 1
    assert grid(client, key)["rows"][1]["badge"] == "clean"


def test_speaker_without_clips_fails_its_lines(client, video, fake_demucs, fake_tts):
    key = make_ready(client, video)
    client.post(f"/api/projects/{key}/speakers", json={"name": "Ghost"})
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 3, "field": "speaker", "value": "Ghost"}]})
    job = run_dub(client, key, mode="selected", line_nos=[3])
    assert job["result"]["n_failed"] == 1
    row3 = grid(client, key)["rows"][2]
    assert row3["badge"] == "failed"
    assert "no usable reference clips" in row3["error"]


def test_undo_restores_previous_takes(client, video, fake_demucs, fake_tts):
    from backend import store
    from backend.pipeline import dub_work, timefit
    key = make_ready(client, video)
    d = store.project_dir(key)

    # run 1: only lines 1+2 (line 3 stays never-dubbed)
    fake_tts.durations = {1: 1.0, 2: 1.0}
    run_dub(client, key, mode="selected", line_nos=[1, 2])
    # run 2: re-roll 1 + first-time dub 3, with different durations
    fake_tts.durations = {1: 1.8, 3: 0.5}
    run_dub(client, key, mode="selected", line_nos=[1, 3])
    assert timefit.probe_duration(dub_work.raw_path(d, 1)) == pytest.approx(1.8, abs=0.05)
    assert dub_work.raw_path(d, 3).exists()

    r = client.post(f"/api/projects/{key}/dub/undo")
    assert r.status_code == 200
    out = r.json()
    assert out["restored"] == [1, 3]
    # line 1 back to the run-1 take; line 3 back to never-dubbed
    assert timefit.probe_duration(dub_work.raw_path(d, 1)) == pytest.approx(1.0, abs=0.05)
    assert not dub_work.raw_path(d, 3).exists()
    rows = {x["line_no"]: x for x in grid(client, key)["rows"]}
    assert rows[1]["badge"] == "clean"
    assert rows[1]["raw_duration_s"] == pytest.approx(1.0, abs=0.05)
    assert rows[3]["badge"] == "never"
    # single-level: a second undo has nothing
    assert client.post(f"/api/projects/{key}/dub/undo").status_code == 400


def test_run_guards(client, video, fake_demucs, fake_tts):
    import threading
    key = make_ready(client, video)
    # selected with a non-dubbable line
    r = client.post(f"/api/projects/{key}/dub/run",
                    json={"mode": "selected", "line_nos": [999]})
    assert r.status_code == 400
    assert client.post(f"/api/projects/{key}/dub/run",
                       json={"mode": "selected", "line_nos": []}).status_code == 400
    # busy -> run and undo both 409
    release = threading.Event()
    job_queue.submit("hold", key, lambda job: release.wait(5) and None)
    assert client.post(f"/api/projects/{key}/dub/run",
                       json={"mode": "all"}).status_code == 409
    assert client.post(f"/api/projects/{key}/dub/undo").status_code == 409
    release.set()
    job_queue.wait_idle()


def test_edit_blocked_while_generating_is_409(client, video, fake_demucs):
    from backend import store
    key = make_ready(client, video)
    with store.locked(key):
        ps = store.load(key)
        ps.mark_generating([1])
        ps.save()
    r = client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "text", "value": "x"}]})
    assert r.status_code == 409


# ---------------------------- SRT export/revert -------------------------------

def test_srt_export_reflects_edits(client, video, fake_demucs):
    key = make_ready(client, video)
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 2, "field": "text", "value": "Edited reading"}]})
    srt_text = client.get(f"/api/projects/{key}/srt/export").text
    assert "Edited reading" in srt_text and "General Kenobi" not in srt_text
    assert "00:00:02,600 --> 00:00:04,600" in srt_text


def test_srt_revert_full_reset(client, video, fake_demucs, fake_tts):
    from backend import store
    key = make_ready(client, video)
    run_dub(client, key)
    r = client.post(f"/api/projects/{key}/srt/revert")
    assert r.status_code == 200 and r.json()["n_lines"] == 3
    g = grid(client, key)
    assert all(row["badge"] == "never" for row in g["rows"])
    assert g["speakers"] == {}                      # diarization gone
    assert not (store.project_dir(key) / "dub_work").exists()
    # no SRT uploaded -> revert is a 400
    key2 = client.post("/api/projects", json={"name": "bare3"}).json()["key"]
    assert client.post(f"/api/projects/{key2}/srt/revert").status_code == 400


def test_dub_changed_mode(client, video, fake_demucs, fake_tts):
    """'Dub Changed' regenerates only stale takes; tier filter narrows it;
    never-dubbed/failed lines are left for Dub All."""
    key = make_ready(client, video)
    run_dub(client, key)                                   # all clean
    # nothing changed yet
    r = client.post(f"/api/projects/{key}/dub/run", json={"mode": "changed"})
    assert r.status_code == 400
    # line 1 light (timing), line 2 full (text); line 3 stays clean
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "end", "value": 2.4},
        {"line_no": 2, "field": "text", "value": "Changed reading"}]})
    # tier filter: light only -> just line 1
    r = client.post(f"/api/projects/{key}/dub/run",
                    json={"mode": "changed", "tier": "light"})
    assert r.status_code == 202 and r.json()["n_targets"] == 1
    assert job_queue.wait_idle(60)
    rows = {x["line_no"]: x["badge"] for x in grid(client, key)["rows"]}
    assert rows == {1: "clean", 2: "needs_full", 3: "clean"}
    # both tiers -> the remaining full line
    r = client.post(f"/api/projects/{key}/dub/run", json={"mode": "changed"})
    assert r.json()["n_targets"] == 1
    assert job_queue.wait_idle(60)
    assert grid(client, key)["rows"][1]["badge"] == "clean"


def test_manual_fac_updates_placed_ratio(client, video, fake_demucs, fake_tts):
    """fac in Manual mode changes fit_duration_s, so the ratio column
    (placed/slot) reflects it after the light regen."""
    key = make_ready(client, video)
    fake_tts.durations = {1: 2.0}
    run_dub(client, key)
    row = grid(client, key)["rows"][0]
    assert row["fit_duration_s"] == pytest.approx(2.0, abs=0.15)  # slot 2.0, in tol
    # switch line 1 to manual x2 speed -> light regen -> placed 1.0s
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "fit_mode", "value": "manual"},
        {"line_no": 1, "field": "manual_factor", "value": 2.0}]})
    run_dub(client, key, mode="changed")
    row = grid(client, key)["rows"][0]
    assert row["raw_duration_s"] == pytest.approx(2.0, abs=0.05)  # raw unchanged
    assert row["fit_duration_s"] == pytest.approx(1.0, abs=0.12)  # placed halved


def test_project_defaults_endpoint_cascades(client, video, fake_demucs, fake_tts):
    key = make_ready(client, video)
    run_dub(client, key)
    r = client.put(f"/api/projects/{key}/defaults",
                   json={"field": "exaggeration", "value": 0.9})
    assert r.status_code == 200
    out = r.json()
    assert out["project_defaults"]["exaggeration"] == 0.9
    assert set(out["deltas"].values()) == {"needs_full"}
    assert client.put(f"/api/projects/{key}/defaults",
                      json={"field": "nope", "value": 1}).status_code == 400


# ------------------------- dub-selected queueing (P6.3) -------------------------

def test_dub_selected_appends_to_running_run(client, video, fake_demucs, fake_tts):
    """A second 'Dub Selected' submitted while a dub is running appends its
    lines to the current run instead of 409'ing. Uses a gate inside the
    fake synth so the first run is provably still in-flight when the second
    submits."""
    import threading
    from backend.pipeline import tts_chatterbox

    key = make_ready(client, video)   # lines 1,2,3
    gate = threading.Event()
    seen_line1 = threading.Event()
    orig = tts_chatterbox.synthesize_line

    def gated(*, out_path, **kw):
        n = int(out_path.stem.split("_")[1])
        if n == 1:
            seen_line1.set()
            gate.wait(10)   # hold line 1 until the test releases it
        return orig(out_path=out_path, **kw)
    tts_chatterbox.synthesize_line = gated
    try:
        r1 = client.post(f"/api/projects/{key}/dub/run",
                         json={"mode": "selected", "line_nos": [1]})
        assert r1.status_code == 202
        assert seen_line1.wait(10)   # run is now parked inside line 1

        # second Dub Selected while the first is mid-flight -> appended, not 409
        r2 = client.post(f"/api/projects/{key}/dub/run",
                         json={"mode": "selected", "line_nos": [2, 3]})
        assert r2.status_code == 202, r2.text
        assert r2.json().get("appended") is True
        assert r2.json()["job"]["id"] == r1.json()["job"]["id"]  # same run

        gate.set()   # let it finish
        assert job_queue.wait_idle(30)
    finally:
        tts_chatterbox.synthesize_line = orig
        gate.set()

    snap = client.get("/api/jobs/current").json()["job"]
    assert snap["status"] == "done", snap
    assert snap["result"]["n_ok"] == 3   # all three lines dubbed in ONE run
    # all three clean now
    assert all(row["badge"] == "clean" for row in grid(client, key)["rows"])


def test_dub_selected_append_dedupes(client, video, fake_demucs, fake_tts):
    """Appending a line already in the running plan is a no-op (no double
    synthesis of the same line)."""
    import threading
    from backend.pipeline import tts_chatterbox

    key = make_ready(client, video)
    gate = threading.Event()
    seen = threading.Event()
    orig = tts_chatterbox.synthesize_line

    def gated(*, out_path, **kw):
        n = int(out_path.stem.split("_")[1])
        if n == 1:
            seen.set(); gate.wait(10)
        return orig(out_path=out_path, **kw)
    tts_chatterbox.synthesize_line = gated
    try:
        client.post(f"/api/projects/{key}/dub/run",
                    json={"mode": "selected", "line_nos": [1, 2]})
        assert seen.wait(10)
        # re-submit line 1 (already in plan) + new line 3
        r = client.post(f"/api/projects/{key}/dub/run",
                        json={"mode": "selected", "line_nos": [1, 3]})
        assert r.json().get("appended") is True
        gate.set()
        assert job_queue.wait_idle(30)
    finally:
        tts_chatterbox.synthesize_line = orig
        gate.set()

    cfg = fake_tts
    # line 1 synthesized exactly once despite being submitted twice
    assert cfg.calls.count(1) == 1
    assert set(cfg.calls) == {1, 2, 3}


def test_dub_all_still_409s_when_busy(client, video, fake_demucs, fake_tts):
    """Only 'selected' appends; 'all'/'changed' still reject with 409 while
    a run is active (they'd race the badges they're computed from)."""
    import threading
    from backend.pipeline import tts_chatterbox

    key = make_ready(client, video)
    gate = threading.Event()
    seen = threading.Event()
    orig = tts_chatterbox.synthesize_line

    def gated(*, out_path, **kw):
        n = int(out_path.stem.split("_")[1])
        if n == 1:
            seen.set(); gate.wait(10)
        return orig(out_path=out_path, **kw)
    tts_chatterbox.synthesize_line = gated
    try:
        client.post(f"/api/projects/{key}/dub/run",
                    json={"mode": "selected", "line_nos": [1]})
        assert seen.wait(10)
        r = client.post(f"/api/projects/{key}/dub/run", json={"mode": "all"})
        assert r.status_code == 409
        gate.set()
        assert job_queue.wait_idle(30)
    finally:
        tts_chatterbox.synthesize_line = orig
        gate.set()


def test_appended_lines_share_one_undo(client, video, fake_demucs, fake_tts):
    """One Undo after an appended run reverts the WHOLE combined run."""
    import threading
    from backend.pipeline import tts_chatterbox

    key = make_ready(client, video)
    gate = threading.Event()
    seen = threading.Event()
    orig = tts_chatterbox.synthesize_line

    def gated(*, out_path, **kw):
        n = int(out_path.stem.split("_")[1])
        if n == 1:
            seen.set(); gate.wait(10)
        return orig(out_path=out_path, **kw)
    tts_chatterbox.synthesize_line = gated
    try:
        client.post(f"/api/projects/{key}/dub/run",
                    json={"mode": "selected", "line_nos": [1]})
        assert seen.wait(10)
        client.post(f"/api/projects/{key}/dub/run",
                    json={"mode": "selected", "line_nos": [2, 3]})
        gate.set()
        assert job_queue.wait_idle(30)
    finally:
        tts_chatterbox.synthesize_line = orig
        gate.set()

    assert all(row["badge"] == "clean" for row in grid(client, key)["rows"])
    r = client.post(f"/api/projects/{key}/dub/undo")
    assert r.status_code == 200
    # all three restored to never-dubbed in a single undo
    assert sorted(r.json()["restored"]) == [1, 2, 3]
