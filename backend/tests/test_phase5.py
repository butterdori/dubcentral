"""Phase 5 tests: per-project switchable engine (Fun-CosyVoice3).

The CosyVoice model itself is faked (synthesize_line writes real wavs, same
pattern as the chatterbox fake) so dispatch, reference selection, engine
switching, and the per-engine payloads all run for real against the actual
state machine and file layout.
"""
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend.job_queue import jobs as job_queue

SRT = """\
1
00:00:00,200 --> 00:00:04,200
Hello there.

2
00:00:04,600 --> 00:00:05,400
Hi.

3
00:00:05,500 --> 00:00:05,900
Mm.
"""

# original-language SRT: entries 1-2 share indices; entry 9 mismatches on
# index but its start matches line 3 (timestamp-fallback path)
ORIG_SRT = """\
1
00:00:00,200 --> 00:00:04,200
こんにちは、そこの人。

2
00:00:04,600 --> 00:00:05,400
やあ。

9
00:00:05,600 --> 00:00:05,900
うん。
"""

CSV = ("speaker,line_no,start,end,text\n"
       "A,1,0.200,4.200,Hello there.\n"
       "A,2,4.600,5.400,Hi.\n"
       "B,3,5.500,5.900,Mm.\n")


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
    p = tmp_path_factory.mktemp("m5") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=7:size=160x120:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=7",
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
def fake_cosy(monkeypatch):
    """Fake CosyVoice synthesize_line: records the kwargs it was called
    with, writes a real wav of a controllable duration."""
    from backend.pipeline import tts_cosyvoice

    class Cfg:
        durations: dict[int, float] = {}
        calls: list[dict] = []
    cfg = Cfg()

    def synth(*, text, prompt_text, ref_path, speed, instruct_text,
              out_path, precision="fp16", force_cpu=False):
        n = int(out_path.stem.split("_")[1])
        assert ref_path.exists()
        cfg.calls.append({"line": n, "text": text, "prompt_text": prompt_text,
                          "speed": speed, "instruct_text": instruct_text})
        dur = cfg.durations.get(n, 1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        f"sine=frequency=550:duration={dur}",
                        str(out_path)], check=True, capture_output=True)
        return dur
    monkeypatch.setattr(tts_cosyvoice, "synthesize_line", synth)
    return cfg


@pytest.fixture()
def fake_chatter_fc(monkeypatch):
    """Like fake_chatter, but records force_cpu per call for the CUDA-toggle
    tests (kept separate so the other tests' simpler fixture is unaffected)."""
    from backend.pipeline import tts_chatterbox
    calls = []

    def synth(*, text, language, ref_path, exaggeration, cfg_weight,
              out_path, precision="fp16", force_cpu=False):
        n = int(out_path.stem.split("_")[1])
        calls.append({"line": n, "force_cpu": force_cpu})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1.0",
                        str(out_path)], check=True, capture_output=True)
        return 1.0
    monkeypatch.setattr(tts_chatterbox, "synthesize_line", synth)
    calls_holder = calls
    class Holder:
        pass
    h = Holder()
    h.calls = calls_holder
    return h


@pytest.fixture()
def fake_chatter(monkeypatch):
    from backend.pipeline import tts_chatterbox
    calls = []

    def synth(*, text, language, ref_path, exaggeration, cfg_weight,
              out_path, precision="fp16", force_cpu=False):
        n = int(out_path.stem.split("_")[1])
        calls.append(n)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1.0",
                        str(out_path)], check=True, capture_output=True)
        return 1.0
    monkeypatch.setattr(tts_chatterbox, "synthesize_line", synth)
    return calls


def make_project(client, video, with_original=True, engine="cosyvoice3"):
    key = client.post("/api/projects", json={"name": "P5"}).json()["key"]
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    client.put(f"/api/projects/{key}/csv", content=CSV,
               headers={"content-type": "text/csv"})
    # explicit regardless of the app's current default -- isolates tests
    # from ever silently depending on that default again
    r = client.put(f"/api/projects/{key}/engine", json={"engine": engine})
    assert r.status_code == 200, r.text
    if with_original:
        r = client.post(f"/api/projects/{key}/upload/original_srt",
                        files={"file": ("o.srt", ORIG_SRT, "text/plain")})
        assert r.status_code == 200, r.text
    client.post(f"/api/projects/{key}/extract_clips")
    assert job_queue.wait_idle(30)
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


# --------------------------- original SRT alignment ---------------------------

def test_original_srt_alignment(client, video, fake_demucs):
    key = make_project(client, video)
    from backend import store
    ps = store.load(key)
    assert ps.line(1)["original_text"].startswith("こんにちは")
    assert ps.line(2)["original_text"] == "やあ。"
    # index 9 didn't match any line_no; fell back to nearest start (line 3)
    assert ps.line(3)["original_text"] == "うん。"
    g = grid(client, key)
    assert g["rows"][0]["original_text"].startswith("こんにちは")


def test_original_srt_requires_lines(client):
    key = client.post("/api/projects", json={"name": "bare"}).json()["key"]
    r = client.post(f"/api/projects/{key}/upload/original_srt",
                    files={"file": ("o.srt", ORIG_SRT, "text/plain")})
    assert r.status_code == 400


def test_original_text_edit_no_badge_under_chatterbox(client, video,
                                                      fake_demucs, fake_chatter):
    key = make_project(client, video, engine="chatterbox")
    run_dub(client, key)
    deltas = client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "original_text", "value": "書き換え"}]}
    ).json()["deltas"]
    assert deltas == {}                        # chatterbox doesn't care
    assert grid(client, key)["rows"][0]["badge"] == "clean"


# ------------------------------ engine switch ---------------------------------

def test_engine_switch_semantics(client, video, fake_demucs, fake_chatter):
    key = make_project(client, video, engine="chatterbox")
    run_dub(client, key)                       # 3 clean chatterbox takes
    # some engine-specific state on both sides
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 2, "field": "exaggeration", "value": 0.9}]})
    client.put(f"/api/projects/{key}/speakers/A/defaults",
               json={"field": "cfg_weight", "value": 0.3})

    r = client.put(f"/api/projects/{key}/engine", json={"engine": "cosyvoice3"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["engine"] == "cosyvoice3"
    # every line with a take ends up needs_full; deltas only report actual
    # CHANGES — lines 1-2 were already needs_full from the edits above, so
    # only line 3's badge moved
    assert out["deltas"] == {"3": "needs_full"}
    assert all(r["badge"] == "needs_full"
               for r in grid(client, key)["rows"])
    # inactive (chatterbox) overrides + speaker defaults cleared
    assert out["cleared_line_overrides"] == 1
    assert out["cleared_speaker_defaults"] == 1
    from backend import store
    ps = store.load(key)
    assert "exaggeration" not in ps.line(2)["overrides"]
    assert "cfg_weight" not in ps.data["speakers"]["A"]["defaults"]
    # grid payload flips engine + fields
    g = grid(client, key)
    assert g["engine"] == "cosyvoice3"
    assert g["engine_fields"] == ["speed", "instruct_text"]
    # bad engine + same-engine no-op
    assert client.put(f"/api/projects/{key}/engine",
                      json={"engine": "nope"}).status_code == 400
    r = client.put(f"/api/projects/{key}/engine", json={"engine": "cosyvoice3"})
    assert r.json()["deltas"] == {}


def test_engine_switch_back_clears_cosy_fields(client, video, fake_demucs):
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/engine", json={"engine": "cosyvoice3"})
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "speed", "value": 1.2},
        {"line_no": 1, "field": "instruct_text", "value": "whisper"}]})
    out = client.put(f"/api/projects/{key}/engine",
                     json={"engine": "chatterbox"}).json()
    assert out["cleared_line_overrides"] == 2
    from backend import store
    assert store.load(key).line(1)["overrides"] == {}


# ------------------------- cosyvoice dispatch e2e -----------------------------

def test_cosyvoice_dub_run(client, video, fake_demucs, fake_cosy):
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/engine", json={"engine": "cosyvoice3"})
    # per-line knobs: speed override on 1, instruct on 2
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "speed", "value": 1.3},
        {"line_no": 2, "field": "instruct_text", "value": "speak sadly"}]})
    job = run_dub(client, key)
    assert job["result"]["engine"] == "cosyvoice3"
    assert job["result"]["n_ok"] == 3
    calls = {c["line"]: c for c in fake_cosy.calls}
    assert calls[1]["speed"] == 1.3 and calls[1]["instruct_text"] == ""
    assert calls[2]["speed"] == 1.0 and calls[2]["instruct_text"] == "speak sadly"
    # speaker A's reference prompt is a JA transcript from A's clips
    assert calls[1]["prompt_text"] in ("こんにちは、そこの人。", "やあ。")
    assert calls[3]["prompt_text"] == "うん。"        # speaker B's only clip
    assert all(r["badge"] == "clean" for r in grid(client, key)["rows"])


def test_speed_edit_is_full_tier(client, video, fake_demucs, fake_cosy):
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/engine", json={"engine": "cosyvoice3"})
    run_dub(client, key)
    n = len(fake_cosy.calls)
    deltas = client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "speed", "value": 0.9}]}).json()["deltas"]
    assert deltas == {"1": "needs_full"}       # generation-time knob
    run_dub(client, key, mode="changed")
    assert [c["line"] for c in fake_cosy.calls[n:]] == [1]


def test_cosyvoice_needs_original_text(client, video, fake_demucs, fake_cosy):
    """No original SRT uploaded -> reference selection fails per speaker,
    lines fail with an actionable message, run survives."""
    key = make_project(client, video, with_original=False)
    client.put(f"/api/projects/{key}/engine", json={"engine": "cosyvoice3"})
    job = run_dub(client, key)
    assert job["result"]["n_failed"] == 3 and job["result"]["n_ok"] == 0
    err = grid(client, key)["rows"][0]["error"]
    assert "original_text" in err and "original-language SRT" in err


def test_reference_selection_prefers_sweet_spot_and_skips_pruned(
        client, video, fake_demucs, fake_cosy):
    from backend import store
    from backend.pipeline import tts_cosyvoice
    key = make_project(client, video)
    d = store.project_dir(key)
    ps = store.load(key)
    # speaker A has clips for lines 1 (4.0s slot) and 2 (0.8s slot):
    # the 3-15s sweet spot picks line 1
    path, text = tts_cosyvoice.select_reference(ps, d, "A")
    assert path.name == "line_0001.wav" and text.startswith("こんにちは")
    # prune line 1's clip -> falls back to line 2's
    client.delete(f"/api/projects/{key}/speakers/A/clips/1")
    ps = store.load(key)
    path, text = tts_cosyvoice.select_reference(ps, d, "A")
    assert path.name == "line_0002.wav" and text == "やあ。"


def test_chatterbox_projects_completely_unaffected(client, video, fake_demucs,
                                                   fake_chatter):
    """The default engine's whole flow — grid shape, dub, badges — is
    byte-for-byte the pre-P5 behavior."""
    key = make_project(client, video, engine="chatterbox")
    g = grid(client, key)
    assert g["engine"] == "chatterbox"
    assert g["engine_fields"] == ["exaggeration", "cfg_weight"]
    job = run_dub(client, key)
    assert job["result"]["engine"] == "chatterbox"
    assert job["result"]["n_ok"] == 3
    assert sorted(set(fake_chatter)) == [1, 2, 3]


def test_engine_switch_blocked_while_busy(client, video, fake_demucs):
    import threading
    key = make_project(client, video)
    release = threading.Event()
    job_queue.submit("hold", key, lambda job: release.wait(5) and None)
    assert client.put(f"/api/projects/{key}/engine",
                      json={"engine": "cosyvoice3"}).status_code == 409
    release.set()
    job_queue.wait_idle()


# ------------------------------ CUDA toggle -----------------------------------

def test_cuda_enabled_default_and_toggle(client, video, fake_demucs):
    key = make_project(client, video)
    g = grid(client, key)
    assert g["cuda_enabled"] is False   # default: Force CPU is ON
    r = client.put(f"/api/projects/{key}/dub/cuda", json={"value": True})
    assert r.status_code == 200 and r.json()["cuda_enabled"] is True
    assert grid(client, key)["cuda_enabled"] is True
    # flips back
    r = client.put(f"/api/projects/{key}/dub/cuda", json={"value": False})
    assert r.json()["cuda_enabled"] is False


def test_cuda_disabled_forces_force_cpu_into_synthesis(client, video,
                                                       fake_demucs, fake_chatter_fc):
    """The dub run reads cuda_enabled ONCE per run and passes force_cpu into
    every synthesize_line call — verified via a fake that records it."""
    key = make_project(client, video, engine="chatterbox")
    client.put(f"/api/projects/{key}/dub/cuda", json={"value": False})
    job = run_dub(client, key)
    assert job["result"]["n_ok"] == 3
    assert all(c["force_cpu"] is True for c in fake_chatter_fc.calls)


def test_cuda_enabled_leaves_force_cpu_false(client, video, fake_demucs,
                                             fake_chatter_fc):
    key = make_project(client, video, engine="chatterbox")
    client.put(f"/api/projects/{key}/dub/cuda", json={"value": True})  # Force CPU is on by default now
    run_dub(client, key)
    assert all(c["force_cpu"] is False for c in fake_chatter_fc.calls)


def test_cuda_toggle_does_not_affect_demucs_extraction(client, video, fake_demucs):
    """Force CPU is scoped to TTS/dubbing only -- extraction (Demucs) never
    sees it, confirmed by calling the REAL (unmodified) _run_cli signature:
    if the router still tried to pass force_cpu, this fake -- which takes
    no such argument -- would raise a TypeError instead of just recording
    a call count."""
    key = client.post("/api/projects", json={"name": "P5cuda"}).json()["key"]
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    client.put(f"/api/projects/{key}/csv", content=CSV,
               headers={"content-type": "text/csv"})
    client.put(f"/api/projects/{key}/dub/cuda", json={"value": False})
    r = client.post(f"/api/projects/{key}/extract_clips")
    assert r.status_code == 202
    assert job_queue.wait_idle(30)
    snap = client.get("/api/jobs/current").json()["job"]
    assert snap["status"] == "done", snap   # would be "failed" (TypeError) otherwise


# ------------------------- reference pins (multi) + renew -----------------------

def test_has_original_text_flag(client, video, fake_demucs):
    key = make_project(client, video, with_original=False)
    assert client.get(f"/api/projects/{key}/speakers").json()["has_original_text"] is False
    client.post(f"/api/projects/{key}/upload/original_srt",
               files={"file": ("o.srt", ORIG_SRT, "text/plain")})
    assert client.get(f"/api/projects/{key}/speakers").json()["has_original_text"] is True


def run_renew(client, key):
    r = client.post(f"/api/projects/{key}/renew_reference")
    assert r.status_code == 202, r.text
    assert job_queue.wait_idle(30)
    snap = client.get("/api/jobs/current").json()["job"]
    assert snap["status"] == "done", snap
    return snap


def speaker_payload(client, key, spk):
    sp = client.get(f"/api/projects/{key}/speakers").json()["speakers"]
    return next(s for s in sp if s["key"] == spk)


def test_auto_pick_indicator_before_any_pin(client, video, fake_demucs):
    """Nothing pinned yet -> is_reference marks the single best auto-picked
    clip (unchanged single-clip behavior), no committed set."""
    from backend import store
    from backend.pipeline import tts_cosyvoice
    key = make_project(client, video)   # speaker A: lines 1 (4.0s), 2 (0.8s)
    ps = store.load(key)
    d = store.project_dir(key)
    assert tts_cosyvoice.auto_pick_line(ps, d, "A") == 1   # 4.0s is in the sweet spot

    a = speaker_payload(client, key, "A")
    assert a["ref_pins"] == [] and a["ref_pins_committed"] == []
    assert a["ref_dirty"] is False
    clips = {c["line_no"]: c for c in a["clips"]}
    assert clips[1]["is_reference"] is True
    assert clips[2]["is_reference"] is False


def test_pin_staging_has_no_cascade(client, video, fake_demucs, fake_chatter):
    """Toggling pins is pure staging -- no badge cascade until Renew,
    regardless of engine."""
    key = make_project(client, video, engine="chatterbox")
    run_dub(client, key)
    r = client.put(f"/api/projects/{key}/speakers/A/reference_pins",
                   json={"line_nos": [1, 2]})
    assert r.status_code == 200, r.text
    assert r.json()["ref_pins"] == [1, 2]
    assert r.json()["ref_dirty"] is True   # staged != committed ([])
    assert grid(client, key)["rows"][0]["badge"] == "clean"   # untouched


def test_renew_concatenates_multiple_pins_and_cascades(client, video,
                                                        fake_demucs, fake_cosy):
    key = make_project(client, video)   # cosyvoice3 default
    client.put(f"/api/projects/{key}/speakers/A/reference_pins",
               json={"line_nos": [1, 2]})
    run_dub(client, key)   # all clean, using the pre-renew auto-pick (line 1)
    assert all(r["badge"] == "clean" for r in grid(client, key)["rows"])

    job = run_renew(client, key)
    assert job["result"]["speakers"]["A"]["mode"] == "pinned"
    assert job["result"]["speakers"]["A"]["committed"] == [1, 2]
    # A's dubbed lines (1, 2) flip to needs_full; B's line 3 untouched
    rows = {r["line_no"]: r for r in grid(client, key)["rows"]}
    assert rows[1]["badge"] == rows[2]["badge"] == "needs_full"
    assert rows[3]["badge"] == "clean"

    a = speaker_payload(client, key, "A")
    assert a["ref_pins"] == a["ref_pins_committed"] == [1, 2]
    assert a["ref_dirty"] is False
    clips = {c["line_no"]: c for c in a["clips"]}
    assert clips[1]["is_reference"] is True and clips[2]["is_reference"] is True

    # the concatenated file actually exists and is longer than either source clip alone
    from backend import store
    from backend.pipeline import tts_cosyvoice, timefit
    d = store.project_dir(key)
    concat_dur = timefit.probe_duration(tts_cosyvoice.cosy_ref_path(d, "A"))
    assert concat_dur > 4.0   # 4.0s + 0.8s + gaps > either clip alone
    text = tts_cosyvoice.cosy_ref_text_path(d, "A").read_text(encoding="utf-8")
    assert text == "こんにちは、そこの人。 やあ。"   # concatenated in line_no order

    # re-dub: synthesis actually uses the concatenated file + joined transcript
    run_dub(client, key, mode="changed")
    calls = {c["line"]: c for c in fake_cosy.calls[-2:]}
    assert all(c["prompt_text"] == text for c in calls.values())


def test_renew_with_no_pins_reverts_to_auto(client, video, fake_demucs, fake_cosy):
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/speakers/A/reference_pins",
               json={"line_nos": [1, 2]})
    run_renew(client, key)
    from backend import store
    from backend.pipeline import tts_cosyvoice
    d = store.project_dir(key)
    assert tts_cosyvoice.cosy_ref_path(d, "A").exists()

    r = client.put(f"/api/projects/{key}/speakers/A/reference_pins",
                   json={"line_nos": []})
    assert r.json()["ref_dirty"] is True
    run_renew(client, key)
    assert not tts_cosyvoice.cosy_ref_path(d, "A").exists()   # reverted
    a = speaker_payload(client, key, "A")
    assert a["ref_pins_committed"] == []
    clips = {c["line_no"]: c for c in a["clips"]}
    assert clips[1]["is_reference"] is True   # back to auto-pick


def test_renew_skips_stale_pins_gracefully(client, video, fake_demucs, fake_cosy):
    """A pinned clip that's since been pruned is skipped, not a hard
    failure -- the rest of the pin set still gets committed."""
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/speakers/A/reference_pins",
               json={"line_nos": [1, 2]})
    client.delete(f"/api/projects/{key}/speakers/A/clips/2")   # prune one pin
    a = speaker_payload(client, key, "A")
    assert a["ref_pins"] == [1]   # prune auto-unstaged it

    job = run_renew(client, key)
    assert job["result"]["speakers"]["A"]["committed"] == [1]
    assert job["result"]["speakers"]["A"]["skipped"] == []   # nothing stale left staged


def test_prune_unstages_pin_but_keeps_committed_reference(client, video,
                                                          fake_demucs, fake_cosy):
    """Pruning a clip that's already baked into a committed reference does
    NOT retroactively invalidate that (self-contained) file -- it only
    marks the staged set dirty relative to it."""
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/speakers/A/reference_pins",
               json={"line_nos": [1, 2]})
    run_renew(client, key)
    from backend import store
    from backend.pipeline import tts_cosyvoice
    d = store.project_dir(key)
    before = tts_cosyvoice.cosy_ref_path(d, "A").read_bytes()

    client.delete(f"/api/projects/{key}/speakers/A/clips/2")
    a = speaker_payload(client, key, "A")
    assert a["ref_pins"] == [1]                    # unstaged
    assert a["ref_pins_committed"] == [1, 2]        # committed file untouched
    assert a["ref_dirty"] is True
    assert tts_cosyvoice.cosy_ref_path(d, "A").read_bytes() == before  # unchanged bytes


def test_renew_bad_engine_no_cascade(client, video, fake_demucs, fake_chatter):
    """Renewing is harmless under chatterbox (no lines flip) -- there's
    nothing cosyvoice-specific gating the endpoint itself, just the
    badge-cascade consequence, which stays engine-conditional."""
    key = make_project(client, video, engine="chatterbox")
    run_dub(client, key)
    client.put(f"/api/projects/{key}/speakers/A/reference_pins",
               json={"line_nos": [1]})
    run_renew(client, key)
    assert grid(client, key)["rows"][0]["badge"] == "clean"


def test_pin_bad_line_and_bad_speaker(client, video, fake_demucs):
    key = make_project(client, video)
    assert client.put(f"/api/projects/{key}/speakers/A/reference_pins",
                      json={"line_nos": [999]}).status_code == 404
    assert client.put(f"/api/projects/{key}/speakers/Ghost/reference_pins",
                      json={"line_nos": [1]}).status_code == 404


def test_renew_busy_guard(client, video, fake_demucs):
    import threading
    key = make_project(client, video)
    release = threading.Event()
    job_queue.submit("hold", key, lambda job: release.wait(5) and None)
    assert client.post(f"/api/projects/{key}/renew_reference").status_code == 409
    release.set()
    job_queue.wait_idle()


# --------------------------- Upload Center Original/Dubbed ---------------------

def test_project_meta_has_final(client, video, fake_demucs):
    key = client.post("/api/projects", json={"name": "P5final"}).json()["key"]
    assert client.get(f"/api/projects/{key}").json()["has_final"] is False
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    assert client.get(f"/api/projects/{key}").json()["has_final"] is False
    assert client.get(f"/api/projects/{key}/remix/video").status_code == 404


def test_project_meta_has_final_after_remix(client, video, fake_demucs, fake_chatter):
    key = make_project(client, video, engine="chatterbox")
    run_dub(client, key)
    r = client.post(f"/api/projects/{key}/remix")
    assert r.status_code == 202
    assert job_queue.wait_idle(60)
    assert client.get("/api/jobs/current").json()["job"]["status"] == "done"
    assert client.get(f"/api/projects/{key}").json()["has_final"] is True
    r = client.get(f"/api/projects/{key}/remix/video")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert "attachment" not in r.headers.get("content-disposition", "")  # inline, not forced download
    dl = client.get(f"/api/projects/{key}/remix/download")
    assert "attachment" in dl.headers.get("content-disposition", "")     # download keeps forcing it


# -------------------------------- new defaults ----------------------------------

def test_new_project_defaults(client):
    key = client.post("/api/projects", json={"name": "Pdef"}).json()["key"]
    g = client.get(f"/api/projects/{key}/dub/grid").json()
    assert g["engine"] == "cosyvoice3"
    assert g["cuda_enabled"] is False       # Force CPU on by default
    assert g["timestamp_mute"] is True      # Timestamp mute on by default


# ------------------------ endofprompt marker wrapping ---------------------------

def test_wrap_prompt_text_marker_placement():
    """zero-shot: marker sits BETWEEN the style prefix and the reference
    transcript (content follows the marker) -- confirmed against the model
    card's zero_shot examples."""
    from backend.pipeline.tts_cosyvoice import _wrap_prompt_text
    out = _wrap_prompt_text("希望你以后能够做的比我还好呦。")
    assert out == "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    # already-wrapped input passes through unchanged (no double-wrapping)
    assert _wrap_prompt_text("X<|endofprompt|>Y") == "X<|endofprompt|>Y"


def test_wrap_instruct_text_marker_placement():
    """instruct2: marker is a SUFFIX with nothing after it -- confirmed
    against the model card's own instruct2 example (a dialect-conversion
    instruction), distinct from zero-shot's placement."""
    from backend.pipeline.tts_cosyvoice import _wrap_instruct_text
    out = _wrap_instruct_text("请用广东话表达。")
    assert out == "You are a helpful assistant.请用广东话表达。<|endofprompt|>"
    assert _wrap_instruct_text("X<|endofprompt|>") == "X<|endofprompt|>"  # no double-wrap


# ------------------------- bulk-clear instr (frontend contract) -----------------

def test_bulk_clear_instruct_text_via_null(client, video, fake_demucs, fake_cosy):
    """The bulk-clear UI control sends instruct_text: null (same contract
    as clearing a single line's field) -- verify the API side of that
    round-trips correctly for multiple lines at once."""
    key = make_project(client, video)
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "instruct_text", "value": "speak sadly"},
        {"line_no": 2, "field": "instruct_text", "value": "speak sadly"}]})
    rows = grid(client, key)["rows"]
    assert rows[0]["fields"]["instruct_text"]["source"] == "line"
    r = client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "instruct_text", "value": None},
        {"line_no": 2, "field": "instruct_text", "value": None}]})
    assert r.status_code == 200
    rows = grid(client, key)["rows"]
    assert rows[0]["fields"]["instruct_text"] == {"value": "", "source": "project"}
    assert rows[1]["fields"]["instruct_text"] == {"value": "", "source": "project"}
