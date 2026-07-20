"""Phase 2 tests.

Demucs itself is faked (no GPU / model weights here): the fake writes real
vocals.wav / no_vocals.wav via ffmpeg so everything downstream — pydub
loading, silence-trim extraction, clip serving, pruning — runs for real.

Covers: single-slot job queue (Busy -> 409, terminal states, post-run
visibility), the extract_clips job end-to-end, pruning persistence across
re-extraction, refs/ invalidation, speaker CRUD + defaults cascade over the
API, clip serving, and extract_reference_clip's trim/pad rules.
"""
import subprocess
import threading
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
Short one.
"""

CSV = ("speaker,line_no,start,end,text\n"
       "Obi Wan,1,0.200,2.200,Hello there.\n"
       "Grievous,2,2.600,4.600,General Kenobi!\n"
       "Obi Wan,3,4.800,5.300,Short one.\n")


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
    """6s clip whose audio is speech-shaped enough for silence detection:
    tone bursts separated by silence."""
    p = tmp_path_factory.mktemp("m") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=6:size=160x120:rate=10",
         "-f", "lavfi", "-i",
         "sine=frequency=300:duration=6",
         "-af", "volume='if(lt(mod(t,1.0),0.6),1.0,0.0)':eval=frame",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


@pytest.fixture()
def fake_demucs(monkeypatch):
    """Stand-in separator: 'vocals' = the real extracted audio, 'no_vocals' =
    silence of the same length. Writes to the exact demucs output layout."""
    from backend.pipeline import demucs_runner

    def fake(video, sep_root, force_cpu=False):
        from backend import config
        stem = sep_root / config.DEMUCS_MODEL / "source"
        stem.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn",
                        str(stem / "vocals.wav")], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(stem / 'vocals.wav'),
                        "-af", "volume=0", str(stem / "no_vocals.wav")],
                       check=True, capture_output=True)

    calls = {"n": 0}
    def counting(video, sep_root, force_cpu=False):
        calls["n"] += 1
        fake(video, sep_root, force_cpu)
    monkeypatch.setattr(demucs_runner, "_run_cli", counting)
    return calls


def project_with_media(client, video):
    r = client.post("/api/projects", json={"name": "P"})
    key = r.json()["key"]
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    client.put(f"/api/projects/{key}/csv", content=CSV,
               headers={"content-type": "text/csv"})
    return key


def run_extraction(client, key):
    r = client.post(f"/api/projects/{key}/extract_clips")
    assert r.status_code == 202, r.text
    assert job_queue.wait_idle(30), "job did not finish"
    snap = client.get("/api/jobs/current").json()
    assert snap["job"]["status"] == "done", snap["job"]
    return snap["job"]


# ------------------------------ job queue -------------------------------------

def test_single_slot_busy(client):
    release = threading.Event()
    job_queue.submit("sleepy", "x", lambda job: release.wait(5) and None)
    with pytest.raises(Exception):
        job_queue.submit("second", "x", lambda job: None)
    assert client.get("/api/jobs/current").json()["busy"] is True
    release.set()
    assert job_queue.wait_idle()
    # finished job remains visible for post-run stats
    assert client.get("/api/jobs/current").json()["job"]["kind"] == "sleepy"


def test_failed_job_never_kills_service(client):
    def boom(job):
        raise RuntimeError("synthetic explosion")
    job_queue.submit("boom", "x", boom)
    job_queue.wait_idle()
    snap = client.get("/api/jobs/current").json()["job"]
    assert snap["status"] == "failed"
    assert "synthetic explosion" in snap["error"]
    assert client.get("/api/health").json()["ok"] is True


# --------------------------- extraction job -----------------------------------

def test_extract_clips_end_to_end(client, video, fake_demucs):
    key = project_with_media(client, video)
    job = run_extraction(client, key)
    assert job["result"]["per_speaker"] == {"Obi_Wan": 2, "Grievous": 1}
    assert fake_demucs["n"] == 1

    sp = client.get(f"/api/projects/{key}/speakers").json()["speakers"]
    by = {s["key"]: s for s in sp}
    assert [c["line_no"] for c in by["Obi_Wan"]["clips"]] == [1, 3]
    assert by["Obi_Wan"]["n_lines"] == 2
    # every clip has a positive duration and serves as audio/wav
    for s in sp:
        for c in s["clips"]:
            assert c["duration_s"] and c["duration_s"] > 0
            r = client.get(
                f"/api/projects/{key}/speakers/{s['key']}/clips/{c['line_no']}")
            assert r.status_code == 200
            assert r.headers["content-type"] == "audio/wav"

    # demucs cached: second run does NOT re-separate
    run_extraction(client, key)
    assert fake_demucs["n"] == 1


def test_extraction_requires_video_and_speakers(client, video):
    r = client.post("/api/projects", json={"name": "bare"})
    key = r.json()["key"]
    assert client.post(f"/api/projects/{key}/extract_clips").status_code == 400
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    assert client.post(f"/api/projects/{key}/extract_clips").status_code == 400


def test_extract_409_while_busy(client, video, fake_demucs):
    key = project_with_media(client, video)
    release = threading.Event()
    job_queue.submit("hold", key, lambda job: release.wait(5) and None)
    assert client.post(f"/api/projects/{key}/extract_clips").status_code == 409
    release.set()
    job_queue.wait_idle()


# ------------------------- pruning + refs -------------------------------------

def test_prune_persists_across_reextraction(client, video, fake_demucs):
    from backend import store
    from backend.pipeline.reference_clips import clip_path, ref_path
    key = project_with_media(client, video)
    run_extraction(client, key)
    d = store.project_dir(key)

    # simulate a Phase-3-built concatenated ref, then prune line 3
    ref_path(d, "Obi_Wan").parent.mkdir(exist_ok=True)
    ref_path(d, "Obi_Wan").write_bytes(b"fake")
    r = client.delete(f"/api/projects/{key}/speakers/Obi_Wan/clips/3")
    assert r.status_code == 200
    assert r.json()["pruned"] == [3]
    assert not clip_path(d, "Obi_Wan", 3).exists()
    assert not ref_path(d, "Obi_Wan").exists()      # derived state invalidated
    # deleting again -> 404
    assert client.delete(
        f"/api/projects/{key}/speakers/Obi_Wan/clips/3").status_code == 404

    # re-extraction must NOT resurrect the pruned clip
    job = run_extraction(client, key)
    assert job["result"]["skipped_pruned"] == 1
    assert not clip_path(d, "Obi_Wan", 3).exists()
    assert clip_path(d, "Obi_Wan", 1).exists()


# ------------------------ speaker CRUD + defaults -----------------------------

def test_add_rename_and_defaults_cascade(client, video, fake_demucs):
    from backend import store
    key = project_with_media(client, video)
    # add + duplicate rejected
    assert client.post(f"/api/projects/{key}/speakers",
                       json={"name": "Narrator"}).status_code == 201
    assert client.post(f"/api/projects/{key}/speakers",
                       json={"name": "Narrator"}).status_code == 409
    # rename is display-only
    r = client.patch(f"/api/projects/{key}/speakers/Narrator",
                     json={"display_name": "The Narrator"})
    assert r.json()["display_name"] == "The Narrator"

    # make Obi Wan's lines clean so the cascade is observable
    with store.locked(key):
        ps = store.load(key)
        for n in (1, 3):
            ps.mark_generating([n]); ps.mark_success(n, 1.0)
        ps.apply_line_edits([{"line_no": 3, "field": "exaggeration", "value": 0.9}])
        ps.mark_generating([3]); ps.mark_success(3, 1.0)   # regen after override
        ps.save()

    r = client.put(f"/api/projects/{key}/speakers/Obi_Wan/defaults",
                   json={"field": "exaggeration", "value": 0.8})
    assert r.status_code == 200
    out = r.json()
    # line 1 inherits -> needs_full; line 3 overridden -> untouched
    assert out["deltas"] == {"1": "needs_full"}
    assert out["speaker"]["effective"]["exaggeration"] == \
        {"value": 0.8, "source": "speaker"}

    # bad field / unknown speaker
    assert client.put(f"/api/projects/{key}/speakers/Obi_Wan/defaults",
                      json={"field": "text", "value": "x"}).status_code == 400
    assert client.put(f"/api/projects/{key}/speakers/Ghost/defaults",
                      json={"field": "tolerance", "value": 1.2}).status_code == 404


# --------------------- extract_reference_clip rules ---------------------------

def _seg(ms, silent=False, level_db=-6):
    from pydub import AudioSegment
    from pydub.generators import Sine
    if silent:
        return AudioSegment.silent(duration=ms)
    return Sine(440).to_audio_segment(duration=ms).apply_gain(level_db)


def test_short_slot_extracted_raw_with_context_pad():
    from backend.pipeline.reference_clips import extract_reference_clip
    track = _seg(5000)
    out = extract_reference_clip(track, 2000, 2500)   # 500ms < NO_TRIM_UNDER_MS
    assert len(out) == 500 + 2 * 100                  # widened by MIN_EDGE_PAD_MS


def test_silence_trimmed_but_edge_floor_respected():
    from backend.pipeline.reference_clips import extract_reference_clip
    # [1s silence][2s tone][1s silence] — window covers all 4s
    track = _seg(1000, silent=True) + _seg(2000) + _seg(1000, silent=True)
    out = extract_reference_clip(track, 0, 4000)
    # leading/trailing silence largely stripped (pad_ms=60 cosmetic remains),
    # but never trimmed into the tone itself
    assert 2000 <= len(out) <= 2400


def test_track_edge_shortfall_pulls_context():
    from backend.pipeline.reference_clips import extract_reference_clip
    # tone starts at 0 — no leading silence to strip, floor pulls from track
    track = _seg(3000) + _seg(1000, silent=True)
    out = extract_reference_clip(track, 0, 3000)
    assert len(out) >= 2900


def test_build_concat_reference_caps_length(tmp_path):
    from backend.pipeline.reference_clips import build_concat_reference
    clips = []
    for i in range(5):
        p = tmp_path / f"c{i}.wav"
        _seg(8000).export(p, format="wav")
        clips.append(p)
    out = tmp_path / "ref.wav"
    build_concat_reference(clips, out, max_s=20)
    from pydub import AudioSegment
    dur = len(AudioSegment.from_wav(out))
    # stops adding once >= 20s: 3 clips * (8000+200) = 24.6s max
    assert 20000 <= dur <= 25000
