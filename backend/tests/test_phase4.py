"""Phase 4 tests: Final Remix.

The audio math is verified by construction: the test video's audio is a LOUD
constant tone, the fake no_vocals is SILENT, and the dubbed takes are silent
wavs — so a correctly spliced window must measure near-silent in the final
mix while everything outside stays loud. Natural-mode bleed is a take longer
than its slot: its silence must extend past the slot boundary (mixed over,
never cut).

Also: merge_spans unit behavior, effective_window, mux integrity (video
stream copied, duration preserved), has_final in the grid payload, download,
and the guard rails (no video / nothing dubbed / 404 before remix).
"""
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend.job_queue import jobs as job_queue

SRT = """\
1
00:00:01,000 --> 00:00:02,000
First line.

2
00:00:03,500 --> 00:00:04,500
Second line.
"""

CSV = ("speaker,line_no,start,end,text\n"
       "A,1,1.0,2.0,First line.\n"
       "A,2,3.5,4.5,Second line.\n")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "STORAGE_ROOT", tmp_path / "storage")
    job_queue.wait_idle()
    job_queue.reset_for_tests()
    from backend.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def tone_video(tmp_path_factory):
    """6s video whose audio is a CONSTANT loud tone — silence anywhere in the
    final mix proves a splice happened there."""
    p = tmp_path_factory.mktemp("m4") / "tone.mp4"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=6:size=160x120:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


@pytest.fixture()
def fake_demucs(monkeypatch):
    """no_vocals = SILENCE (all 'speech' stripped), vocals = the real audio."""
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


def silent_wav(path, dur):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"anullsrc=r=24000:cl=mono:d={dur}", str(path)],
                   check=True, capture_output=True)


def make_dubbed(client, tone_video, fit_durations: dict[int, float]):
    """Project with fake dubbed takes: silent fit wavs + clean badges."""
    from backend import store
    from backend.pipeline import dub_work
    key = client.post("/api/projects", json={"name": "R4"}).json()["key"]
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("t.mp4", tone_video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    client.put(f"/api/projects/{key}/csv", content=CSV,
               headers={"content-type": "text/csv"})
    d = store.project_dir(key)
    with store.locked(key):
        ps = store.load(key)
        for n, dur in fit_durations.items():
            silent_wav(dub_work.raw_path(d, n), dur)
            silent_wav(dub_work.fit_path(d, n), dur)
            ps.mark_generating([n])
            ps.mark_success(n, raw_duration_s=dur)
        ps.save()
    return key


def run_remix(client, key):
    r = client.post(f"/api/projects/{key}/remix")
    assert r.status_code == 202, r.text
    assert job_queue.wait_idle(60)
    snap = client.get("/api/jobs/current").json()["job"]
    assert snap["status"] == "done", snap
    return snap


def final_audio_dbfs(path, start_s, end_s):
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)[int(start_s * 1000):int(end_s * 1000)]
    return seg.dBFS


# ------------------------------ unit pieces -----------------------------------

def test_merge_spans():
    from backend.pipeline.assembly import merge_spans
    assert merge_spans([]) == []
    assert merge_spans([(5, 9), (0, 3)]) == [(0, 3), (5, 9)]
    assert merge_spans([(0, 4), (2, 6), (6, 8), (10, 11)]) == [(0, 8), (10, 11)]


def test_effective_window(tmp_path):
    from backend.pipeline import timefit
    p = tmp_path / "c.wav"
    silent_wav(p, 1.5)
    s, e = timefit.effective_window(2.0, p)
    assert s == 2000 and e == pytest.approx(3500, abs=60)


# ------------------------------ remix e2e -------------------------------------

def test_remix_splices_dubbed_windows_only(client, tone_video, fake_demucs):
    from backend import store
    from backend.pipeline import assembly
    key = make_dubbed(client, tone_video, {1: 1.0, 2: 1.0})   # exact slots
    job = run_remix(client, key)
    assert job["result"]["n_lines_spliced"] == 2
    assert job["result"]["n_windows_merged"] == 2

    out = assembly.final_path(store.project_dir(key))
    assert out.exists()
    # inside dubbed windows: original tone stripped + silent take -> quiet
    assert final_audio_dbfs(out, 1.15, 1.85) < -45
    assert final_audio_dbfs(out, 3.65, 4.35) < -45
    # outside: original tone untouched -> loud
    assert final_audio_dbfs(out, 0.2, 0.8) > -30
    assert final_audio_dbfs(out, 2.3, 3.2) > -30
    assert final_audio_dbfs(out, 5.0, 5.8) > -30


def test_natural_bleed_is_mixed_never_cut(client, tone_video, fake_demucs):
    from backend import store
    from backend.pipeline import assembly
    # line 1's take is 3.0s in a 1.0s slot (Natural-style bleed): its window
    # [1.0, 4.0] overlaps line 2's [3.5, 4.5] -> merged into one span
    key = make_dubbed(client, tone_video, {1: 3.0, 2: 1.0})
    job = run_remix(client, key)
    assert job["result"]["n_lines_spliced"] == 2
    assert job["result"]["n_windows_merged"] == 1          # overlap merged
    out = assembly.final_path(store.project_dir(key))
    # the bleed region past line 1's slot (2.0..3.5) is inside the effective
    # window: vocals stripped there too, silent take over it -> quiet
    assert final_audio_dbfs(out, 2.2, 3.3) < -45
    assert final_audio_dbfs(out, 4.0, 4.4) < -45
    # before/after the merged window: untouched original
    assert final_audio_dbfs(out, 0.2, 0.8) > -30
    assert final_audio_dbfs(out, 4.8, 5.6) > -30


def test_failed_and_undubbed_lines_keep_original(client, tone_video, fake_demucs):
    from backend import store
    from backend.pipeline import assembly
    key = make_dubbed(client, tone_video, {1: 1.0})   # line 2 never dubbed
    with store.locked(key):
        ps = store.load(key)
        ps.mark_generating([2])
        ps.mark_failed(2, "boom")                     # and then failed
        ps.save()
    job = run_remix(client, key)
    assert job["result"]["n_lines_spliced"] == 1
    out = assembly.final_path(store.project_dir(key))
    assert final_audio_dbfs(out, 1.15, 1.85) < -45    # dubbed window spliced
    assert final_audio_dbfs(out, 3.6, 4.4) > -30      # failed line: original


def test_mux_integrity_and_download(client, tone_video, fake_demucs):
    import json
    from backend import store
    from backend.pipeline import assembly
    key = make_dubbed(client, tone_video, {1: 1.0})
    # 404 before any remix; has_final false
    assert client.get(f"/api/projects/{key}/remix/download").status_code == 404
    assert client.get(f"/api/projects/{key}/dub/grid").json()["has_final"] is False
    run_remix(client, key)
    out = assembly.final_path(store.project_dir(key))
    cp = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", str(out)], capture_output=True, text=True, check=True)
    info = json.loads(cp.stdout)
    kinds = {s["codec_type"] for s in info["streams"]}
    assert kinds == {"video", "audio"}
    assert float(info["format"]["duration"]) == pytest.approx(6.0, abs=0.4)
    # grid + download reflect the output
    assert client.get(f"/api/projects/{key}/dub/grid").json()["has_final"] is True
    r = client.get(f"/api/projects/{key}/remix/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert "R4_dubbed.mp4" in r.headers["content-disposition"]


def test_remix_guards(client, tone_video, fake_demucs):
    # no video
    key = client.post("/api/projects", json={"name": "bare4"}).json()["key"]
    assert client.post(f"/api/projects/{key}/remix").status_code == 400
    # video but nothing dubbed
    key2 = client.post("/api/projects", json={"name": "bare5"}).json()["key"]
    client.post(f"/api/projects/{key2}/upload/video",
                files={"file": ("t.mp4", tone_video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key2}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    r = client.post(f"/api/projects/{key2}/remix")
    assert r.status_code == 400
    assert "no successfully dubbed lines" in r.json()["detail"]


# --------------------------- timestamp mute -----------------------------------

def test_timestamp_mute_off_by_default_short_dub_bleeds(client, tone_video, fake_demucs):
    """Documents the OFF behavior: a dub SHORTER than its slot only mutes
    for its own duration -- the tail of the original slot bleeds through
    when timestamp_mute is off. (timestamp_mute defaults ON for new
    projects as of Phase 5.4 -- explicitly disabled here to test this case.)"""
    from backend import store
    from backend.pipeline import assembly
    key = make_dubbed(client, tone_video, {1: 0.4})   # slot is 1.0s (1.0-2.0)
    assert client.get(f"/api/projects/{key}/dub/grid").json()["timestamp_mute"] is True  # new default
    client.put(f"/api/projects/{key}/dub/timestamp_mute", json={"value": False})
    assert client.get(f"/api/projects/{key}/dub/grid").json()["timestamp_mute"] is False
    run_remix(client, key)
    out = assembly.final_path(store.project_dir(key))
    assert final_audio_dbfs(out, 1.05, 1.35) < -45     # muted: dub's own span
    assert final_audio_dbfs(out, 1.6, 1.9) > -30        # BLEED: original tone audible


def test_timestamp_mute_on_covers_full_slot(client, tone_video, fake_demucs):
    from backend import store
    from backend.pipeline import assembly
    key = make_dubbed(client, tone_video, {1: 0.4})   # slot 1.0-2.0, dub only 0.4s
    r = client.put(f"/api/projects/{key}/dub/timestamp_mute", json={"value": True})
    assert r.status_code == 200 and r.json()["timestamp_mute"] is True
    assert client.get(f"/api/projects/{key}/dub/grid").json()["timestamp_mute"] is True

    job = run_remix(client, key)
    assert job["result"]["timestamp_mute"] is True
    out = assembly.final_path(store.project_dir(key))
    assert final_audio_dbfs(out, 1.05, 1.35) < -45     # dub's own span: muted
    assert final_audio_dbfs(out, 1.6, 1.9) < -45        # tail: now ALSO muted
    # untouched surroundings still loud
    assert final_audio_dbfs(out, 0.2, 0.8) > -30
    assert final_audio_dbfs(out, 2.2, 2.9) > -30


def test_timestamp_mute_never_narrows_a_longer_dub(client, tone_video, fake_demucs):
    """A dub LONGER than its slot (Natural bleed) must not be shortened back
    down when timestamp_mute is on — mute widens, never narrows."""
    from backend import store
    from backend.pipeline import assembly
    key = make_dubbed(client, tone_video, {1: 3.0})   # slot 1.0-2.0, dub 3.0s
    client.put(f"/api/projects/{key}/dub/timestamp_mute", json={"value": True})
    run_remix(client, key)
    out = assembly.final_path(store.project_dir(key))
    assert final_audio_dbfs(out, 1.1, 1.9) < -45       # inside slot: muted
    assert final_audio_dbfs(out, 2.2, 3.3) < -45       # bleed past slot: still muted
