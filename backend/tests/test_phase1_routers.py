"""Phase 1 API tests via TestClient: project CRUD + recency sort, video
upload validation (real ffprobe on a generated clip, rejects garbage and
audio-less video), SRT -> line table, CSV export/edit/re-upload round-trips,
and the full-reset semantics (badges/overrides/derived dirs wiped,
separated/ preserved on line resets, wiped on video replacement)."""
import subprocess

import pytest
from fastapi.testclient import TestClient

SRT = """\
1
00:00:00,500 --> 00:00:02,000
Hello there.

2
00:00:02,500 --> 00:00:04,000
General Kenobi!
You are a bold one.

3
00:00:05,000 --> 00:00:06,500
(dramatic music)
"""


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from backend import config
    monkeypatch.setattr(config, "STORAGE_ROOT", tmp_path / "storage")
    from backend.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    """2s test pattern with a sine audio track."""
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(p)],
        check=True, capture_output=True)
    return p


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("media") / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
        check=True, capture_output=True)
    return p


def make_project(client, name="My Show"):
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["key"]


def upload_srt(client, key):
    r = client.post(f"/api/projects/{key}/upload/srt",
                    files={"file": ("subs.srt", SRT, "text/plain")})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------ projects --------------------------------------

def test_create_list_delete(client):
    key = make_project(client)
    assert key == "My_Show"
    # duplicate rejected
    assert client.post("/api/projects", json={"name": "My Show"}).status_code == 409
    listing = client.get("/api/projects").json()["projects"]
    assert [p["key"] for p in listing] == ["My_Show"]
    assert listing[0]["stats"]["n_lines"] == 0
    assert client.delete("/api/projects/My_Show").status_code == 204
    assert client.get("/api/projects").json()["projects"] == []
    assert client.delete("/api/projects/My_Show").status_code == 404


def test_recency_sort(client):
    make_project(client, "older")
    key2 = make_project(client, "newer")
    upload_srt(client, "older")  # touching "older" makes it most recent
    listing = client.get("/api/projects").json()["projects"]
    assert [p["key"] for p in listing] == ["older", key2]


def test_404s(client):
    assert client.get("/api/projects/nope").status_code == 404
    assert client.get("/api/projects/nope/thumbnail").status_code == 404
    assert client.get("/api/projects/nope/csv").status_code == 404


# ------------------------------- video ----------------------------------------

def test_video_upload_probe_thumbnail(client, video):
    key = make_project(client)
    r = client.post(f"/api/projects/{key}/upload/video",
                    files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["has_video"] and meta["has_thumbnail"]
    assert 1.5 < meta["stats"]["duration_s"] < 2.6
    # thumbnail served
    t = client.get(f"/api/projects/{key}/thumbnail")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    # video streams with range support (player seeking)
    v = client.get(f"/api/projects/{key}/video", headers={"Range": "bytes=0-99"})
    assert v.status_code == 206 and len(v.content) == 100


def test_garbage_video_rejected(client):
    key = make_project(client)
    r = client.post(f"/api/projects/{key}/upload/video",
                    files={"file": ("x.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 400
    assert "rejected" in r.json()["detail"]
    assert client.get(f"/api/projects/{key}/video").status_code == 404


def test_audio_less_video_rejected(client, silent_video):
    key = make_project(client)
    r = client.post(f"/api/projects/{key}/upload/video",
                    files={"file": ("s.mp4", silent_video.read_bytes(), "video/mp4")})
    assert r.status_code == 400
    assert "no audio" in r.json()["detail"]


def test_video_reupload_wipes_separated_only(client, video):
    key = make_project(client)
    upload_srt(client, key)
    from backend import store
    d = store.project_dir(key)
    (d / "separated").mkdir()
    (d / "separated" / "marker").touch()
    (d / "dub_work").mkdir()
    (d / "dub_work" / "marker").touch()
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("clip.mp4", video.read_bytes(), "video/mp4")})
    assert not (d / "separated").exists()      # stale for the new video
    assert (d / "dub_work" / "marker").exists()  # line table untouched


# ------------------------------ SRT / prep ------------------------------------

def test_srt_upload_builds_lines(client):
    key = make_project(client)
    out = upload_srt(client, key)
    assert out["n_lines"] == 3
    from backend import store
    ps = store.load(key)
    l2 = ps.line(2)
    assert l2["text"] == "General Kenobi! You are a bold one."  # single-lined
    assert l2["start"] == 2.5 and l2["end"] == 4.0
    assert all(l["badge"] == "never" for l in ps.data["lines"])
    assert ps.data["speakers"] == {}  # SRT has no speakers yet


def test_bad_srt_rejected(client):
    key = make_project(client)
    r = client.post(f"/api/projects/{key}/upload/srt",
                    files={"file": ("subs.srt", "not an srt", "text/plain")})
    assert r.status_code == 400


# -------------------------------- CSV -----------------------------------------

def test_csv_roundtrip_with_speakers(client):
    key = make_project(client)
    upload_srt(client, key)
    csv_text = client.get(f"/api/projects/{key}/csv").text
    assert csv_text.splitlines()[0] == "speaker,line_no,start,end,text"
    # diarize: fill speakers in the exported CSV, save via editor endpoint
    edited = csv_text.replace(",1,", "Obi Wan,1,", 1).replace(
        ",2,", "Grievous,2,", 1)
    r = client.put(f"/api/projects/{key}/csv", content=edited,
                   headers={"content-type": "text/csv"})
    assert r.status_code == 200, r.text
    from backend import store
    ps = store.load(key)
    assert set(ps.data["speakers"]) == {"Obi_Wan", "Grievous"}
    assert ps.data["speakers"]["Obi_Wan"]["display_name"] == "Obi Wan"
    assert ps.line(3)["speaker"] is None  # music row left blank
    # export again: display names round-trip
    again = client.get(f"/api/projects/{key}/csv").text
    assert "Obi Wan,1," in again and "Grievous,2," in again


def test_csv_save_resets_progress_and_derived_dirs(client):
    key = make_project(client)
    upload_srt(client, key)
    csv_text = client.get(f"/api/projects/{key}/csv").text
    edited = csv_text.replace(",1,", "A,1,", 1)
    client.put(f"/api/projects/{key}/csv", content=edited,
               headers={"content-type": "text/csv"})
    from backend import store
    # simulate progress + derived audio
    with store.locked(key):
        ps = store.load(key)
        ps.mark_generating([1]); ps.mark_success(1, 1.0)
        ps.apply_line_edits([{"line_no": 2, "field": "tolerance", "value": 1.4}])
        ps.save()
    d = store.project_dir(key)
    for sub in ("speaker_clips", "refs", "dub_work", "separated"):
        (d / sub).mkdir(exist_ok=True)
        (d / sub / "marker").touch()
    # editor save again -> full reset
    client.put(f"/api/projects/{key}/csv", content=edited,
               headers={"content-type": "text/csv"})
    ps = store.load(key)
    assert all(l["badge"] == "never" for l in ps.data["lines"])
    assert all(l["overrides"] == {} for l in ps.data["lines"])
    for sub in ("speaker_clips", "refs", "dub_work"):
        assert not (d / sub).exists()
    assert (d / "separated" / "marker").exists()  # Demucs output survives


@pytest.mark.parametrize("bad,why", [
    ("speaker,line_no\nA,1", "missing columns"),
    ("speaker,line_no,start,end,text\nA,1,2.0,1.0,hi", "end <= start"),
    ("speaker,line_no,start,end,text\nA,x,0,1,hi", "bad number"),
    ("speaker,line_no,start,end,text\nA,1,0,1,hi\nB,1,2,3,yo", "duplicate"),
    ("speaker,line_no,start,end,text\n", "no data rows"),
])
def test_bad_csv_rejected(client, bad, why):
    key = make_project(client)
    r = client.put(f"/api/projects/{key}/csv", content=bad,
                   headers={"content-type": "text/csv"})
    assert r.status_code == 400
    assert why.split()[0] in r.json()["detail"]


def test_csv_file_upload(client):
    key = make_project(client)
    csv_body = ("speaker,line_no,start,end,text\n"
                "Alice,1,0.0,2.0,Hi there\n"
                ",2,3.0,4.0,(door slams)\n")
    r = client.post(f"/api/projects/{key}/upload/csv",
                    files={"file": ("segments.csv", csv_body, "text/csv")})
    assert r.status_code == 200
    assert r.json()["n_lines"] == 2
    assert r.json()["meta"]["stats"]["n_dubbable"] == 1
