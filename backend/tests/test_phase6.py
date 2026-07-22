"""Phase 6 tests: the CrispASR engine.

Two layers, tested separately:
  1. tts_crispasr.synthesize_line's HTTP behavior in isolation (fake
     requests.post) -- URL selection, payload shape, error translation.
     This is the layer whose exact wire format is UNVERIFIED against a
     real CrispASR instance (see the module's own docstring) -- these
     tests pin down what THIS APP sends/expects, not what CrispASR
     actually wants, which is exactly what crispasr_probe.py is for.
  2. Dispatch through the dub loop (fake synthesize_line, same pattern as
     the chatterbox/cosyvoice3 fakes) -- confirms routing, reference-
     selection reuse, and badge/engine-switch semantics end-to-end.
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
"""

ORIG_SRT = """\
1
00:00:00,200 --> 00:00:04,200
こんにちは、そこの人。

2
00:00:04,600 --> 00:00:05,400
やあ。
"""

CSV = ("speaker,line_no,start,end,text\n"
       "A,1,0.200,4.200,Hello there.\n"
       "A,2,4.600,5.400,Hi.\n")


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
    p = tmp_path_factory.mktemp("m6") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=7:size=160x120:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=7",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


@pytest.fixture()
def fake_demucs(monkeypatch):
    from backend.pipeline import demucs_runner

    def fake(video, sep_root):
        from backend import config
        stem = sep_root / config.DEMUCS_MODEL / "source"
        stem.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn",
                        str(stem / "vocals.wav")], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(stem / "vocals.wav"),
                        "-af", "volume=0", str(stem / "no_vocals.wav")],
                       check=True, capture_output=True)
    monkeypatch.setattr(demucs_runner, "_run_cli", fake)


def make_project(client, video, engine="crispasr"):
    key = client.post("/api/projects", json={"name": "P6"}).json()["key"]
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    client.put(f"/api/projects/{key}/csv", content=CSV,
               headers={"content-type": "text/csv"})
    r = client.put(f"/api/projects/{key}/engine", json={"engine": engine})
    assert r.status_code == 200, r.text
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


# ---------------------------- HTTP layer (isolated) ----------------------------

class FakeResponse:
    def __init__(self, status_code=200, content=b"RIFF....WAVEfake", text=""):
        self.status_code = status_code
        self.content = content
        self.text = text or content.decode(errors="replace")


def test_http_payload_shape_and_url_selection(tmp_path, monkeypatch):
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_URL", "http://gpu-host:9")
    monkeypatch.setattr(config, "CRISPASR_CPU_URL", "http://cpu-host:9")
    voice_dir = tmp_path / "voices"
    monkeypatch.setattr(config, "CRISPASR_VOICE_DIR", voice_dir)

    calls = []
    # make the fake wav a real (tiny) valid wav so probe_duration doesn't choke
    import wave
    real_wav = tmp_path / "real.wav"
    with wave.open(str(real_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    monkeypatch.setattr("requests.post",
        lambda url, json, timeout: (calls.append((url, json, timeout)),
                                    FakeResponse(200, content=real_wav.read_bytes()))[1])

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    out = tmp_path / "out.wav"
    dur = tts_crispasr.synthesize_line(
        text="hello", prompt_text="こんにちは", ref_path=ref,
        backend="cosyvoice3-tts", out_path=out, force_cpu=False)
    assert dur > 0
    url, payload, timeout = calls[0]
    assert url == "http://gpu-host:9/v1/audio/speech"
    assert payload["model"] == "cosyvoice3-tts"
    assert payload["input"] == "hello"
    # `voice` must be a bare name -- CrispASR's HTTP server rejects any
    # path separator, '..', leading '/' or '~' in this field (confirmed:
    # 400 invalid_voice against a real server)
    voice_name = payload["voice"]
    assert "/" not in voice_name and ".." not in voice_name and not voice_name.startswith("~")
    assert (voice_dir / f"{voice_name}.wav").read_bytes() == ref.read_bytes()
    assert (voice_dir / f"{voice_name}.txt").read_text(encoding="utf-8") == "こんにちは"
    assert payload["ref_text"] == "こんにちは"   # confirmed field name (not prompt_text)
    # consent + disclosure sent on every request (see config.py + VOICE_CLONING_NOTICE.md)
    assert payload["consent_attestation"] == config.CRISPASR_CONSENT_ATTESTATION
    assert payload["spoken_disclaimer"] is False
    assert timeout == config.CRISPASR_TIMEOUT_S

    calls.clear()
    tts_crispasr.synthesize_line(
        text="hi", prompt_text="", ref_path=ref, backend="cosyvoice3-tts",
        out_path=out, force_cpu=True)
    assert calls[0][0] == "http://cpu-host:9/v1/audio/speech"
    assert "/" not in calls[0][1]["voice"]   # still a bare staged name
    assert "ref_text" not in calls[0][1]   # empty transcript omitted
    assert not (voice_dir / f"{calls[0][1]['voice']}.txt").exists()  # no stale .txt either
    # consent/disclosure still sent even with no ref_text (voice is still a .wav)
    assert calls[0][1]["consent_attestation"] == config.CRISPASR_CONSENT_ATTESTATION
    assert calls[0][1]["spoken_disclaimer"] is False


def test_stage_voice_name_is_deterministic_and_safe(tmp_path, monkeypatch):
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_VOICE_DIR", tmp_path / "voices")
    ref = tmp_path / "clip.wav"
    ref.write_bytes(b"abc")
    n1 = tts_crispasr._stage_voice(ref, "hello")
    n2 = tts_crispasr._stage_voice(ref, "hello")
    assert n1 == n2   # same source path -> same staged name every time
    assert all(c.isalnum() or c == "_" for c in n1)   # no separators possible


def test_http_error_translation(tmp_path, monkeypatch):
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_VOICE_DIR", tmp_path / "voices")
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    out = tmp_path / "out.wav"

    monkeypatch.setattr("requests.post", lambda *a, **kw: FakeResponse(500, text="CUDA out of memory"))
    with pytest.raises(tts_crispasr.CrispASRError, match="500"):
        tts_crispasr.synthesize_line(text="t", prompt_text="", ref_path=ref,
                                     backend="cosyvoice3-tts", out_path=out)

    monkeypatch.setattr("requests.post", lambda *a, **kw: FakeResponse(200, content=b""))
    with pytest.raises(tts_crispasr.CrispASRError, match="empty response"):
        tts_crispasr.synthesize_line(text="t", prompt_text="", ref_path=ref,
                                     backend="cosyvoice3-tts", out_path=out)

    import requests
    def raise_conn(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr("requests.post", raise_conn)
    with pytest.raises(tts_crispasr.CrispASRError, match="is the server running"):
        tts_crispasr.synthesize_line(text="t", prompt_text="", ref_path=ref,
                                     backend="cosyvoice3-tts", out_path=out)


# -------------------------------- dispatch e2e ---------------------------------

@pytest.fixture()
def fake_crisp(monkeypatch):
    from backend.pipeline import tts_crispasr
    calls = []

    def synth(*, text, prompt_text, ref_path, backend, out_path, force_cpu=False):
        parts = out_path.stem.split("_")
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        assert ref_path.exists()
        calls.append({"line": n, "text": text, "prompt_text": prompt_text,
                      "backend": backend, "force_cpu": force_cpu})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=600:duration=1.0",
                        str(out_path)], check=True, capture_output=True)
        return 1.0
    monkeypatch.setattr(tts_crispasr, "synthesize_line", synth)
    return calls


def test_crispasr_dub_run_dispatch(client, video, fake_demucs, fake_crisp):
    key = make_project(client, video)   # crispasr default in this test file
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "crispasr_backend", "value": "chatterbox"}]})
    job = run_dub(client, key)
    assert job["result"]["engine"] == "crispasr"
    assert job["result"]["n_ok"] == 2
    calls = {c["line"]: c for c in fake_crisp}
    assert calls[1]["backend"] == "chatterbox"       # per-line override
    assert calls[2]["backend"] == "cosyvoice3-tts"    # project default
    # reference reused from the shared cosyvoice3 pin/auto-pick machinery
    assert calls[1]["prompt_text"] in ("こんにちは、そこの人。", "やあ。")
    assert all(r["badge"] == "clean" for r in grid(client, key)["rows"])


def test_crispasr_needs_original_text(client, video, fake_demucs, fake_crisp):
    """Same reference requirement as cosyvoice3 (shared machinery) -- fails
    per-line with the same actionable message when it's missing."""
    key = client.post("/api/projects", json={"name": "P6b"}).json()["key"]
    client.post(f"/api/projects/{key}/upload/video",
                files={"file": ("c.mp4", video.read_bytes(), "video/mp4")})
    client.post(f"/api/projects/{key}/upload/srt",
                files={"file": ("s.srt", SRT, "text/plain")})
    client.put(f"/api/projects/{key}/csv", content=CSV,
               headers={"content-type": "text/csv"})
    client.put(f"/api/projects/{key}/engine", json={"engine": "crispasr"})
    client.post(f"/api/projects/{key}/extract_clips")
    assert job_queue.wait_idle(30)
    job = run_dub(client, key)
    assert job["result"]["n_failed"] == 2
    err = grid(client, key)["rows"][0]["error"]
    assert "original-language SRT" in err


def test_crispasr_backend_edit_is_full_tier(client, video, fake_demucs, fake_crisp):
    key = make_project(client, video)
    run_dub(client, key)
    deltas = client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "crispasr_backend", "value": "f5-tts"}]}
    ).json()["deltas"]
    assert deltas == {"1": "needs_full"}


# --------------------------- engine switch field isolation ---------------------

def test_crispasr_fields_dont_overlap_cosyvoice(client, video, fake_demucs,
                                                fake_crisp):
    """crispasr deliberately uses its OWN field name (crispasr_backend),
    not speed/instruct_text, specifically so switching to/from cosyvoice3
    can't cross-clobber the other's overrides (see project_state.
    ENGINE_FIELDS comment). Guard the design decision directly..."""
    from backend.models.project_state import ENGINE_FIELDS
    assert not set(ENGINE_FIELDS["crispasr"]) & set(ENGINE_FIELDS["cosyvoice3"])

    # ...and behaviorally: switching away from cosyvoice3 correctly clears
    # cosyvoice3's OWN fields (expected -- same as any other engine switch),
    # and switching to/from crispasr afterward only ever touches
    # crispasr_backend, never speed/instruct_text again.
    key = make_project(client, video, engine="cosyvoice3")
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "speed", "value": 1.3}]})
    client.put(f"/api/projects/{key}/engine", json={"engine": "crispasr"})
    from backend import store
    assert "speed" not in store.load(key).line(1)["overrides"]   # expected: cleared

    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "crispasr_backend", "value": "vibevoice"}]})
    client.put(f"/api/projects/{key}/engine", json={"engine": "cosyvoice3"})
    ps = store.load(key)
    assert "crispasr_backend" not in ps.line(1)["overrides"]
    assert "speed" not in ps.line(1)["overrides"]   # not resurrected either


def test_grid_engine_fields_for_crispasr(client, video, fake_demucs):
    key = make_project(client, video)
    g = grid(client, key)
    assert g["engine"] == "crispasr"
    assert g["engine_fields"] == ["crispasr_backend"]


def test_pin_renew_ui_reused_under_crispasr(client, video, fake_demucs, fake_crisp):
    """The Phase 5 multi-pin/Renew reference machinery works identically
    under crispasr -- confirms the reuse actually functions, not just
    that the frontend gates were widened."""
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/speakers/A/reference_pins",
               json={"line_nos": [1, 2]})
    r = client.post(f"/api/projects/{key}/renew_reference")
    assert r.status_code == 202
    assert job_queue.wait_idle(30)
    assert client.get("/api/jobs/current").json()["job"]["status"] == "done"

    run_dub(client, key)
    calls = {c["line"]: c for c in fake_crisp}
    # both lines got the CONCATENATED transcript, confirming crispasr's
    # synthesis actually used the renewed multi-pin reference
    assert calls[1]["prompt_text"] == calls[2]["prompt_text"]
    assert calls[1]["prompt_text"] == "こんにちは、そこの人。 やあ。"


# ------------------------------ test speech --------------------------------

def test_test_speech_generate_play_download(client, video, fake_demucs, fake_crisp):
    key = make_project(client, video)   # crispasr, default backend
    assert client.get(f"/api/projects/{key}/speakers/A/test_speech/audio").status_code == 404

    r = client.post(f"/api/projects/{key}/speakers/A/test_speech",
                    json={"text": "This is a test."})
    assert r.status_code == 202, r.text
    assert job_queue.wait_idle(30)
    snap = client.get("/api/jobs/current").json()["job"]
    assert snap["status"] == "done", snap
    assert snap["result"]["speaker"] == "A"
    assert snap["result"]["engine"] == "crispasr"

    call = fake_crisp[-1]
    assert call["text"] == "This is a test."
    assert call["backend"] == "cosyvoice3-tts"   # project default

    sp = client.get(f"/api/projects/{key}/speakers").json()["speakers"]
    a = next(s for s in sp if s["key"] == "A")
    assert a["has_test_speech"] is True

    r = client.get(f"/api/projects/{key}/speakers/A/test_speech/audio")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert "attachment" not in r.headers.get("content-disposition", "")

    r = client.get(f"/api/projects/{key}/speakers/A/test_speech/download")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert f"{key}_A_test.wav" in r.headers["content-disposition"]


def test_test_speech_uses_committed_pin_reference(client, video, fake_demucs, fake_crisp):
    """Confirms the test-speech path reuses the SAME reference resolution
    as a real dub (committed multi-pin concat, not just auto-pick)."""
    key = make_project(client, video)
    client.put(f"/api/projects/{key}/speakers/A/reference_pins",
               json={"line_nos": [1, 2]})
    r = client.post(f"/api/projects/{key}/renew_reference")
    assert job_queue.wait_idle(30)

    client.post(f"/api/projects/{key}/speakers/A/test_speech",
               json={"text": "concat check"})
    assert job_queue.wait_idle(30)
    call = fake_crisp[-1]
    assert call["prompt_text"] == "こんにちは、そこの人。 やあ。"


def test_test_speech_bad_speaker_and_busy(client, video, fake_demucs):
    key = make_project(client, video)
    assert client.post(f"/api/projects/{key}/speakers/Ghost/test_speech",
                       json={"text": "hi"}).status_code == 404

    import threading
    release = threading.Event()
    job_queue.submit("hold", key, lambda job: release.wait(5) and None)
    r = client.post(f"/api/projects/{key}/speakers/A/test_speech", json={"text": "hi"})
    assert r.status_code == 409
    release.set()
    job_queue.wait_idle()
