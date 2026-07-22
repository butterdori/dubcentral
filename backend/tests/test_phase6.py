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
    monkeypatch.setattr(config, "CRISPASR_TRANSPORT", "http")  # pure-HTTP assertions
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

    def synth(*, text, prompt_text, ref_path, backend, out_path,
              force_cpu=False, instruct_text=""):
        parts = out_path.stem.split("_")
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        assert ref_path.exists()
        calls.append({"line": n, "text": text, "prompt_text": prompt_text,
                      "backend": backend, "force_cpu": force_cpu,
                      "instruct_text": instruct_text})
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

def test_crispasr_speed_field_does_not_overlap_cosyvoice(client, video, fake_demucs,
                                                          fake_crisp):
    """crispasr uses its OWN field name for backend selection
    (crispasr_backend, not speed) specifically so switching to/from
    cosyvoice3 can't cross-clobber a numeric speed override with a backend
    name or vice versa (see project_state.ENGINE_FIELDS comment).
    instruct_text IS intentionally shared between the two -- same
    conceptual field (free-text style/language instruction), and CrispASR's
    CLI has native --instruct support -- so switching engines correctly
    clears it either way (a stale instruction for a different engine isn't
    something worth preserving)."""
    from backend.models.project_state import ENGINE_FIELDS
    assert "speed" not in ENGINE_FIELDS["crispasr"]
    assert "crispasr_backend" not in ENGINE_FIELDS["cosyvoice3"]
    assert "instruct_text" in ENGINE_FIELDS["crispasr"]
    assert "instruct_text" in ENGINE_FIELDS["cosyvoice3"]   # intentionally shared

    # behaviorally: switching away from cosyvoice3 correctly clears
    # cosyvoice3's OWN fields (expected -- same as any other engine switch),
    # and switching to/from crispasr afterward only ever touches its own
    # fields, never resurrecting a stale "speed" override.
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


def test_crispasr_instruct_text_reaches_synthesis(client, video, fake_demucs,
                                                   fake_crisp):
    """The whole point of adding instruct_text to crispasr: it actually
    reaches synthesize_line (both dub runs and test-speech), where it maps
    to CrispASR's --instruct / "instructions"."""
    key = make_project(client, video)
    client.patch(f"/api/projects/{key}/dub/lines", json={"edits": [
        {"line_no": 1, "field": "instruct_text", "value": "Speak in Korean."}]})
    run_dub(client, key, mode="selected", line_nos=[1])
    call = next(c for c in fake_crisp if c.get("line") == 1)
    assert call["instruct_text"] == "Speak in Korean."


def test_grid_engine_fields_for_crispasr(client, video, fake_demucs):
    key = make_project(client, video)
    g = grid(client, key)
    assert g["engine"] == "crispasr"
    assert g["engine_fields"] == ["crispasr_backend", "instruct_text"]


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


# ------------------------ transport modes + auto-fallback -----------------------

@pytest.fixture(autouse=True)
def _reset_forced_cli():
    """_FORCED_CLI is module-level sticky state (by design, per process) --
    reset around every test so a fallback in one test can't poison others."""
    from backend.pipeline import tts_crispasr
    tts_crispasr._FORCED_CLI = False
    yield
    tts_crispasr._FORCED_CLI = False


def _wav_bytes(tmp_path):
    import wave
    p = tmp_path / "real.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return p.read_bytes()


def test_cli_transport_command_shape(tmp_path, monkeypatch):
    """cli mode: correct binary invocation incl. consent/disclosure/
    watermark flags, ref-text, and --no-gpu under force_cpu."""
    import subprocess as sp
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_TRANSPORT", "cli")
    monkeypatch.setattr(config, "CRISPASR_BIN", "/opt/crispasr/bin/crispasr")

    import pathlib as pl
    wav = _wav_bytes(tmp_path)
    calls = []
    real_run = sp.run
    def fake_run(cmd, *a, **kw):
        if cmd[0] != config.CRISPASR_BIN:      # ffprobe etc: pass through
            return real_run(cmd, *a, **kw)
        calls.append((cmd, kw.get("timeout")))
        # emulate the CLI writing its --tts-output file
        pl.Path(cmd[cmd.index("--tts-output") + 1]).write_bytes(wav)
        class P: returncode = 0; stdout = "ok"; stderr = ""
        return P()
    monkeypatch.setattr(sp, "run", fake_run)

    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    out = tmp_path / "o" / "out.wav"
    dur = tts_crispasr.synthesize_line(
        text="hello", prompt_text="こんにちは", ref_path=ref,
        backend="cosyvoice3-tts", out_path=out, force_cpu=True)
    assert dur > 0
    cmd, timeout = calls[0]
    assert cmd[0] == "/opt/crispasr/bin/crispasr"
    assert cmd[cmd.index("--backend") + 1] == "cosyvoice3-tts"
    assert cmd[cmd.index("--voice") + 1] == str(ref)
    assert cmd[cmd.index("--ref-text") + 1] == "こんにちは"
    assert cmd[cmd.index("--tts") + 1] == "hello"
    assert "--i-have-rights" in cmd
    assert "--no-watermark" in cmd
    assert "--no-spoken-disclaimer" in cmd   # CRISPASR_SPOKEN_DISCLAIMER=False
    assert "--no-gpu" in cmd                  # force_cpu WORKS in cli mode
    assert timeout == config.CRISPASR_CLI_TIMEOUT_S

    # force_cpu=False -> no --no-gpu
    tts_crispasr.synthesize_line(
        text="hi", prompt_text="", ref_path=ref,
        backend="cosyvoice3-tts", out_path=out, force_cpu=False)
    cmd2, _ = calls[1]
    assert "--no-gpu" not in cmd2
    assert "--ref-text" not in cmd2   # empty transcript omitted


def test_auto_falls_back_on_synthesis_failed_and_sticks(tmp_path, monkeypatch):
    """auto mode: the confirmed 500 synthesis_failed signature flips to the
    CLI and STAYS there -- the second call never touches HTTP again."""
    import subprocess as sp
    import pathlib as pl
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_TRANSPORT", "auto")
    monkeypatch.setattr(config, "CRISPASR_VOICE_DIR", tmp_path / "voices")

    http_calls = []
    monkeypatch.setattr("requests.post", lambda *a, **kw: (
        http_calls.append(1),
        FakeResponse(500, text='{"error": {"code": "synthesis_failed"}}'))[1])

    wav = _wav_bytes(tmp_path)
    cli_calls = []
    real_run = sp.run
    def fake_run(cmd, *a, **kw):
        if cmd[0] != config.CRISPASR_BIN:      # ffprobe etc: pass through
            return real_run(cmd, *a, **kw)
        cli_calls.append(cmd)
        pl.Path(cmd[cmd.index("--tts-output") + 1]).write_bytes(wav)
        class P: returncode = 0; stdout = ""; stderr = ""
        return P()
    monkeypatch.setattr(sp, "run", fake_run)

    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    out = tmp_path / "out.wav"
    kw = dict(text="t", prompt_text="p", ref_path=ref,
              backend="cosyvoice3-tts", out_path=out)
    assert tts_crispasr.synthesize_line(**kw) > 0
    assert len(http_calls) == 1 and len(cli_calls) == 1   # tried http, fell back

    assert tts_crispasr.synthesize_line(**kw) > 0
    assert len(http_calls) == 1 and len(cli_calls) == 2   # sticky: straight to CLI


def test_auto_does_not_fall_back_on_other_errors(tmp_path, monkeypatch):
    """A 500 that ISN'T the voice-dir signature (e.g. a real OOM) surfaces
    as-is -- falling back to the CLI there would just OOM harder."""
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_TRANSPORT", "auto")
    monkeypatch.setattr(config, "CRISPASR_VOICE_DIR", tmp_path / "voices")
    monkeypatch.setattr("requests.post",
        lambda *a, **kw: FakeResponse(500, text="CUDA out of memory"))
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    with pytest.raises(tts_crispasr.CrispASRError, match="out of memory"):
        tts_crispasr.synthesize_line(text="t", prompt_text="", ref_path=ref,
                                     backend="cosyvoice3-tts",
                                     out_path=tmp_path / "out.wav")
    assert tts_crispasr._FORCED_CLI is False   # not sticky for this


def test_cli_failure_surfaces_output_tail(tmp_path, monkeypatch):
    import subprocess as sp
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_TRANSPORT", "cli")
    def fake_run(cmd, capture_output, text, timeout):
        class P:
            returncode = 3
            stdout = ""
            stderr = "line1\nsomething exploded: no such tensor\n"
        return P()
    monkeypatch.setattr(sp, "run", fake_run)
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    with pytest.raises(tts_crispasr.CrispASRError,
                       match="something exploded"):
        tts_crispasr.synthesize_line(text="t", prompt_text="", ref_path=ref,
                                     backend="cosyvoice3-tts",
                                     out_path=tmp_path / "out.wav")


def test_instruct_text_maps_to_instructions_over_http(tmp_path, monkeypatch):
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_TRANSPORT", "http")
    monkeypatch.setattr(config, "CRISPASR_VOICE_DIR", tmp_path / "voices")
    import wave
    real_wav = tmp_path / "real.wav"
    with wave.open(str(real_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    calls = []
    monkeypatch.setattr("requests.post", lambda url, json, timeout: (
        calls.append(json), FakeResponse(200, content=real_wav.read_bytes()))[1])
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    tts_crispasr.synthesize_line(
        text="t", prompt_text="", ref_path=ref, backend="cosyvoice3-tts",
        out_path=tmp_path / "out.wav", instruct_text="Speak in Korean.")
    assert calls[0]["instructions"] == "Speak in Korean."


def test_instruct_text_maps_to_instruct_flag_over_cli(tmp_path, monkeypatch):
    import subprocess as sp
    import wave
    from backend import config
    from backend.pipeline import tts_crispasr
    monkeypatch.setattr(config, "CRISPASR_TRANSPORT", "cli")
    real_wav = tmp_path / "real.wav"
    with wave.open(str(real_wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    calls = []
    real_run = sp.run
    def fake_run(cmd, *a, **kw):
        if cmd[0] != config.CRISPASR_BIN:
            return real_run(cmd, *a, **kw)
        calls.append(cmd)
        import pathlib as pl
        pl.Path(cmd[cmd.index("--tts-output") + 1]).write_bytes(real_wav.read_bytes())
        class P: returncode = 0; stdout = ""; stderr = ""
        return P()
    monkeypatch.setattr(sp, "run", fake_run)
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    dur = tts_crispasr.synthesize_line(
        text="t", prompt_text="", ref_path=ref, backend="cosyvoice3-tts",
        out_path=tmp_path / "out.wav", instruct_text="Speak in Korean.")
    assert dur > 0
    cmd = calls[0]
    assert cmd[cmd.index("--instruct") + 1] == "Speak in Korean."
