"""CrispASR engine
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..models.project_state import ProjectState
from . import timefit, tts_cosyvoice


class CrispASRError(RuntimeError):
    pass


class CrispASRConnectionError(CrispASRError):
    """Server unreachable — distinct so auto mode can fall back to CLI."""


class CrispASRHTTPError(CrispASRError):
    """Non-2xx from the server, carrying enough to classify the failure."""
    def __init__(self, status_code: int, body: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

select_reference = tts_cosyvoice.select_reference


def _stage_voice(ref_path: Path, prompt_text: str) -> str:
    """CrispASR's HTTP `voice` field."""
    import hashlib
    import shutil

    name = "ref_" + hashlib.sha1(str(ref_path.resolve()).encode()).hexdigest()[:16]
    voice_dir = config.CRISPASR_VOICE_DIR
    voice_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ref_path, voice_dir / f"{name}.wav")
    if prompt_text:
        (voice_dir / f"{name}.txt").write_text(prompt_text, encoding="utf-8")
    else:
        (voice_dir / f"{name}.txt").unlink(missing_ok=True)
    return name


_FORCED_CLI = False


def _should_fall_back(status_code: int | None, body: str) -> bool:
    if status_code == 400 and "invalid_voice" in body:
        return True
    if status_code == 500 and "synthesis_failed" in body:
        return True
    return False


def _synthesize_http(*, text: str, prompt_text: str, ref_path: Path,
                     backend: str, out_path: Path, force_cpu: bool,
                     instruct_text: str = "") -> float:
    import requests

    url = config.CRISPASR_CPU_URL if force_cpu else config.CRISPASR_URL
    voice_name = _stage_voice(ref_path, prompt_text)
    payload = {
        "model": backend,
        "input": text,
        "voice": voice_name,
        "response_format": "wav",
        "consent_attestation": config.CRISPASR_CONSENT_ATTESTATION,
        "spoken_disclaimer": config.CRISPASR_SPOKEN_DISCLAIMER,
    }
    if prompt_text:
        payload["ref_text"] = prompt_text
    if instruct_text:
        payload["instructions"] = instruct_text

    try:
        resp = requests.post(f"{url}/v1/audio/speech", json=payload,
                             timeout=config.CRISPASR_TIMEOUT_S)
    except requests.exceptions.ConnectionError as e:
        raise CrispASRConnectionError(
            f"could not reach CrispASR at {url} — is the server running? "
            f"({e})") from None
    except requests.exceptions.Timeout:
        raise CrispASRError(
            f"CrispASR at {url} did not respond within "
            f"{config.CRISPASR_TIMEOUT_S}s") from None

    if resp.status_code != 200:
        body = resp.text[:500]
        raise CrispASRHTTPError(resp.status_code, body,
            f"CrispASR returned {resp.status_code} for backend={backend!r}: "
            f"{body}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    if len(resp.content) == 0:
        raise CrispASRError(
            f"CrispASR returned an empty response for backend={backend!r} "
            "(200 OK, but no audio bytes)")
    return timefit.probe_duration(out_path)


def _synthesize_cli(*, text: str, prompt_text: str, ref_path: Path,
                    backend: str, out_path: Path, force_cpu: bool,
                    instruct_text: str = "") -> float:
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [config.CRISPASR_BIN, "--backend", backend, "-m", "auto",
           "--voice", str(ref_path),
           "--i-have-rights",
           "--no-watermark",
           "--tts", text,
           "--tts-output", str(out_path)]
    if not config.CRISPASR_SPOKEN_DISCLAIMER:
        cmd.append("--no-spoken-disclaimer")
    if prompt_text:
        cmd += ["--ref-text", prompt_text]
    if instruct_text:
        cmd += ["--instruct", instruct_text]
    if force_cpu:
        cmd.append("--no-gpu")
    if config.CRISPASR_FREQUENCY_PENALTY:
        cmd += ["--frequency-penalty", config.CRISPASR_FREQUENCY_PENALTY]
    if config.CRISPASR_MAX_NEW_TOKENS:
        cmd += ["-n", config.CRISPASR_MAX_NEW_TOKENS]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=config.CRISPASR_CLI_TIMEOUT_S)
    except FileNotFoundError:
        raise CrispASRError(
            f"crispasr binary not found ({config.CRISPASR_BIN!r}) — set "
            "CRISPASR_BIN to its full path, or put it on PATH") from None
    except subprocess.TimeoutExpired:
        raise CrispASRError(
            f"crispasr CLI did not finish within "
            f"{config.CRISPASR_CLI_TIMEOUT_S}s") from None

    if proc.returncode != 0 or not out_path.exists() \
            or out_path.stat().st_size == 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise CrispASRError(
            f"crispasr CLI failed (exit {proc.returncode}) for "
            f"backend={backend!r}: " + " | ".join(tail))
    return timefit.probe_duration(out_path)


def synthesize_line(*, text: str, prompt_text: str, ref_path: Path,
                    backend: str, out_path: Path,
                    force_cpu: bool = False, instruct_text: str = "") -> float:
    global _FORCED_CLI
    kwargs = dict(text=text, prompt_text=prompt_text, ref_path=ref_path,
                  backend=backend, out_path=out_path, force_cpu=force_cpu,
                  instruct_text=instruct_text)

    mode = config.CRISPASR_TRANSPORT
    if mode == "cli":
        return _synthesize_cli(**kwargs)
    if mode == "http":
        return _synthesize_http(**kwargs)

    # auto: sticky CLI once HTTP has proven itself unable
    if _FORCED_CLI:
        return _synthesize_cli(**kwargs)
    try:
        return _synthesize_http(**kwargs)
    except CrispASRHTTPError as e:
        if not _should_fall_back(e.status_code, e.body):
            raise
        print(f"[crispasr] HTTP transport can't clone from voice-dir "
              f"({e.status_code}) — falling back to CLI for the rest of "
              f"this session", flush=True)
        _FORCED_CLI = True
        return _synthesize_cli(**kwargs)
    except CrispASRConnectionError:
        print("[crispasr] server unreachable — falling back to CLI for the "
              "rest of this session", flush=True)
        _FORCED_CLI = True
        return _synthesize_cli(**kwargs)
