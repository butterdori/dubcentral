"""CrispASR engine
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..models.project_state import ProjectState
from . import timefit, tts_cosyvoice


class CrispASRError(RuntimeError):
    pass

select_reference = tts_cosyvoice.select_reference


def synthesize_line(*, text: str, prompt_text: str, ref_path: Path,
                    backend: str, out_path: Path,
                    force_cpu: bool = False) -> float:

    import requests

    url = config.CRISPASR_CPU_URL if force_cpu else config.CRISPASR_URL
    payload = {
        "model": backend,
        "input": text,
        "voice": str(ref_path),
        "response_format": "wav",
        "consent_attestation": config.CRISPASR_CONSENT_ATTESTATION,
        "spoken_disclaimer": config.CRISPASR_SPOKEN_DISCLAIMER,
    }
    if prompt_text:
        payload["ref_text"] = prompt_text

    try:
        resp = requests.post(f"{url}/v1/audio/speech", json=payload,
                             timeout=config.CRISPASR_TIMEOUT_S)
    except requests.exceptions.ConnectionError as e:
        raise CrispASRError(
            f"could not reach CrispASR at {url} — is the server running? "
            f"({e})") from None
    except requests.exceptions.Timeout:
        raise CrispASRError(
            f"CrispASR at {url} did not respond within "
            f"{config.CRISPASR_TIMEOUT_S}s") from None

    if resp.status_code != 200:
        body = resp.text[:500]
        raise CrispASRError(
            f"CrispASR returned {resp.status_code} for backend={backend!r}: "
            f"{body}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    if len(resp.content) == 0:
        raise CrispASRError(
            f"CrispASR returned an empty response for backend={backend!r} "
            "(200 OK, but no audio bytes)")
    return timefit.probe_duration(out_path)
