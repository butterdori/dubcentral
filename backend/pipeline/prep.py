"""Port of the CLI's cmd_prep + read_csv: SRT -> segment rows, and the
segments.csv interchange format (import/export only — project.json is the
runtime source of truth).

CSV columns match the CLI exactly (speaker,line_no,start,end,text) so existing
hand-diarized CSVs from the CLI workflow import unchanged.
"""
from __future__ import annotations

import csv
import io
from typing import Any

import srt as srtlib


class PrepError(ValueError):
    """Validation failure in an uploaded SRT/CSV — surfaced as HTTP 400."""


REQUIRED_COLUMNS = {"speaker", "line_no", "start", "end", "text"}


def parse_srt_bytes(data: bytes) -> list[dict[str, Any]]:
    """SRT file -> segment rows with empty speaker column (manual diarization
    happens in the CSV editor / Dub Center afterwards)."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise PrepError(f"SRT is not UTF-8: {e}") from None
    try:
        subs = list(srtlib.parse(text))
    except Exception as e:  # srt raises assorted exception types
        raise PrepError(f"SRT parse failed: {e}") from None
    if not subs:
        raise PrepError("no subtitles parsed from the SRT")
    rows = []
    for sub in subs:
        rows.append({
            "speaker": "",
            "line_no": sub.index,
            "start": round(sub.start.total_seconds(), 3),
            "end": round(sub.end.total_seconds(), 3),
            # single-line the text so the CSV stays easy to edit (CLI behavior)
            "text": " ".join(sub.content.split()),
        })
    return rows


def csv_text_to_rows(text: str) -> list[dict[str, Any]]:
    """Parse segments.csv content (editor save or file re-upload). Same
    required columns as the CLI's read_csv; per-row errors report the row."""
    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise PrepError(
            f"CSV missing columns; need {sorted(REQUIRED_COLUMNS)}, "
            f"got {reader.fieldnames}")
    rows = []
    seen: set[int] = set()
    for i, r in enumerate(reader, start=2):  # data starts on file line 2
        try:
            line_no = int(r["line_no"])
            start = float(r["start"])
            end = float(r["end"])
        except (TypeError, ValueError) as e:
            raise PrepError(f"CSV row {i}: bad number ({e})") from None
        if line_no in seen:
            raise PrepError(f"CSV row {i}: duplicate line_no {line_no}")
        seen.add(line_no)
        if end <= start:
            raise PrepError(f"CSV row {i} (line {line_no}): end <= start")
        rows.append({
            "speaker": (r["speaker"] or "").strip(),
            "line_no": line_no,
            "start": start,
            "end": end,
            "text": (r["text"] or "").strip(),
        })
    if not rows:
        raise PrepError("CSV has no data rows")
    return rows


def state_to_srt_text(ps) -> str:
    """Export the CURRENT line table as an SRT (timings + text as edited).
    Speaker assignments are not encoded — SRT has no speaker field, and
    prefixing names into the text would corrupt a re-import; speakers
    round-trip through segments.csv instead."""
    from datetime import timedelta
    subs = [
        srtlib.Subtitle(index=l["line_no"],
                        start=timedelta(seconds=l["start"]),
                        end=timedelta(seconds=l["end"]),
                        content=l["text"])
        for l in ps.data["lines"]
    ]
    return srtlib.compose(subs)


def state_to_csv_text(ps) -> str:
    """Export current project state as segments.csv (uses speaker display
    names so the file round-trips human-readably)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["speaker", "line_no", "start", "end", "text"])
    speakers = ps.data["speakers"]
    for l in ps.data["lines"]:
        name = speakers[l["speaker"]]["display_name"] if l["speaker"] else ""
        w.writerow([name, l["line_no"], f"{l['start']:.3f}", f"{l['end']:.3f}",
                    l["text"]])
    return buf.getvalue()
