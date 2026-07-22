"""Single-slot job execution: exactly one background job (Demucs separation,
clip extraction, later TTS runs) at a time — the process-level guarantee that
only one GPU operation is ever in flight (single user, no concurrency needed).

A submit while a job is queued/running raises Busy (routers surface it as
409). The finished job stays visible as `current()` until the next submit, so
the UI can show post-run stats after completion.

Jobs report through the Job object: set_message() for the status line and
add_deltas() for per-line badge flips — exactly what GET /jobs/current relays
to the polling frontend.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable


class Busy(RuntimeError):
    """A job is already queued or running."""


class Job:
    def __init__(self, kind: str, project_key: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.project_key = project_key
        self.status = "queued"          # queued | running | done | failed
        self.message = ""
        self.error: str | None = None
        self.result: dict[str, Any] = {}
        self.deltas: dict[int, str] = {}  # line_no -> badge (accumulated)
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._lock = threading.Lock()
        # Append buffer: line numbers submitted (for the SAME project + kind)
        # while this job is already running, to be drained into the current
        # run instead of rejected as Busy. Only meaningful for "dub" jobs;
        # see JobQueue.append_targets and routers/dub.py's drain loop.
        self._appended: list[int] = []
        self.total_targets = 0   # for "(i/N)" — grows as appends land

    # ---- append-queue (called from submit path + the running job) ----
    def append_targets(self, line_nos: list[int]) -> None:
        with self._lock:
            self._appended.extend(line_nos)

    def drain_appended(self) -> list[int]:
        """Return and clear any line numbers appended since the last drain."""
        with self._lock:
            out = self._appended
            self._appended = []
            return out

    # ---- called from inside the running job ----
    def set_message(self, msg: str) -> None:
        with self._lock:
            self.message = msg

    def add_deltas(self, deltas: dict[int, str]) -> None:
        with self._lock:
            self.deltas.update(deltas)

    # ---- called from the polling endpoint ----
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "project_key": self.project_key,
                "status": self.status,
                "message": self.message,
                "error": self.error,
                "result": dict(self.result),
                "deltas": dict(self.deltas),
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")


class JobQueue:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._current: Job | None = None

    def submit(self, kind: str, project_key: str,
               fn: Callable[[Job], dict[str, Any] | None]) -> Job:
        """Run fn(job) on a worker thread. fn's return dict lands in
        job.result; an exception marks the job failed (never kills the
        service — the CLI's die() behavior does not carry over)."""
        with self._guard:
            if self._current is not None and self._current.active:
                raise Busy(
                    f"a job is already {self._current.status}: "
                    f"{self._current.kind} on {self._current.project_key}")
            job = Job(kind, project_key)
            self._current = job

        def run() -> None:
            job.status = "running"
            job.started_at = time.time()
            try:
                result = fn(job)
                if result:
                    job.result.update(result)
                job.status = "done"
            except Exception as e:  # noqa: BLE001 — jobs must never crash the app
                job.error = f"{type(e).__name__}: {e}"
                job.status = "failed"
                traceback.print_exc()
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, name=f"job-{kind}-{job.id}",
                         daemon=True).start()
        return job

    def try_append(self, kind: str, project_key: str,
                   line_nos: list[int]) -> Job | None:
        """If a job of the same kind for the same project is currently
        active, append these line numbers to it (to be drained into the
        running loop) and return that job. Otherwise return None — the
        caller should submit() a fresh job as usual. Thread-safe against a
        job finishing concurrently: the append happens under the same guard
        that submit() uses to swap _current, so we can't append to a job
        that's already being replaced."""
        with self._guard:
            c = self._current
            if c is not None and c.active and c.kind == kind \
                    and c.project_key == project_key:
                c.append_targets(line_nos)
                return c
            return None

    def current(self) -> Job | None:
        """Running/queued job, or the last finished one (for post-run stats)."""
        return self._current

    def busy(self) -> bool:
        c = self._current
        return c is not None and c.active

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Test helper: block until no job is active."""
        deadline = time.time() + timeout
        while self.busy():
            if time.time() > deadline:
                return False
            time.sleep(0.02)
        return True

    def reset_for_tests(self) -> None:
        if self.busy():
            raise Busy("cannot reset while a job is active")
        self._current = None


# process-wide singleton (single user, single app instance)
jobs = JobQueue()
