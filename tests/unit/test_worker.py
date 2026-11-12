"""Worker process shell + heartbeat healthcheck (P2-01)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.jobs import worker


def test_heartbeat_roundtrip(tmp_path: Path) -> None:
    beat = tmp_path / "hb"
    worker.write_heartbeat(beat, now=1000.0)
    assert worker.heartbeat_is_fresh(beat, max_age=60, now=1030.0)
    assert not worker.heartbeat_is_fresh(beat, max_age=60, now=1061.0)
    assert not (tmp_path / "hb.tmp").exists()


@pytest.mark.parametrize("content", [None, "", "not-a-number"])
def test_missing_or_garbage_heartbeat_is_stale(tmp_path: Path, content: str | None) -> None:
    beat = tmp_path / "hb"
    if content is not None:
        beat.write_text(content)
    assert not worker.heartbeat_is_fresh(beat)


def test_future_heartbeat_is_stale(tmp_path: Path) -> None:
    beat = tmp_path / "hb"
    worker.write_heartbeat(beat, now=2000.0)
    assert not worker.heartbeat_is_fresh(beat, now=1000.0)


def test_check_flag_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    beat = tmp_path / "hb"
    monkeypatch.setenv("WORKER_HEARTBEAT_FILE", str(beat))
    assert worker.main(["--check"]) == 1
    worker.write_heartbeat(beat)
    assert worker.main(["--check"]) == 0


def test_run_beats_until_stopped(tmp_path: Path) -> None:
    beat = tmp_path / "hb"
    stop = threading.Event()
    url = f"sqlite:///{tmp_path / 'w.db'}"
    thread = threading.Thread(target=worker.run, args=(stop, beat, 0.01, url))
    thread.start()
    try:
        for _ in range(500):
            if beat.exists():
                break
            stop.wait(0.01)
        assert worker.heartbeat_is_fresh(beat)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not beat.exists()  # removed on clean shutdown


def test_run_fails_fast_without_database(tmp_path: Path) -> None:
    beat = tmp_path / "hb"
    with pytest.raises(Exception):  # noqa: B017, PT011 - driver-specific error type
        worker.run(threading.Event(), beat, 0.01, "sqlite:////nonexistent-dir/x/y.db")
    assert not beat.exists()
