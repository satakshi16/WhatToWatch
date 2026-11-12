"""Worker process (P2-01).

Runs in the same image as the API (`python -m app.jobs.worker`). P2-01 only provides the
process: it checks the database, then writes a heartbeat file until SIGTERM / SIGINT.
P2-04 adds the scheduler, queue and operator jobs (ingest, availability, retention).

`python -m app.jobs.worker --check` is the container healthcheck: exit 0 while the
heartbeat is fresh, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import check_database, create_db_engine

HEARTBEAT_INTERVAL_SECONDS = 15.0
HEARTBEAT_MAX_AGE_SECONDS = 60.0
DEFAULT_HEARTBEAT_FILE = Path(tempfile.gettempdir()) / "whattowatch-worker.heartbeat"

log = get_logger(__name__)


def heartbeat_path() -> Path:
    return Path(os.environ.get("WORKER_HEARTBEAT_FILE") or DEFAULT_HEARTBEAT_FILE)


def write_heartbeat(path: Path, now: float | None = None) -> None:
    """Atomically record `now` (epoch seconds) in `path`."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{time.time() if now is None else now:.3f}", encoding="utf-8")
    tmp.replace(path)


def heartbeat_is_fresh(
    path: Path, max_age: float = HEARTBEAT_MAX_AGE_SECONDS, now: float | None = None
) -> bool:
    try:
        beat = float(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return 0 <= (time.time() if now is None else now) - beat <= max_age


def run(
    stop: threading.Event,
    path: Path,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
    database_url: str | None = None,
) -> None:
    """Check the database, then beat every `interval` seconds until `stop` is set."""
    engine = create_db_engine(database_url or get_settings().database_url)
    try:
        check_database(engine)
    finally:
        engine.dispose()
    log.info("worker_started", jobs=0, heartbeat_file=str(path))
    while not stop.is_set():
        write_heartbeat(path)
        stop.wait(interval)
    path.unlink(missing_ok=True)
    log.info("worker_stopped")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.jobs.worker")
    parser.add_argument("--check", action="store_true", help="healthcheck: is the heartbeat fresh")
    args = parser.parse_args(argv)
    path = heartbeat_path()
    if args.check:
        return 0 if heartbeat_is_fresh(path) else 1

    configure_logging(get_settings().log_level)
    stop = threading.Event()

    def _handle(signum: int, _frame: FrameType | None) -> None:
        log.info("worker_signal", signal=signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    try:
        run(stop, path)
    except Exception as exc:
        log.error("worker_failed", exc_type=type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
