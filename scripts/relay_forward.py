#!/usr/bin/env python3
"""Forward local capsule-bench event logs to a relay ingest websocket."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from contextlib import closing

from websocket import WebSocketConnectionClosedException, create_connection


def _forward_events(events_path: Path, ingest_url: str, *, skip_history: bool, poll_interval: float) -> None:
    events_path = events_path.expanduser().resolve()
    if not events_path.exists():
        raise FileNotFoundError(f"events log missing: {events_path}")

    with events_path.open("r", encoding="utf-8") as fh, closing(create_connection(ingest_url)) as ws:
        if not skip_history:
            for line in fh:
                data = line.strip()
                if data:
                    ws.send(data)
        else:
            fh.seek(0, 2)

        while True:
            line = fh.readline()
            if line:
                data = line.strip()
                if not data:
                    continue
                try:
                    ws.send(data)
                except WebSocketConnectionClosedException as exc:  # pragma: no cover - network dependent
                    raise RuntimeError("relay websocket closed") from exc
            else:
                time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward events.jsonl to relay ingest websocket")
    parser.add_argument("events_path", type=Path, help="Path to capsule-bench events.jsonl")
    parser.add_argument(
        "ingest_url",
        type=str,
        help="WebSocket ingest URL (e.g., ws://localhost:8000/ws/ingest/<run_id>)",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Only stream new lines (tail mode) instead of replaying the file",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Sleep interval (seconds) when waiting for new lines",
    )
    args = parser.parse_args()
    _forward_events(args.events_path, args.ingest_url, skip_history=args.skip_history, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
