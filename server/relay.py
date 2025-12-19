"""FastAPI relay for CapsuleBench live events."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .event_store import EventStore

app = FastAPI(title="CapsuleBench Relay")
store = EventStore(Path("server_data/events"))
store.load_existing()

_subscriptions: Dict[str, Set[WebSocket]] = {}
_sub_lock = asyncio.Lock()


async def _broadcast(run_id: str, message: str) -> None:
    async with _sub_lock:
        clients = list(_subscriptions.get(run_id, set()))
    for ws in clients:
        try:
            await ws.send_text(message)
        except RuntimeError:
            pass


@app.websocket("/ws/ingest/{run_id}")
async def ingest_socket(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    try:
        while True:
            message = await ws.receive_text()
            store.append(run_id, message)
            await _broadcast(run_id, message)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/subscribe/{run_id}")
async def subscribe_socket(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    async with _sub_lock:
        _subscriptions.setdefault(run_id, set()).add(ws)
    try:
        history = store.history(run_id)
        for entry in history:
            await ws.send_text(entry)
        while True:
            await asyncio.sleep(3600)
    except WebSocketDisconnect:
        pass
    finally:
        async with _sub_lock:
            _subscriptions.get(run_id, set()).discard(ws)


@app.get("/runs/{run_id}/events")
async def get_events(run_id: str, since_seq: int | None = None) -> JSONResponse:
    history = store.history(run_id, since_seq)
    payload = [json.loads(line) for line in history]
    return JSONResponse({"run_id": run_id, "events": payload})


@app.get("/runs/{run_id}/snapshot")
async def get_snapshot(run_id: str) -> JSONResponse:
    history = store.history(run_id)
    latest = json.loads(history[-1]) if history else None
    response = {
        "run_id": run_id,
        "latest_seq": latest.get("seq") if latest else 0,
        "latest_event": latest,
    }
    status = 200 if latest else 404
    return JSONResponse(response, status_code=status)


def _format_ts(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def _run_metadata(run_id: str) -> dict | None:
    history = store.history(run_id)
    if not history:
        return None
    created_at_ms: int | None = None
    backend = policy_id = track_id = trace_id = None
    for line in history:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if created_at_ms is None:
            created_at_ms = event.get("ts_ms")
        if event.get("type") == "run_started":
            data = event.get("data") or {}
            backend = data.get("backend") or backend
            policy_id = data.get("policy_id") or policy_id
            track_id = data.get("track_id") or track_id
            trace_id = data.get("trace_id") or trace_id
    return {
        "run_id": run_id,
        "trace_id": trace_id or run_id,
        "backend": backend or "unknown",
        "policy_id": policy_id or "unknown",
        "track_id": track_id or "unknown",
        "created_at": _format_ts(created_at_ms),
    }


@app.get("/runs")
async def list_runs() -> JSONResponse:
    runs = []
    for run_id in store.run_ids():
        meta = _run_metadata(run_id)
        if meta:
            runs.append(meta)
    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return JSONResponse({"runs": runs})
@app.get("/")
async def root() -> dict:
    return {"service": "capsule relay", "ok": True}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
