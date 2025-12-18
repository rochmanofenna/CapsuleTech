"""FastAPI relay for CapsuleBench live events."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
