"""FastAPI relay for CapsuleBench live events."""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse

from .event_store import EventStore
try:
    from .event_store_pg import PostgresEventStore
except ImportError:  # pragma: no cover - psycopg optional locally
    PostgresEventStore = None


def _build_store() -> EventStore:
    dsn = os.environ.get("DATABASE_URL")
    if dsn and PostgresEventStore is not None:
        try:
            print("[relay] using PostgresEventStore", flush=True)
            return PostgresEventStore(dsn)
        except Exception as exc:  # pragma: no cover - log fallback
            print(f"[relay] PostgresEventStore init failed: {exc}; falling back", flush=True)
    path_store = EventStore(Path("server_data/events"))
    path_store.load_existing()
    return path_store


app = FastAPI(title="CapsuleBench Relay")
store = _build_store()
ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_ROOT", "server_data/artifacts")).resolve()
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

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
            await asyncio.to_thread(store.append, run_id, message)
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
async def get_events(run_id: str, after_seq: int | None = Query(default=None)) -> JSONResponse:
    history = store.history(run_id, after_seq)
    payload = []
    for line in history:
        try:
            payload.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return JSONResponse(payload)


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


def _artifact_dir(run_id: str) -> Path:
    return (ARTIFACTS_ROOT / run_id).resolve()


def _list_artifacts(run_id: str) -> List[dict]:
    root = _artifact_dir(run_id)
    if not root.exists() or not root.is_dir():
        return []
    result: List[dict] = []
    for child in sorted(root.iterdir()):
        if not child.is_file():
            continue
        stat = child.stat()
        content_type = mimetypes.guess_type(child.name)[0] or "application/octet-stream"
        result.append(
            {
                "name": child.name,
                "size_bytes": stat.st_size,
                "content_type": content_type,
            }
        )
    return result


def _artifact_path(run_id: str, artifact_name: str) -> Path:
    root = _artifact_dir(run_id)
    safe_name = Path(artifact_name).name
    path = (root / safe_name).resolve()
    if not str(path).startswith(str(root)):
        raise HTTPException(status_code=403, detail="forbidden")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return path


@app.get("/runs")
async def list_runs() -> JSONResponse:
    runs = []
    for run_id in store.run_ids():
        meta = _run_metadata(run_id)
        if meta:
            runs.append(meta)
    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return JSONResponse(runs)


@app.get("/runs/{run_id}")
async def run_detail(run_id: str) -> dict:
    history = store.history(run_id)
    if not history:
        raise HTTPException(status_code=404, detail="run not found")
    metadata = _run_metadata(run_id) or {"run_id": run_id}
    # include latest events (e.g., last 100)
    latest_events = []
    for line in history[-100:]:
        try:
            latest_events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    last_seq = 0
    if latest_events:
        last_seq = max(int(evt.get("seq", 0)) for evt in latest_events)
    return {
        "run": metadata,
        "events": latest_events,
        "last_seq": last_seq,
        "artifacts": _list_artifacts(run_id),
    }


@app.get("/runs/{run_id}/artifacts")
async def list_run_artifacts(run_id: str) -> JSONResponse:
    if not _run_metadata(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse(_list_artifacts(run_id))


@app.get("/runs/{run_id}/artifacts/{artifact_name}")
async def download_artifact(run_id: str, artifact_name: str):
    path = _artifact_path(run_id, artifact_name)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)
@app.get("/")
async def root() -> dict:
    return {"service": "capsule relay", "ok": True}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
