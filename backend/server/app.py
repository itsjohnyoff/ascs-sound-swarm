"""FastAPI entry point for the phones, dashboard, and judge console."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Literal

import anyio
import segno
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat, ValidationError

from backend.server.audio_buffer import BUFFER_DURATION_S, decode_audio_frame
from backend.server.events import EventLoop
from backend.server.recording import Recording
from backend.server.session import CalibrationConflict, Session
from shared.protocol import CalibrationMode, RoomConfig, SoundClass

logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path("data/recordings")
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
SocketQueue = asyncio.Queue[dict]


class Position(BaseModel):
    label: str | None = None
    x_m: FiniteFloat
    y_m: FiniteFloat


class PhoneHello(Position):
    type: Literal["phone_hello"]
    phone_id: str
    sample_rate: int = Field(ge=8000, le=192000)
    device_info: str = ""


class CalibrateArm(BaseModel):
    mode: CalibrationMode


class CalibrateSolve(CalibrateArm):
    calibration_id: str
    window_s: float = Field(default=7.0, gt=0, le=BUFFER_DURATION_S)


RECORDING_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}(?:\.npz)?$"


class RecordStart(BaseModel):
    name: str = Field(default="spike-01", pattern=RECORDING_NAME_PATTERN)


class NewTrial(BaseModel):
    label: str
    gt_x_m: FiniteFloat
    gt_y_m: FiniteFloat
    expected_class: SoundClass
    event_id: str | None = None


def _inside(room: RoomConfig, x_m: float, y_m: float) -> bool:
    return 0 <= x_m <= room.width_m and 0 <= y_m <= room.height_m


def _enqueue(queue: SocketQueue, payload: dict) -> None:
    # shortcut: bound each local client's backlog, add replay if clients need
    # delivery guarantees after lagging or reconnecting.
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(payload)


async def _serve_socket(ws: WebSocket, queue: SocketQueue, read: Callable[[], Awaitable[None]]) -> None:
    async with anyio.create_task_group() as group:
        async def write() -> None:
            try:
                while True:
                    payload = await queue.get()
                    with anyio.fail_after(2.0):
                        await ws.send_json(payload)
            except WebSocketDisconnect:
                pass
            except TimeoutError:
                await ws.close(code=1013)
            finally:
                group.cancel_scope.cancel()

        group.start_soon(write)
        await read()
        group.cancel_scope.cancel()


def create_app(
    session_factory: Callable[[], Session] = Session,
    *,
    frontend_dir: Path = FRONTEND_DIST,
    recordings_dir: Path = RECORDINGS_DIR,
) -> FastAPI:
    dashboards: set[SocketQueue] = set()
    phones: dict[str, SocketQueue] = {}

    def broadcast(payload: dict) -> None:
        for queue in dashboards:
            _enqueue(queue, payload)

    def push_status(session: Session, phone_id: str) -> None:
        if phone_id in phones:
            _enqueue(phones[phone_id], session.phone_status(phone_id))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session = app.state.session = session_factory()
        session.log_backends()
        processor = EventLoop(
            session, broadcast, min_phones=2 if os.getenv("ASCS_DIAGNOSTIC_TWO_PHONE") == "1" else 3,
        )
        async with asyncio.TaskGroup() as group:
            tasks = (group.create_task(processor.run()), group.create_task(processor.run_telemetry()))
            try:
                yield
            finally:
                for task in tasks:
                    task.cancel()
                if session.recording is not None:
                    discard_recording(session)

    app = FastAPI(title="ASCS Sound Swarm", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["http://localhost:5173"],
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health(request: Request):
        session: Session = request.app.state.session
        return {
            "status": "ok",
            "phones_connected": len(session.active_phones()),
            "synced_phones": len(session.synced_phones()),
            "classifier_backend": session.classifier.backend_name,
        }

    @app.get("/api/session")
    async def get_session(request: Request):
        return request.app.state.session.snapshot()

    @app.put("/api/room")
    async def put_room(room: RoomConfig, request: Request):
        session: Session = request.app.state.session
        if not _inside(room, room.speaker_x_m, room.speaker_y_m):
            raise HTTPException(422, "Speaker coordinates must be inside the room")
        if any(not _inside(room, p.x_m, p.y_m) for p in session.phones.values()):
            raise HTTPException(422, "Phone coordinates must be inside the room")
        session.update_room(room)
        broadcast(session.snapshot())
        for phone_id in phones:
            push_status(session, phone_id)
        return session.room

    @app.put("/api/phones/{phone_id}")
    async def put_phone(phone_id: str, position: Position, request: Request):
        session: Session = request.app.state.session
        if not _inside(session.room, position.x_m, position.y_m):
            raise HTTPException(422, "Phone coordinates must be inside the room")
        phone = session.set_phone(phone_id, **position.model_dump(exclude_none=True))
        session.positioned.add(phone_id)
        broadcast(session.snapshot())
        push_status(session, phone_id)
        return phone

    @app.delete("/api/phones/{phone_id}", status_code=204)
    async def delete_phone(phone_id: str, request: Request):
        session: Session = request.app.state.session
        phone = session.phones.get(phone_id)
        if phone is None:
            raise HTTPException(404, "Unknown phone")
        if phone.connected:
            raise HTTPException(409, "Disconnect the phone before removing it")
        del session.phones[phone_id]
        broadcast(session.snapshot())

    @app.post("/api/calibrate/arm")
    async def arm_calibration(body: CalibrateArm, request: Request):
        session: Session = request.app.state.session
        try:
            run = session.arm(body.mode)
        except CalibrationConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        broadcast(session.snapshot())
        return run.response()

    @app.post("/api/calibrate/solve")
    async def solve_calibration(body: CalibrateSolve, request: Request):
        session: Session = request.app.state.session
        revision = session.revision
        try:
            result = await session.solve(body.calibration_id, body.mode, body.window_s)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except CalibrationConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        payload = {"type": "calibration_result", **result.model_dump(mode="json")}
        if session.revision != revision:
            broadcast(payload)
            broadcast(session.snapshot())
            for phone_id in phones:
                push_status(session, phone_id)
        return payload

    def discard_recording(session: Session) -> None:
        recording, session.recording = session.recording, None
        if recording is not None:
            try:
                recording.close()
            except OSError:
                logger.exception("Could not clean up recording %s", recording.path)

    @app.get("/api/record/status")
    async def record_status(request: Request):
        recording = request.app.state.session.recording
        if recording is None:
            return {"status": "idle", "path": None}
        if recording.saving:
            status = "saving"
        elif recording.stopped_at_s is not None:
            status = "awaiting_save"
        else:
            status = "recording"
        return {"status": status, "path": str(recording.path)}

    @app.post("/api/record/start")
    async def start_recording(request: Request, body: RecordStart = RecordStart()):
        session: Session = request.app.state.session
        if session.recording is not None:
            raise HTTPException(409, "A recording is already active or awaiting save")
        path = recordings_dir / f"{body.name.removesuffix('.npz')}.npz"
        try:
            if path.exists():
                raise HTTPException(409, "A recording with this name already exists")
            session.recording = Recording(path, session.room)
        except OSError as exc:
            logger.exception("Could not start recording %s", path)
            raise HTTPException(507, "Could not create recording storage") from exc
        return {"status": "recording", "path": str(path)}

    @app.post("/api/record/stop")
    async def stop_recording(
        request: Request,
        name: str | None = Query(default=None, pattern=RECORDING_NAME_PATTERN),
    ):
        session: Session = request.app.state.session
        recording = session.recording
        if recording is None or recording.saving:
            raise HTTPException(409, "No recording is ready to stop")
        if recording.failed:
            discard_recording(session)
            raise HTTPException(507, "Capture storage failed")
        if not recording.has_audio:
            discard_recording(session)
            raise HTTPException(409, "Recording received no phone audio")
        if name is not None:
            target = recordings_dir / f"{name.removesuffix('.npz')}.npz"
            if target.exists():
                raise HTTPException(409, "A recording with this name already exists")
            recording.path = target

        recording.stopped_at_s = recording.stopped_at_s or time.time()
        recording.saving = True
        try:
            # Keep ownership of the spool until the writer returns, including if
            # the caller disconnects during the save.
            await anyio.to_thread.run_sync(recording.save)
        except FileExistsError as exc:
            raise HTTPException(409, "A recording with this name already exists") from exc
        except OSError as exc:
            logger.exception("Could not save recording %s", recording.path)
            raise HTTPException(507, "Could not save the recording. Retry stop to save captured audio.") from exc
        else:
            discard_recording(session)
            return {"status": "saved", "path": str(recording.path)}
        finally:
            recording.saving = False

    @app.get("/api/eval/trials")
    async def get_trials(request: Request):
        judge = request.app.state.session.judge
        return {"trials": judge.trials, "summary": judge.summary()}

    @app.post("/api/eval/trials")
    async def post_trial(body: NewTrial, request: Request):
        session: Session = request.app.state.session
        event = None
        if body.event_id is not None:
            event = session.detections.get(body.event_id)
            if event is None:
                raise HTTPException(404, "Detection is no longer available")
        return session.judge.add_trial(
            body.label, body.gt_x_m, body.gt_y_m, body.expected_class, event, hull=session.phone_hull(),
        )

    @app.post("/api/eval/known-point")
    async def post_known_point(body: NewTrial, request: Request):
        session: Session = request.app.state.session
        event = session.detections.get(body.event_id) if body.event_id is not None else None
        if body.event_id is not None and event is None:
            raise HTTPException(404, "Detection is no longer available")
        return session.judge.add_trial(
            body.label, body.gt_x_m, body.gt_y_m, body.expected_class, event,
            hull=session.phone_hull(), known_point=True,
        )

    @app.get("/api/eval/known-point")
    async def get_known_points(request: Request):
        return {"checks": request.app.state.session.judge.known_points}

    @app.get("/api/detections/recent")
    async def recent_detections(request: Request, limit: int = 5):
        detections = list(request.app.state.session.detections.values())[-limit:]
        return {"detections": [d.model_dump(mode="json") for d in reversed(detections)]}

    @app.delete("/api/eval/trials", status_code=204)
    async def delete_trials(request: Request):
        request.app.state.session.judge.clear()

    @app.get("/api/eval/export.csv")
    async def export_csv(request: Request):
        return Response(
            request.app.state.session.judge.export_csv(), media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="judge-eval.csv"'},
        )

    @app.get("/api/join-info")
    async def join_info(request: Request):
        base = os.getenv("PUBLIC_BASE_URL") or str(request.base_url)
        join_url = f"{base.rstrip('/')}/phone.html"
        svg = BytesIO()
        segno.make_qr(join_url).save(svg, kind="svg", scale=4, xmldecl=False, light="white")
        return {"join_url": join_url, "qr_svg": svg.getvalue().decode("utf-8")}

    @app.websocket("/ws/dashboard")
    async def dashboard_socket(ws: WebSocket):
        await ws.accept()
        session: Session = ws.app.state.session
        queue: SocketQueue = asyncio.Queue(maxsize=32)
        queue.put_nowait(session.snapshot())
        dashboards.add(queue)

        async def read() -> None:
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass

        try:
            await _serve_socket(ws, queue, read)
        finally:
            dashboards.discard(queue)

    @app.websocket("/ws/phone/{phone_id}")
    async def phone_socket(ws: WebSocket, phone_id: str):
        await ws.accept()
        if phone_id in phones:
            await ws.close(code=1008, reason="Phone ID is already connected")
            return
        session: Session = ws.app.state.session
        queue: SocketQueue = asyncio.Queue(maxsize=32)
        session.connect(phone_id)
        phones[phone_id] = queue
        broadcast(session.snapshot())

        async def read() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                binary, raw_text = message.get("bytes"), message.get("text")
                if binary is not None:
                    try:
                        frame = decode_audio_frame(binary)
                    except ValueError:
                        continue
                    if session.ingest(phone_id, frame):
                        for pid in phones:
                            push_status(session, pid)
                        broadcast(session.snapshot())
                elif raw_text is not None:
                    try:
                        hello = PhoneHello.model_validate_json(raw_text)
                    except ValidationError:
                        continue
                    if hello.phone_id != phone_id or not _inside(session.room, hello.x_m, hello.y_m):
                        continue
                    fields = session.hello_position(phone_id, hello.model_dump(
                        exclude={"type", "phone_id", "sample_rate"}, exclude_none=True))
                    if session.buffers[phone_id].latest_sample_index == 0:
                        fields["sample_rate"] = hello.sample_rate
                    session.set_phone(phone_id, **fields)
                    push_status(session, phone_id)
                    broadcast(session.snapshot())

        try:
            await _serve_socket(ws, queue, read)
        finally:
            del phones[phone_id]
            session.disconnect(phone_id)
            for pid in phones:
                push_status(session, pid)
            broadcast(session.snapshot())

    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


app = create_app()
