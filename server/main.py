"""FastAPI 앱 진입점. 정적 화면 서빙 + /ws 하나로 화면과 서버를 잇는다."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.contracts import (
    MESSAGE_DATA_MODELS,
    AckData,
    AgentAdoptData,
    AgentButtonData,
    AgentCorrectData,
    AgentEndcallData,
    AuthHelloData,
    AuthOkData,
    envelope,
)
from server.hub import hub

load_dotenv()

logger = logging.getLogger("deundeun.main")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
UI_DIR = BASE_DIR / "ui"
INDEX_HTML = UI_DIR / "index.html"
ADMIN_HTML = UI_DIR / "admin.html"

WS_TOKEN = os.getenv("WS_TOKEN", "")
AUTH_TIMEOUT_S = 5.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    hub.set_loop(asyncio.get_running_loop())
    # db.py가 아직 없을 수 있으니(오늘 만들어짐) 감싸서 로그만 남기고 넘어간다
    try:
        from server.db import init_db, seed_demo

        init_db()
        seed_demo()
    except Exception:
        logger.exception("init_db/seed_demo 건너뜀 — server/db.py 준비 전이거나 실패")
    yield


app = FastAPI(lifespan=lifespan)


# ── demo.py / api.py 라우터 등록 — 아직 없으면 자리만 ─────────────────────
try:
    from server.demo import router as demo_router

    app.include_router(demo_router)
except Exception:
    logger.info("server/demo.py 미존재 — 라우터 등록 건너뜀")

try:
    from server.api import router as api_router

    app.include_router(api_router)
except Exception:
    logger.info("server/api.py 미존재 — 라우터 등록 건너뜀")


@app.get("/", response_model=None)
async def serve_index() -> FileResponse | JSONResponse:
    # 상담사 화면 진입점
    if not INDEX_HTML.exists():
        return JSONResponse({"error": "ui/index.html 없음"}, status_code=404)
    return FileResponse(INDEX_HTML)


@app.get("/admin", response_model=None)
async def serve_admin() -> FileResponse | JSONResponse:
    # 관리자 화면 진입점
    if not ADMIN_HTML.exists():
        return JSONResponse({"error": "ui/admin.html 없음"}, status_code=404)
    return FileResponse(ADMIN_HTML)


@app.get("/api/token")
async def get_ws_token() -> dict:
    # 데모 전용 · 로컬 접속만 하므로 토큰을 그대로 내어주는 우회를 허용한다
    return {"token": WS_TOKEN}


async def _handle_client_message(agent_id: str, msg_type: str, data: dict) -> None:
    # 화면에서 오는 메시지 처리는 이 함수 하나에서만 정의한다 (중복 구현 금지)
    call_id = hub.current_call_id
    if call_id is None:
        logger.info("활성 통화 없이 %s 수신 — 무시", msg_type)
        return

    if msg_type == "agent.button":
        payload = AgentButtonData(**data)
        from server.db import upsert_button_event

        upsert_button_event(event_id=payload.event_id, agent_id=agent_id, call_id=call_id)
        hub.mark_risk_seen(call_id, payload.event_id)
        hub.broadcast("ack", call_id, AckData(event_id=payload.event_id, saved=True).model_dump())

    elif msg_type == "agent.adopt":
        payload = AgentAdoptData(**data)
        from server.db import RagLog, SessionLocal

        with SessionLocal() as session:
            log = session.get(RagLog, payload.rec_id)
            if log is not None:
                log.adopted = True
                session.commit()
        hub.broadcast("ack", call_id, AckData(event_id=payload.event_id, saved=True).model_dump())

    elif msg_type == "agent.endcall":
        payload = AgentEndcallData(**data)
        # protection.py는 D-4에 생긴다 — 그때까지 자리만 남긴다
        # from server.protection import on_agent_event
        # on_agent_event(call_id, "agent.endcall", payload.model_dump())
        hub.broadcast("ack", call_id, AckData(event_id=payload.event_id, saved=True).model_dump())

    elif msg_type == "agent.correct":
        payload = AgentCorrectData(**data)
        # protection.py는 D-4에 생긴다 — 그때까지 자리만 남긴다
        # from server.protection import on_agent_event
        # on_agent_event(call_id, "agent.correct", payload.model_dump())
        hub.mark_risk_seen(call_id, payload.event_id)
        hub.broadcast("ack", call_id, AckData(event_id=payload.event_id, saved=True).model_dump())

    else:
        logger.info("알 수 없는 화면 메시지 타입: %s", msg_type)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    # 접속 → auth.hello 대조 → 등록 → state.snapshot 1회 → 메시지 루프
    await websocket.accept()

    try:
        raw = await asyncio.wait_for(websocket.receive_json(), timeout=AUTH_TIMEOUT_S)
    except (asyncio.TimeoutError, WebSocketDisconnect, Exception):
        await websocket.close()
        return

    if raw.get("type") != "auth.hello":
        await websocket.close()
        return

    try:
        hello = AuthHelloData(**raw.get("data", {}))
    except Exception:
        await websocket.close()
        return

    if not WS_TOKEN or hello.token != WS_TOKEN:
        await websocket.close()
        return

    agent_id = hello.agent_id
    name = agent_id
    try:
        from server.db import Agent, SessionLocal

        with SessionLocal() as session:
            agent_row = session.get(Agent, agent_id)
            if agent_row is not None:
                name = agent_row.name
    except Exception:
        logger.exception("agent 조회 실패 — agent_id를 이름 대신 사용")

    hub.register(agent_id, websocket)
    call_id = hub.current_call_id or "no-call"

    await websocket.send_json(
        envelope("auth.ok", call_id, hub.next_seq(call_id), AuthOkData(agent_id=agent_id, name=name).model_dump())
    )
    await websocket.send_json(
        envelope(
            "state.snapshot",
            call_id,
            hub.next_seq(call_id),
            hub.snapshot(hub.current_call_id).model_dump(),
        )
    )

    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type")
            data = raw.get("data", {})
            if msg_type not in MESSAGE_DATA_MODELS:
                logger.info("알 수 없는 메시지 타입 수신: %s", msg_type)
                continue
            await _handle_client_message(agent_id, msg_type, data)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(agent_id)


# ui/ 폴더가 아직 없을 수 있으니 있을 때만 정적 파일로 마운트한다 (경로 자체는 "/")
# /ws · /admin · /api/token 등 명시적 라우트보다 반드시 뒤에 마운트해야
# 웹소켓 연결이 정적 파일 서브앱으로 잘못 들어가지 않는다.
if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(UI_DIR)), name="ui")
else:
    logger.info("ui/ 디렉터리 없음 — 정적 파일 마운트 건너뜀")


# uvicorn server.main:app --reload --port 8000
