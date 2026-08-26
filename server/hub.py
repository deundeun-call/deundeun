"""화면에 신호를 보내는 Hub. server/contracts.py의 envelope으로 형식을 맞춘다."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import WebSocket

from server.contracts import (
    GaugeUpdateData,
    RiskEventData,
    ScriptSuggestData,
    StateSnapshotData,
    envelope,
)

logger = logging.getLogger("deundeun.hub")


class Hub:
    """연결된 클라이언트 집합을 들고, call_id별 최신 상태를 갱신·보관한다 — 데모 범위: 동시 통화 1건 가정."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}  # agent_id -> websocket
        self._seq: dict[str, int] = {}  # call_id -> 다음 seq
        self.current_call_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # broadcast는 다른 스레드에서도 불릴 수 있다
        # call_id별 「현재 상태」 — state.snapshot이 여기서 나온다
        self._gauge: dict[str, GaugeUpdateData] = {}
        self._active_script: dict[str, ScriptSuggestData] = {}
        self._unseen_risks: dict[str, dict[str, RiskEventData]] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        # FastAPI startup에서 잡은 이벤트 루프를 등록한다 — broadcast가 이 루프로 스레드-세이프하게 보낸다
        self._loop = loop

    def register(self, agent_id: str, websocket: WebSocket) -> None:
        # 인증 통과한 연결을 등록한다
        self._connections[agent_id] = websocket

    def unregister(self, agent_id: str) -> None:
        # 연결이 끊기면 등록을 지운다
        self._connections.pop(agent_id, None)

    def next_seq(self, call_id: str) -> int:
        # call_id별 증가하는 시퀀스 번호를 내어준다 (auth.ok 등 개별 전송에도 씀)
        self._seq[call_id] = self._seq.get(call_id, 0) + 1
        return self._seq[call_id]

    def broadcast(self, type: str, call_id: str, data: dict) -> None:
        # 화면에 신호 보내기 — 인자는 언제나 3개다 (계약 그대로). seq를 1씩 올려 전원에게 보낸다.
        # 동기 메서드라 exposure/protection 등 다른 스레드에서도 호출될 수 있으므로
        # 항상 서버의 이벤트 루프에 스레드-세이프하게 스케줄한다.
        env = envelope(type, call_id, self.next_seq(call_id), data)
        self._update_snapshot(type, call_id, env["data"])
        if self._loop is None:
            logger.warning("이벤트 루프 준비 전 — broadcast(%s) 건너뜀", type)
            return
        for websocket in list(self._connections.values()):
            asyncio.run_coroutine_threadsafe(self._safe_send(websocket, env), self._loop)

    async def _safe_send(self, websocket: WebSocket, env: dict) -> None:
        # 연결이 이미 끊긴 소켓으로 보내다 나는 에러는 무시한다
        try:
            await websocket.send_json(env)
        except Exception:
            logger.info("broadcast 실패 — 연결이 이미 닫힌 것으로 보임")

    def _update_snapshot(self, type: str, call_id: str, data: dict) -> None:
        # broadcast할 때마다 call_id별 최신 gauge·active_script·미확인 risk 목록을 갱신한다
        if type == "call.start":
            self.current_call_id = call_id
            self._gauge.pop(call_id, None)
            self._active_script.pop(call_id, None)
            self._unseen_risks.pop(call_id, None)
        elif type == "gauge.update":
            self._gauge[call_id] = GaugeUpdateData(**data)
        elif type == "script.suggest":
            self._active_script[call_id] = ScriptSuggestData(**data)
        elif type == "script.none":
            self._active_script.pop(call_id, None)
        elif type == "risk.event":
            risk = RiskEventData(**data)
            self._unseen_risks.setdefault(call_id, {})[risk.event_id] = risk
        elif type == "call.end":
            self.current_call_id = None

    def mark_risk_seen(self, call_id: str, event_id: str) -> None:
        # 상담사가 반응한(button/correct) risk는 미확인 목록에서 뺀다
        self._unseen_risks.get(call_id, {}).pop(event_id, None)

    def snapshot(self, call_id: Optional[str]) -> StateSnapshotData:
        # 접속 직후 1회 보낼 현재 게이지·활성 스크립트·미확인 risk 목록 (state.snapshot의 데이터)
        if call_id is None:
            return StateSnapshotData()
        return StateSnapshotData(
            gauge=self._gauge.get(call_id),
            active_script=self._active_script.get(call_id),
            unseen_risks=list(self._unseen_risks.get(call_id, {}).values()),
        )


hub = Hub()  # 서버 전체가 공유하는 단일 Hub 인스턴스
