# ─────────────────────────────────────────────────────────────────────────
# WebSocket 메시지 표 (팀 합의본 — 그대로 옮김. 표 정렬만 맞췄고 내용은 안 바꿈)
#
# type                 방향        data 안에 들어가는 것                                        화면에서 하는 일
# auth.hello           화면→서버   {token, agent_id}                                            접속 직후 1회 — 없으면 서버가 끊음
# auth.ok              서버→화면   {agent_id, name}                                             인증 통과
# call.start           서버→화면   {agent_name, customer_masked, prev_summary}                  화면 초기화 + 이전 상담 이력 패널 (없으면 null)
# utterance.detected   서버→화면   {speaker: "customer"} — 값은 항상 customer                   "듣는 중…" 표시
# stt.final            서버→화면   {speaker, text, t0, t1, masked_n} — speaker는 항상 "customer" 자막 한 줄 추가 (masked_n>0이면 "개인정보 n건 가림" 배지)
# gauge.update         서버→화면   {score: 0~100, level: green/yellow/red, reasons:[...]}       게이지 이동 + 사유 배지
# risk.event           서버→화면   {event_id, reasons:[...], duration_s, stage:1~4, text}       보호조치 카드. text = 그 단계의 정부 매뉴얼 문안 원문
# confirm.request      서버→화면   {reasons:[...]}                                              "음향만 급등 — 지금 힘든 통화인가요?" 확인 카드
# script.suggest       서버→화면   {text, sources:[{doc,clause,page}], score}                   스크립트 카드 + 출처
# script.none          서버→화면   {score, reason:"근거 없음"}                                  "규정 근거 없음 · 이관 안내"
# summary.ready        서버→화면   {facts:[], pending:[], caution:[]}                           3층 요약 패널
# call.end             서버→화면   {duration_s, risk_count, exposure_min}                       통화 종료
# agent.button         화면→서버   {event_id}                                                   [지금 힘듦] 버튼
# agent.adopt          화면→서버   {event_id, rec_id}                                           스크립트 채택
# agent.endcall        화면→서버   {event_id, reason}                                           상담사가 직접 누르는 종료
# agent.correct        화면→서버   {event_id}                                                   [오탐 정정] — 이 판정은 오판이었다
# ack                  서버→화면   {event_id, saved:true}                                       기록 저장됨 표시
# state.snapshot       서버→화면   {gauge, active_script, unseen_risks:[]}                      접속·재접속 직후 1회 — 화면 통째 복구
# system.degraded      서버→화면   {module, reason}                                             "분석 일시 중단" 배지
# ─────────────────────────────────────────────────────────────────────────
"""이 이름과 모양을 글자 그대로 지킨다. 바꾸려면 김형중·김예원에게 먼저 말한다 (CLAUDE.md 참고)."""

import time
from typing import Any, Literal, Optional, Protocol, TypedDict, Union

from pydantic import BaseModel, Field

# ── 1) 내부 함수/메서드 계약 — CLAUDE.md "계약" 절의 이름·인자·반환을 그대로 옮김 ──


class WindowFeatures(TypedDict):
    # acoustic 창 하나의 특징값 — 기준선·무효 구간에도 이 모양 그대로 채운다
    f0_med: Optional[float]
    f0_iqr: Optional[float]
    z: float
    window_score: float


class UtteranceResult(TypedDict, total=False):
    # on_utterance의 반환 — 'window' 키가 반드시 있어야 한다
    window: WindowFeatures


class RiskCard(TypedDict):
    # on_risk의 반환 — 정수만 주면 화면 카드가 빈칸이 되므로 세 키를 항상 채운다
    stage: int  # 1~4
    text: str
    rec_id: str


class SuggestResult(TypedDict):
    # suggest의 반환 — 다섯 키를 항상 전부 채운다
    ok: bool
    text: str
    sources: list
    score: float
    reason: str


class SummaryResult(TypedDict):
    # build_summary의 반환 — 3층 요약. ③층은 DB 수치에서 규칙으로만 만든다
    facts: Any
    pending: Any
    caution: Any


class Hub(Protocol):
    def broadcast(self, type: str, call_id: str, data: dict) -> None:
        # 화면에 신호 보내기 — 인자는 언제나 3개다
        ...


class ExposureEngine(Protocol):
    def start_call(self, call_id: str) -> None:
        # 통화 시작 — 이 통화의 노출량 상태를 초기화한다
        ...

    async def on_utterance(
        self,
        call_id: str,
        speaker: str,
        text: str,
        audio: Any,
        sr: int,
        t0: float,
        t1: float,
    ) -> UtteranceResult:
        # 발화 하나를 처리해 창 특징과 노출량을 갱신한다
        ...

    def end_call(self, call_id: str) -> float:
        # 통화 종료 — 최종 노출량을 반환한다
        ...


class Protection(Protocol):
    def on_risk(self, call_id: str, risk: Any) -> RiskCard:
        # 보호조치 카드를 만든다 — 정수만 반환하면 화면 카드가 빈칸이 된다
        ...


class Transcriber(Protocol):
    async def transcribe(self, audio_np: Any, sr: int) -> str:
        # 음성 인식 — 고객 채널만. speaker 인자 없음
        ...


class VAD(Protocol):
    def flush(self, t_end: float) -> None:
        # 말한 구간 감지 — 파일 끝의 마지막 발화를 강제로 내보낸다
        ...


class Rag(Protocol):
    async def suggest(self, call_id: str, recent_texts: list) -> SuggestResult:
        # 규정집 검색 — 적합도가 문턱(rag.tau) 미달이면 LLM을 호출조차 하지 않는다
        ...


class Summarizer(Protocol):
    async def build_summary(self, call_id: str) -> SummaryResult:
        # 통화 종료 뒤 3층 요약을 만든다
        ...


# ── 2) WebSocket 메시지 계약 — 위 표를 pydantic 모델로 그대로 옮김 ──

PROTOCOL_VERSION = 1

GaugeLevel = Literal["green", "yellow", "red"]


class Envelope(BaseModel):
    # 모든 WS 메시지의 겉면 — type/v/ts/call_id/seq는 메시지 종류와 무관하게 항상 있다
    type: str
    v: int = PROTOCOL_VERSION
    ts: float = Field(default_factory=time.time)
    call_id: str
    seq: int
    data: dict


def envelope(type: str, call_id: str, seq: int, data: Union[BaseModel, dict]) -> dict:
    # data 모델(혹은 dict)을 받아 Envelope 모양의 dict로 감싸서 돌려준다 — hub.broadcast에 바로 넘기면 됨
    payload = data.model_dump() if isinstance(data, BaseModel) else data
    return Envelope(type=type, call_id=call_id, seq=seq, data=payload).model_dump()


# -- 화면→서버 --


class AuthHelloData(BaseModel):
    # 접속 직후 1회 — 없으면 서버가 끊음
    token: str
    agent_id: str


class AgentButtonData(BaseModel):
    # [지금 힘듦] 버튼
    event_id: str


class AgentAdoptData(BaseModel):
    # 스크립트 채택
    event_id: str
    rec_id: str


class AgentEndcallData(BaseModel):
    # 상담사가 직접 누르는 종료
    event_id: str
    reason: str


class AgentCorrectData(BaseModel):
    # [오탐 정정] — 이 판정은 오판이었다
    event_id: str


# -- 서버→화면 --


class AuthOkData(BaseModel):
    # 인증 통과
    agent_id: str
    name: str


class SummaryPanelData(BaseModel):
    # 3층 요약 패널 모양 — summary.ready와 call.start.prev_summary가 함께 쓴다
    facts: list = Field(default_factory=list)
    pending: list = Field(default_factory=list)
    caution: list = Field(default_factory=list)


class CallStartData(BaseModel):
    # 화면 초기화 + 이전 상담 이력 패널 (없으면 null)
    agent_name: str
    customer_masked: str
    prev_summary: Optional[SummaryPanelData] = None


class UtteranceDetectedData(BaseModel):
    # "듣는 중…" 표시 — 값은 항상 customer
    speaker: Literal["customer"] = "customer"


class SttFinalData(BaseModel):
    # 자막 한 줄 추가 (masked_n>0이면 "개인정보 n건 가림" 배지) — speaker는 항상 customer
    speaker: Literal["customer"] = "customer"
    text: str
    t0: float
    t1: float
    masked_n: int = 0


class GaugeUpdateData(BaseModel):
    # 게이지 이동 + 사유 배지
    score: float = Field(ge=0, le=100)
    level: GaugeLevel
    reasons: list[str] = Field(default_factory=list)


class RiskEventData(BaseModel):
    # 보호조치 카드. text = 그 단계의 정부 매뉴얼 문안 원문
    event_id: str
    reasons: list[str] = Field(default_factory=list)
    duration_s: float
    stage: int = Field(ge=1, le=4)
    text: str


class ConfirmRequestData(BaseModel):
    # "음향만 급등 — 지금 힘든 통화인가요?" 확인 카드
    reasons: list[str] = Field(default_factory=list)


class ScriptSource(BaseModel):
    # script.suggest 출처 한 건
    doc: str
    clause: str
    page: int


class ScriptSuggestData(BaseModel):
    # 스크립트 카드 + 출처
    text: str
    sources: list[ScriptSource] = Field(default_factory=list)
    score: float


class ScriptNoneData(BaseModel):
    # "규정 근거 없음 · 이관 안내"
    score: float
    reason: str


class SummaryReadyData(SummaryPanelData):
    # 3층 요약 패널 — 모양은 SummaryPanelData와 같다
    pass


class CallEndData(BaseModel):
    # 통화 종료
    duration_s: float
    risk_count: int
    exposure_min: float


class AckData(BaseModel):
    # 기록 저장됨 표시
    event_id: str
    saved: bool = True


class StateSnapshotData(BaseModel):
    # 접속·재접속 직후 1회 — 화면 통째 복구 (하위 모양은 gauge.update/script.suggest/risk.event와 같다)
    gauge: Optional[GaugeUpdateData] = None
    active_script: Optional[ScriptSuggestData] = None
    unseen_risks: list[RiskEventData] = Field(default_factory=list)


class SystemDegradedData(BaseModel):
    # "분석 일시 중단" 배지
    module: str
    reason: str


# type 문자열 -> data 모델 매핑. 표와 코드가 어긋나지 않았는지 확인할 때 이거 하나만 보면 된다.
MESSAGE_DATA_MODELS: dict[str, type[BaseModel]] = {
    "auth.hello": AuthHelloData,
    "auth.ok": AuthOkData,
    "call.start": CallStartData,
    "utterance.detected": UtteranceDetectedData,
    "stt.final": SttFinalData,
    "gauge.update": GaugeUpdateData,
    "risk.event": RiskEventData,
    "confirm.request": ConfirmRequestData,
    "script.suggest": ScriptSuggestData,
    "script.none": ScriptNoneData,
    "summary.ready": SummaryReadyData,
    "call.end": CallEndData,
    "agent.button": AgentButtonData,
    "agent.adopt": AgentAdoptData,
    "agent.endcall": AgentEndcallData,
    "agent.correct": AgentCorrectData,
    "ack": AckData,
    "state.snapshot": StateSnapshotData,
    "system.degraded": SystemDegradedData,
}

# type 문자열 -> 방향 ("c2s"=화면→서버, "s2c"=서버→화면)
MESSAGE_DIRECTION: dict[str, Literal["c2s", "s2c"]] = {
    "auth.hello": "c2s",
    "auth.ok": "s2c",
    "call.start": "s2c",
    "utterance.detected": "s2c",
    "stt.final": "s2c",
    "gauge.update": "s2c",
    "risk.event": "s2c",
    "confirm.request": "s2c",
    "script.suggest": "s2c",
    "script.none": "s2c",
    "summary.ready": "s2c",
    "call.end": "s2c",
    "agent.button": "c2s",
    "agent.adopt": "c2s",
    "agent.endcall": "c2s",
    "agent.correct": "c2s",
    "ack": "s2c",
    "state.snapshot": "s2c",
    "system.degraded": "s2c",
}
