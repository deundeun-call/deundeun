"""통화 하나를 처음부터 끝까지 흘리는 오케스트레이션. 엔진들이 없어도 부르는 자리는 다 뚫어둔다.

🔴 engine·rag(search)·summary·protection·masking·measure_only 는 아직 없는 모듈이라
   import 를 try/except 로 감싸고, 없으면 파일 아래쪽의 가짜 구현을 쓴다.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sqlalchemy import select

from server import pacer, stt
from server.config import get as cfg_get
from server.contracts import (
    CallEndData,
    CallStartData,
    ConfirmRequestData,
    GaugeUpdateData,
    RiskEventData,
    ScriptNoneData,
    ScriptSuggestData,
    SttFinalData,
    SummaryReadyData,
    SystemDegradedData,
    UtteranceDetectedData,
)
from server.db import (
    Agent,
    Call,
    MetricWindow,
    RiskEvent,
    SessionLocal,
    Summary,
    TranscriptTmp,
    hmac_key,
    save_risk_audio,
)
from server.hub import hub
from server.vad import ChannelVAD

logger = logging.getLogger("deundeun.pipeline")


# ── 아직 없는 모듈의 가짜 구현 — 진짜 모듈이 생기면 import가 그걸 대신 잡는다 ──


async def _fake_on_utterance(*_args: Any, **_kwargs: Any) -> dict:
    # exposure.engine.on_utterance 가짜 — 항상 노랑·확정 아님으로 흘려보낸다
    return {
        "score": 50,
        "level": "yellow",
        "reasons": ["가짜"],
        "confirm_needed": False,
        "risk": None,
        "window": {"f0_med": None, "f0_iqr": None, "z": 0.0, "window_score": 0.0},
    }


_fake_engine = types.SimpleNamespace(
    start_call=lambda call_id: None,
    on_utterance=_fake_on_utterance,
    end_call=lambda call_id: 0.0,
)

try:
    from server.exposure import engine
except Exception:
    logger.info("server/exposure/engine.py 미존재 — 가짜 engine 사용")
    engine = _fake_engine


async def _fake_suggest(_call_id: str, _recent_texts: list) -> dict:
    # rag(search).suggest 가짜 — 근거 없음으로 고정
    return {"ok": False, "text": "", "sources": [], "score": 0.0, "reason": "가짜"}


_fake_rag = types.SimpleNamespace(suggest=_fake_suggest)

try:
    from server.rag import search as rag
except Exception:
    logger.info("server/rag/search.py 미존재 — 가짜 rag 사용")
    rag = _fake_rag


async def _fake_build_summary(_call_id: str) -> dict:
    # summary.build_summary 가짜 — 빈 3층 요약
    return {"facts": [], "pending": [], "caution": []}


_fake_summary = types.SimpleNamespace(build_summary=_fake_build_summary)

try:
    from server import summary
except Exception:
    logger.info("server/summary.py 미존재 — 가짜 summary 사용")
    summary = _fake_summary


def _fake_on_risk(_call_id: str, _risk: Any) -> dict:
    # protection.on_risk 가짜 — 세 키는 항상 채워서 돌려준다
    return {"stage": 1, "text": "(가짜) 보호조치 문안", "rec_id": "fake-rec"}


def _fake_on_call_end(_call_id: str) -> None:
    return None


_fake_protection = types.SimpleNamespace(on_risk=_fake_on_risk, on_call_end=_fake_on_call_end)

try:
    from server import protection
except Exception:
    logger.info("server/protection.py 미존재 — 가짜 protection 사용")
    protection = _fake_protection


def _fake_mask(text: str) -> tuple[str, list]:
    # masking.mask 가짜 — 원문 그대로, 가림 0건
    return text, []


_fake_masking = types.SimpleNamespace(mask=_fake_mask)

try:
    from server import masking
except Exception:
    logger.info("server/masking.py 미존재 — 가짜 masking 사용")
    masking = _fake_masking


def _fake_measure_only(_audio: np.ndarray, _sr: int, _text: str, _vad_stats: dict) -> dict:
    # measure_only 가짜 — 추가 음향 측정 없음
    return {}


try:
    from server.acoustic import measure_only
except Exception:
    logger.info("server/acoustic.py 미존재 — 가짜 measure_only 사용")
    measure_only = _fake_measure_only


# ── 통화 하나의 진행 상태 ──────────────────────────────────────────────────


class _RingBuffer:
    def __init__(self, max_s: float, sr: int) -> None:
        # 고객 채널 오디오를 최근 max_s초만 남기고 흘려보내는 링버퍼
        self._max_samples = int(max_s * sr)
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> None:
        # 새 프레임을 뒤에 붙이고 앞을 잘라 길이를 유지한다
        self._buf = np.concatenate([self._buf, chunk])[-self._max_samples :]

    def dump(self) -> np.ndarray:
        # 지금까지 담긴 오디오 전체를 복사해 돌려준다
        return self._buf.copy()


@dataclass
class _CallState:
    sr: int
    ring_buffer: "_RingBuffer"
    tasks: set = field(default_factory=set)
    recent_texts: list = field(default_factory=list)
    risk_count: int = 0
    t_end: float = 0.0
    last_activity_mono: float = field(default_factory=time.monotonic)
    customer_vad: Optional[ChannelVAD] = None
    agent_vad: Optional[ChannelVAD] = None


def _degrade(call_id: str, module: str, reason: str) -> None:
    # 실패한 단계 하나를 화면에 system.degraded로 알린다 — 이것마저 실패하면 로그만 남긴다
    try:
        hub.broadcast("system.degraded", call_id, SystemDegradedData(module=module, reason=reason).model_dump())
    except Exception:
        logger.exception("system.degraded 브로드캐스트 자체가 실패 call_id=%s module=%s", call_id, module)


def _mask_phone(phone: str) -> str:
    # 전화번호 가운데 자리만 가려 화면 표시용으로 돌려준다 (customer_key와는 별개)
    parts = phone.split("-")
    if len(parts) == 3:
        return f"{parts[0]}-{'*' * len(parts[1])}-{parts[2]}"
    return phone


# ── DB 저장 헬퍼 ────────────────────────────────────────────────────────


def _save_risk_event(call_id: str, risk: dict, stage: int) -> None:
    # risk_events 행을 저장한다 — save_risk_audio가 이 행을 먼저 찾으므로 반드시 이보다 앞서 불러야 한다
    with SessionLocal() as session:
        session.merge(
            RiskEvent(
                event_id=risk["event_id"],
                call_id=call_id,
                ts=dt.datetime.utcnow(),
                reasons=risk.get("reasons", []),
                duration_s=risk.get("duration_s", 0.0),
                stage=stage,
            )
        )
        session.commit()


def _save_metric_window(call_id: str, t0: float, merged: dict) -> None:
    # measure_only 결과와 engine의 window를 합친 값을 metric_windows에 저장한다
    with SessionLocal() as session:
        session.merge(
            MetricWindow(
                call_id=call_id,
                t0=t0,
                f0_med=merged.get("f0_med"),
                f0_iqr=merged.get("f0_iqr"),
                rms_db=merged.get("rms_db"),
                rate=merged.get("rate"),
                pause_ratio=merged.get("pause_ratio"),
                overlap_ratio=merged.get("overlap_ratio"),
                z_score=merged.get("z_score", merged.get("z")),
                window_score=merged.get("window_score"),
            )
        )
        session.commit()


def _append_transcript_tmp(call_id: str, masked: str) -> None:
    # transcripts_tmp에 마스킹된 텍스트를 한 통화당 한 행으로 누적한다
    ttl_min = cfg_get("privacy", "transcript_tmp_ttl_min", default=60)
    expire_at = dt.datetime.utcnow() + dt.timedelta(minutes=ttl_min)
    with SessionLocal() as session:
        row = session.get(TranscriptTmp, call_id)
        if row is None:
            session.add(TranscriptTmp(call_id=call_id, masked_text=masked, expire_at=expire_at))
        else:
            row.masked_text = f"{row.masked_text}\n{masked}"
            row.expire_at = expire_at
        session.commit()


def _save_exposure_score(call_id: str, score: float) -> None:
    # exposure.engine.end_call의 반환값을 calls.exposure_score에 저장한다
    with SessionLocal() as session:
        call = session.get(Call, call_id)
        if call is not None:
            call.exposure_score = score
            session.commit()


def _finalize_call_row(call_id: str, ended_at: dt.datetime, risk_count: int) -> float:
    # calls.ended_at·risk_count를 채우고 실제 통화 길이(초)를 돌려준다
    with SessionLocal() as session:
        call = session.get(Call, call_id)
        if call is None:
            return 0.0
        call.ended_at = ended_at
        call.risk_count = risk_count
        session.commit()
        return (ended_at - call.started_at).total_seconds()


async def _broadcast_call_start(call_id: str, wav_path: str, agent_id: str) -> None:
    # calls 행을 만들고, 같은 고객의 직전 확정 요약을 찾아 call.start를 보낸다
    scenario = Path(wav_path).stem
    phone = cfg_get("scenarios", scenario, "customer_key")
    if not phone:
        logger.warning("scenarios.%s.customer_key 설정 없음 — 익명 키로 대체", scenario)
        phone = f"unknown-{call_id}"
    customer_key = hmac_key(phone)
    started_at = dt.datetime.utcnow()

    with SessionLocal() as session:
        session.merge(
            Call(
                call_id=call_id,
                agent_id=agent_id,
                customer_key=customer_key,
                started_at=started_at,
                risk_count=0,
            )
        )

        agent_row = session.get(Agent, agent_id)
        agent_name = agent_row.name if agent_row is not None else agent_id

        prev_row = session.execute(
            select(Summary)
            .join(Call, Call.call_id == Summary.call_id)
            .where(Call.customer_key == customer_key, Summary.confirmed.is_(True))
            .order_by(Call.started_at.desc())
        ).scalars().first()

        prev_summary = None
        if prev_row is not None:
            prev_summary = {"facts": prev_row.facts, "pending": prev_row.pending, "caution": prev_row.caution}

        session.commit()

    hub.broadcast(
        "call.start",
        call_id,
        CallStartData(
            agent_name=agent_name, customer_masked=_mask_phone(phone), prev_summary=prev_summary
        ).model_dump(),
    )


# ── 발화 하나 처리 ──────────────────────────────────────────────────────


async def _handle_utterance(
    call_id: str, speaker: str, t0: float, t1: float, audio: np.ndarray, state: "_CallState"
) -> None:
    # 상담사 채널은 여기서 끝낸다 — 전사·게이지·보호조치 어느 것도 상담사 발화로는 돌리지 않는다
    if speaker == "agent":
        return

    sr = state.sr

    try:
        text = await stt.transcribe(audio, sr)
    except Exception:
        logger.exception("stt.transcribe 실패 call_id=%s t0=%.2f", call_id, t0)
        _degrade(call_id, "stt", "transcribe 실패")
        return

    try:
        masked, hits = masking.mask(text)
    except Exception:
        logger.exception("masking.mask 실패 call_id=%s", call_id)
        _degrade(call_id, "masking", "mask 실패")
        masked, hits = text, []

    try:
        hub.broadcast(
            "stt.final",
            call_id,
            SttFinalData(speaker="customer", text=masked, t0=t0, t1=t1, masked_n=len(hits)).model_dump(),
        )
    except Exception:
        logger.exception("stt.final 브로드캐스트 실패 call_id=%s", call_id)

    state.recent_texts.append(masked)

    try:
        r = await engine.on_utterance(call_id, speaker, masked, audio, sr, t0, t1)
    except Exception:
        logger.exception("exposure.engine.on_utterance 실패 call_id=%s", call_id)
        _degrade(call_id, "exposure", "on_utterance 실패")
        r = await _fake_on_utterance()

    try:
        hub.broadcast(
            "gauge.update",
            call_id,
            GaugeUpdateData(score=r["score"], level=r["level"], reasons=r.get("reasons", [])).model_dump(),
        )
    except Exception:
        logger.exception("gauge.update 브로드캐스트 실패 call_id=%s", call_id)

    if r.get("confirm_needed"):
        try:
            hub.broadcast("confirm.request", call_id, ConfirmRequestData(reasons=r.get("reasons", [])).model_dump())
        except Exception:
            logger.exception("confirm.request 브로드캐스트 실패 call_id=%s", call_id)

    if r.get("risk"):
        risk = r["risk"]
        try:
            p = protection.on_risk(call_id, risk)
        except Exception:
            logger.exception("protection.on_risk 실패 call_id=%s", call_id)
            _degrade(call_id, "protection", "on_risk 실패")
            p = _fake_on_risk(call_id, risk)

        try:
            _save_risk_event(call_id, risk, p["stage"])
        except Exception:
            logger.exception("risk_events 저장 실패 call_id=%s", call_id)

        try:
            hub.broadcast(
                "risk.event",
                call_id,
                RiskEventData(
                    event_id=risk["event_id"],
                    reasons=risk.get("reasons", []),
                    duration_s=risk.get("duration_s", 0.0),
                    stage=p["stage"],
                    text=p["text"],
                ).model_dump(),
            )
        except Exception:
            logger.exception("risk.event 브로드캐스트 실패 call_id=%s", call_id)

        state.risk_count += 1

        try:
            save_risk_audio(risk["event_id"], state.ring_buffer.dump(), sr)
        except Exception:
            logger.exception("risk_audio 저장 실패 call_id=%s", call_id)
            _degrade(call_id, "risk_audio", "저장 실패")

    try:
        vad_stats = state.customer_vad.get_stats() if state.customer_vad else {}
        m = measure_only(audio, sr, masked, vad_stats)
        _save_metric_window(call_id, t0, {**m, **r.get("window", {})})
    except Exception:
        logger.exception("metric_windows 저장 실패 call_id=%s", call_id)

    try:
        s = await rag.suggest(call_id, list(state.recent_texts))
    except Exception:
        logger.exception("rag.suggest 실패 call_id=%s", call_id)
        _degrade(call_id, "rag", "suggest 실패")
        s = await _fake_suggest(call_id, state.recent_texts)

    try:
        if s.get("ok"):
            hub.broadcast(
                "script.suggest",
                call_id,
                ScriptSuggestData(text=s["text"], sources=s.get("sources", []), score=s.get("score", 0.0)).model_dump(),
            )
        else:
            hub.broadcast(
                "script.none",
                call_id,
                ScriptNoneData(score=s.get("score", 0.0), reason=s.get("reason", "")).model_dump(),
            )
    except Exception:
        logger.exception("script.suggest/none 브로드캐스트 실패 call_id=%s", call_id)

    try:
        _append_transcript_tmp(call_id, masked)
    except Exception:
        logger.exception("transcripts_tmp 저장 실패 call_id=%s", call_id)


# ── 종료 감시 ──────────────────────────────────────────────────────────


async def _silence_watchdog(state: "_CallState", timeout_s: float, stop: asyncio.Event) -> None:
    # 양 채널 모두 timeout_s만큼 무발화면 stop을 세워 재생을 끊는다
    while not stop.is_set():
        await asyncio.sleep(1.0)
        if time.monotonic() - state.last_activity_mono >= timeout_s:
            stop.set()
            return


# ── 종료 순서 (CLAUDE.md 고정 순서 — 바꾸면 마지막 발화·요약이 유실된다) ──


async def _close_call(call_id: str, state: "_CallState") -> None:
    # ① 양 채널 flush — 진행 중이던 발화를 내보낸다
    for label, vad in (("customer", state.customer_vad), ("agent", state.agent_vad)):
        if vad is None:
            continue
        try:
            vad.flush(state.t_end)
        except Exception:
            logger.exception("%s vad.flush 실패 call_id=%s", label, call_id)
            _degrade(call_id, "vad", f"{label} flush 실패")

    # ② 던져둔 발화 처리 작업이 끝날 때까지 기다린다
    if state.tasks:
        await asyncio.gather(*state.tasks, return_exceptions=True)

    # ③ exposure 종료 — 반환값을 calls.exposure_score에 저장
    exposure_score = 0.0
    try:
        exposure_score = engine.end_call(call_id)
    except Exception:
        logger.exception("exposure.engine.end_call 실패 call_id=%s", call_id)
        _degrade(call_id, "exposure", "end_call 실패")
    try:
        _save_exposure_score(call_id, exposure_score)
    except Exception:
        logger.exception("exposure_score 저장 실패 call_id=%s", call_id)

    # ④ 보호조치 종료
    try:
        protection.on_call_end(call_id)
    except Exception:
        logger.exception("protection.on_call_end 실패 call_id=%s", call_id)
        _degrade(call_id, "protection", "on_call_end 실패")

    # ⑤ call.end 브로드캐스트
    ended_at = dt.datetime.utcnow()
    duration_s = 0.0
    try:
        duration_s = _finalize_call_row(call_id, ended_at, state.risk_count)
    except Exception:
        logger.exception("calls 종료 정보 저장 실패 call_id=%s", call_id)

    try:
        hub.broadcast(
            "call.end",
            call_id,
            CallEndData(duration_s=duration_s, risk_count=state.risk_count, exposure_min=exposure_score).model_dump(),
        )
    except Exception:
        logger.exception("call.end 브로드캐스트 실패 call_id=%s", call_id)

    # ⑥ 3층 요약
    try:
        summary_result = await summary.build_summary(call_id)
        hub.broadcast("summary.ready", call_id, SummaryReadyData(**summary_result).model_dump())
    except Exception:
        logger.exception("summary.build_summary 실패 call_id=%s", call_id)
        _degrade(call_id, "summary", "build_summary 실패")


# ── 진입점 ──────────────────────────────────────────────────────────────


async def run_call(call_id: str, wav_path: str, agent_id: str) -> None:
    # 통화 하나를 처음부터 끝까지 흘린다 — 실시간 속도(speed=1.0)로 재생하는 게 "라이브 통화"의 정의라 config로 빼지 않는다
    try:
        await _broadcast_call_start(call_id, wav_path, agent_id)
    except Exception:
        logger.exception("call.start 처리 실패 call_id=%s", call_id)
        _degrade(call_id, "pipeline", "call.start 실패")

    try:
        engine.start_call(call_id)
    except Exception:
        logger.exception("exposure.engine.start_call 실패 call_id=%s", call_id)
        _degrade(call_id, "exposure", "start_call 실패")

    sr = cfg_get("audio", "sample_rate")
    state = _CallState(sr=sr, ring_buffer=_RingBuffer(cfg_get("pipeline", "ring_buffer_s"), sr))

    def _on_customer_start(t: float) -> None:
        state.last_activity_mono = time.monotonic()
        try:
            hub.broadcast("utterance.detected", call_id, UtteranceDetectedData().model_dump())
        except Exception:
            logger.exception("utterance.detected 브로드캐스트 실패 call_id=%s", call_id)

    def _on_agent_start(_t: float) -> None:
        state.last_activity_mono = time.monotonic()

    def _spawn(speaker: str, t0: float, t1: float, audio: np.ndarray) -> None:
        state.last_activity_mono = time.monotonic()
        task = asyncio.create_task(_handle_utterance(call_id, speaker, t0, t1, audio, state))
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)

    state.customer_vad = ChannelVAD(
        on_start=_on_customer_start,
        on_utterance=lambda t0, t1, audio: _spawn("customer", t0, t1, audio),
    )
    state.agent_vad = ChannelVAD(
        on_start=_on_agent_start,
        on_utterance=lambda t0, t1, audio: _spawn("agent", t0, t1, audio),
    )

    async def on_frame(t: float, left: np.ndarray, right: np.ndarray) -> None:
        state.customer_vad.push_frame(t, left)
        state.agent_vad.push_frame(t, right)
        state.ring_buffer.push(left)
        state.t_end = max(state.t_end, t + len(left) / sr)

    play_task = asyncio.create_task(pacer.play(wav_path, on_frame, speed=1.0))
    watchdog_stop = asyncio.Event()
    watchdog_task = asyncio.create_task(
        _silence_watchdog(state, cfg_get("pipeline", "silence_timeout_s"), watchdog_stop)
    )

    try:
        await asyncio.wait({play_task, watchdog_task}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # ⓑ 수동 종료(외부에서 이 run_call task를 cancel) — 그래도 같은 종료 절차를 타야 한다
        logger.info("run_call 취소됨(수동 종료로 간주) call_id=%s", call_id)
    finally:
        watchdog_stop.set()
        for t in (play_task, watchdog_task):
            if not t.done():
                t.cancel()
        await asyncio.gather(play_task, watchdog_task, return_exceptions=True)

    await _close_call(call_id, state)
