"""SQLAlchemy 2.0 스타일 DB 계층. 접속 문자열은 .env의 DB_URL(기본 sqlite:///deundeun.db)에서 읽는다."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from server.config import get

load_dotenv()

# .env 값 — DB 접속 문자열 / HMAC 비밀키 / 오디오 암호화 키
DB_URL = os.getenv("DB_URL", "sqlite:///deundeun.db")
SECRET_KEY = os.getenv("SECRET_KEY", "")
AUDIO_KEY = os.getenv("AUDIO_KEY", "")

# risk_audio 저장 위치 — 암호화된 파일이 실제로 쌓이는 폴더
RISK_AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "risk_audio"

engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


# ── 테이블 정의 ──────────────────────────────────────────────────────────


class Agent(Base):
    __tablename__ = "agents"

    agent_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    team: Mapped[str] = mapped_column(String)


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (
        Index("ix_calls_customer_key_started_at", "customer_key", "started_at"),
        Index("ix_calls_agent_id_started_at", "agent_id", "started_at"),
    )

    call_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.agent_id"))
    customer_key: Mapped[str] = mapped_column(String)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime)
    ended_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    exposure_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_count: Mapped[int] = mapped_column(Integer, default=0)
    summary_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("summaries.summary_id"), nullable=True
    )


class RiskEvent(Base):
    __tablename__ = "risk_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID 문자열
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"))
    ts: Mapped[dt.datetime] = mapped_column(DateTime)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    duration_s: Mapped[float] = mapped_column(Float)
    stage: Mapped[int] = mapped_column(Integer)
    corrected: Mapped[bool] = mapped_column(Boolean, default=False)  # 오탐 정정 이력


class MetricWindow(Base):
    __tablename__ = "metric_windows"

    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"), primary_key=True)
    t0: Mapped[float] = mapped_column(Float, primary_key=True)
    f0_med: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f0_iqr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rms_db: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pause_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overlap_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    window_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class RiskAudio(Base):
    __tablename__ = "risk_audio"

    event_id: Mapped[str] = mapped_column(ForeignKey("risk_events.event_id"), primary_key=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"))
    path: Mapped[str] = mapped_column(String)
    retention_until: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)


class Recommendation(Base):
    __tablename__ = "recommendations"

    rec_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.agent_id"))
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"))
    type: Mapped[str] = mapped_column(String)
    issued_at: Mapped[dt.datetime] = mapped_column(DateTime)
    decided_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    decided_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class RagLog(Base):
    __tablename__ = "rag_logs"

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"))
    query_text: Mapped[str] = mapped_column(String)
    chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    top_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    generated: Mapped[bool] = mapped_column(Boolean, default=False)
    adopted: Mapped[bool] = mapped_column(Boolean, default=False)


class Summary(Base):
    __tablename__ = "summaries"

    summary_id: Mapped[str] = mapped_column(String, primary_key=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"))
    facts: Mapped[list] = mapped_column(JSON, default=list)
    pending: Mapped[list] = mapped_column(JSON, default=list)
    caution: Mapped[list] = mapped_column(JSON, default=list)
    edited: Mapped[bool] = mapped_column(Boolean, default=False)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    gen_status: Mapped[str] = mapped_column(String, default="done")


class ButtonEvent(Base):
    __tablename__ = "button_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID 문자열
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.agent_id"))
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"))
    ts: Mapped[dt.datetime] = mapped_column(DateTime)


class ExposureDaily(Base):
    __tablename__ = "exposure_daily"

    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.agent_id"), primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    risk_count: Mapped[int] = mapped_column(Integer, default=0)
    exposure_min: Mapped[float] = mapped_column(Float, default=0.0)
    state: Mapped[str] = mapped_column(String)


class AccessLog(Base):
    __tablename__ = "access_logs"

    log_id: Mapped[str] = mapped_column(String, primary_key=True)
    viewer_id: Mapped[str] = mapped_column(String)
    target_agent_id: Mapped[str] = mapped_column(String)
    resource: Mapped[str] = mapped_column(String)
    reason: Mapped[str] = mapped_column(String)
    ts: Mapped[dt.datetime] = mapped_column(DateTime)


class TranscriptTmp(Base):
    __tablename__ = "transcripts_tmp"

    # PK 지정이 없었으나 통화당 1건(임시 마스킹 버퍼)이 자연스러워 call_id를 PK로 둔다
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"), primary_key=True)
    masked_text: Mapped[str] = mapped_column(String)
    expire_at: Mapped[dt.datetime] = mapped_column(DateTime)


class LlmLog(Base):
    __tablename__ = "llm_logs"
    __table_args__ = (CheckConstraint("kind in ('script', 'summary')", name="ck_llm_logs_kind"),)

    log_id: Mapped[str] = mapped_column(String, primary_key=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.call_id"))
    kind: Mapped[str] = mapped_column(String)  # 'script' | 'summary'
    masked_len: Mapped[int] = mapped_column(Integer)
    sent_at: Mapped[dt.datetime] = mapped_column(DateTime)


# ── customer_key 유틸 — 평문 전화번호를 저장하지 않는다 ──────────────────


def hmac_key(phone: str) -> str:
    # 전화번호를 SECRET_KEY로 HMAC-SHA256 해서 customer_key로 쓴다 (평문 저장 금지)
    return hmac.new(SECRET_KEY.encode("utf-8"), phone.encode("utf-8"), hashlib.sha256).hexdigest()


# ── button_events 멱등 삽입 ──────────────────────────────────────────────


def upsert_button_event(
    event_id: str, agent_id: str, call_id: str, ts: Optional[dt.datetime] = None
) -> None:
    # 이미 있는 event_id면 아무것도 하지 않는다 (멱등)
    with SessionLocal() as session:
        if session.get(ButtonEvent, event_id) is not None:
            return
        session.add(
            ButtonEvent(
                event_id=event_id,
                agent_id=agent_id,
                call_id=call_id,
                ts=ts or dt.datetime.utcnow(),
            )
        )
        session.commit()


# ── 만료 데이터 정리 ──────────────────────────────────────────────────────


def purge_expired() -> None:
    # transcripts_tmp의 만료 행 삭제 + risk_audio의 만료 행은 파일까지 지우고 행도 삭제
    now = dt.datetime.utcnow()
    with SessionLocal() as session:
        expired_transcripts = session.scalars(
            select(TranscriptTmp).where(TranscriptTmp.expire_at < now)
        ).all()
        for row in expired_transcripts:
            session.delete(row)

        expired_audio = session.scalars(
            select(RiskAudio).where(
                RiskAudio.retention_until.is_not(None), RiskAudio.retention_until < now
            )
        ).all()
        for row in expired_audio:
            file_path = Path(row.path)
            if file_path.exists():
                file_path.unlink()
            session.delete(row)

        session.commit()


# ── risk_audio 암호화 저장/조회 — 키는 .env의 AUDIO_KEY ──────────────────


def _fernet() -> Fernet:
    # AUDIO_KEY 문자열을 SHA-256으로 늘려 Fernet이 요구하는 32바이트 키로 만든다
    digest = hashlib.sha256(AUDIO_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def save_risk_audio(event_id: str, audio: np.ndarray, sr: int) -> str:
    # risk_events에서 call_id를 찾아 오디오를 암호화해 파일로 쓰고 risk_audio 행을 남긴다
    with SessionLocal() as session:
        risk_event = session.get(RiskEvent, event_id)
        if risk_event is None:
            raise ValueError(f"risk_events에 event_id={event_id} 가 없습니다")
        call_id = risk_event.call_id

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        encrypted = _fernet().encrypt(buf.getvalue())

        RISK_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        file_path = RISK_AUDIO_DIR / f"{event_id}.bin"
        file_path.write_bytes(encrypted)

        retention_days = get("privacy", "risk_audio_retention_days", default=90)
        retention_until = dt.datetime.utcnow() + dt.timedelta(days=retention_days)

        row = session.get(RiskAudio, event_id)
        if row is None:
            row = RiskAudio(event_id=event_id, call_id=call_id, path=str(file_path))
            session.add(row)
        row.call_id = call_id
        row.path = str(file_path)
        row.retention_until = retention_until
        session.commit()

        return str(file_path)


def load_risk_audio(event_id: str) -> tuple[np.ndarray, int]:
    # risk_audio에서 경로를 찾아 복호화한 뒤 (audio, sr)로 돌려준다
    with SessionLocal() as session:
        row = session.get(RiskAudio, event_id)
        if row is None:
            raise ValueError(f"risk_audio에 event_id={event_id} 가 없습니다")

        encrypted = Path(row.path).read_bytes()
        try:
            decrypted = _fernet().decrypt(encrypted)
        except InvalidToken as exc:
            raise ValueError(f"AUDIO_KEY가 맞지 않아 event_id={event_id} 오디오를 복호화할 수 없습니다") from exc

        audio, sr = sf.read(io.BytesIO(decrypted))
        return audio, sr


# ── 초기화 / 시드 — 몇 번을 실행해도 같은 결과여야 한다 ────────────────────

SEED_AGENTS = [
    {"agent_id": "agent-001", "name": "상담사1", "role": "counselor", "team": "team-a"},
    {"agent_id": "agent-002", "name": "상담사2", "role": "counselor", "team": "team-a"},
    {"agent_id": "agent-003", "name": "상담사3", "role": "counselor", "team": "team-b"},
]


def init_db() -> None:
    # 테이블 생성(이미 있으면 건너뜀) + 시드 상담사 3명을 삽입/갱신한다
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        for data in SEED_AGENTS:
            session.merge(Agent(**data))
        session.commit()


def seed_demo() -> None:
    # call3(경계 콜) 3일 전 통화 이력 + 확정 요약을 고정 ID로 미리 넣는다 (재실행해도 행이 늘지 않음)
    call_id = "seed-call3"
    summary_id = "seed-sum3"
    agent_id = "agent-001"
    started_at = dt.datetime.utcnow() - dt.timedelta(days=3)
    ended_at = started_at + dt.timedelta(minutes=6)

    with SessionLocal() as session:
        session.merge(
            Call(
                call_id=call_id,
                agent_id=agent_id,
                customer_key=hmac_key("010-0000-0003"),
                started_at=started_at,
                ended_at=ended_at,
                exposure_score=56.9,  # CLAUDE.md 경계 콜 검증값
                risk_count=0,
                summary_id=summary_id,
            )
        )
        session.merge(
            Summary(
                summary_id=summary_id,
                call_id=call_id,
                facts=["요금제 변경 문의로 시작된 통화"],
                pending=["환불 여부 재확인 필요"],
                caution=["경계 수준 노출 — 위험 확정 아님"],
                edited=False,
                confirmed=True,
                gen_status="done",
            )
        )
        session.commit()


if __name__ == "__main__":
    init_db()
    seed_demo()
    print(f"DB_URL = {DB_URL}")
    print("init_db() 완료 - agents 3명 시드됨")
    print("seed_demo() 완료 - seed-call3 / seed-sum3 준비됨")
