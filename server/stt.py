"""고객 채널만 전사하는 STT. ProcessPoolExecutor 하나로 faster-whisper를 돌린다."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Optional

import numpy as np

from server.config import get as cfg_get

logger = logging.getLogger("deundeun.stt")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_model = None  # 워커 프로세스 전역 — _load_model이 채운다


def _load_model() -> None:
    # ProcessPoolExecutor(initializer=_load_model)로 워커 프로세스가 뜰 때 한 번만 불린다
    global _model
    from faster_whisper import WhisperModel

    download_root = cfg_get("stt", "download_root")
    root_path = Path(download_root)
    if not root_path.is_absolute():
        root_path = _PROJECT_ROOT / root_path

    _model = WhisperModel(
        cfg_get("stt", "model"),
        device="cpu",
        compute_type=cfg_get("stt", "compute_type"),
        download_root=str(root_path),
    )


def _transcribe_worker(audio_np: np.ndarray, sr: int) -> str:
    # 워커 프로세스 안에서 실행 — 전역 _model을 그대로 쓴다
    segments, _info = _model.transcribe(audio_np.astype(np.float32), language="ko")
    return "".join(segment.text for segment in segments)


_executor: Optional[ProcessPoolExecutor] = None


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=1, initializer=_load_model)
    return _executor


async def transcribe(audio_np: np.ndarray, sr: int) -> str:
    # 음성 인식 — 고객 채널만. speaker 인자 없음 (CLAUDE.md 계약)
    target_sr = cfg_get("audio", "sample_rate")
    if sr != target_sr:
        raise ValueError(f"STT는 {target_sr}Hz만 지원합니다 (받은 값: {sr}Hz)")

    loop = asyncio.get_running_loop()
    executor = _get_executor()
    try:
        return await loop.run_in_executor(executor, _transcribe_worker, audio_np, sr)
    except BrokenProcessPool:
        logger.exception("STT 워커 풀이 깨짐 — 새로 만들고 system.degraded를 보낸다")
        global _executor
        _executor.shutdown(wait=False)
        _executor = None

        from server.hub import hub

        hub.broadcast(
            "system.degraded",
            hub.current_call_id or "no-call",
            {"module": "stt", "reason": "worker pool crashed"},
        )
        return ""
