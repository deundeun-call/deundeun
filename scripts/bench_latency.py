"""call1.wav를 실제 서버(server.main:app)에 태워 발화별 지연을 재는 벤치마크.

VAD 종료 시각 -> STT 완료 -> 점수 계산 완료 -> WS 송신, 네 구간을 time.perf_counter로 잰다.
"WS 송신"은 hub.broadcast가 불린 시각이 아니라, 실제 웹소켓 클라이언트가 gauge.update를
받은 시각이다 — 서버를 in-process로 띄우고 진짜 웹소켓 클라이언트로 붙어서 잰다.

실행:
    .venv/bin/python -m scripts.bench_latency

출력:
    data/bench_latency/log.jsonl   발화별 구간 기록
    data/bench_latency/histogram.png  구간별 히스토그램
    표준출력에 구간별 p50/p95(ms) 표
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "AppleGothic"  # macOS 한글 렌더링 — 없는 환경이면 기본 폰트로 자동 대체됨
matplotlib.rcParams["axes.unicode_minus"] = False
import uvicorn
import websockets

from server import pipeline
from server.db import init_db, seed_demo
from server.hub import hub

WAV_PATH = Path(__file__).resolve().parent.parent / "data" / "audio" / "merged" / "call1.wav"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bench_latency"
CALL_ID = "bench-call1"
AGENT_ID = "agent-001"
PORT = 8199

# dataviz 스킬 참조 팔레트 — 3구간을 고정 순서(blue/orange/aqua)로 칠한다
_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"

_current_record: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("bench_record", default=None)


def _install_probes(records: list[dict]) -> None:
    # _handle_utterance/stt.transcribe/engine.on_utterance를 감싸 구간별 perf_counter를 찍는다
    original_handle = pipeline._handle_utterance
    original_transcribe = pipeline.stt.transcribe
    original_on_utterance = pipeline.engine.on_utterance

    async def timed_handle_utterance(call_id, speaker, t0, t1, audio, state):
        if speaker != "customer":
            return await original_handle(call_id, speaker, t0, t1, audio, state)
        record = {"call_id": call_id, "t0": t0, "t1": t1, "vad_end": time.perf_counter()}
        records.append(record)
        token = _current_record.set(record)
        try:
            return await original_handle(call_id, speaker, t0, t1, audio, state)
        finally:
            _current_record.reset(token)

    async def timed_transcribe(audio_np, sr):
        text = await original_transcribe(audio_np, sr)
        record = _current_record.get()
        if record is not None:
            record["stt_done"] = time.perf_counter()
        return text

    async def timed_on_utterance(*args, **kwargs):
        result = await original_on_utterance(*args, **kwargs)
        record = _current_record.get()
        if record is not None:
            record["score_done"] = time.perf_counter()
        return result

    pipeline._handle_utterance = timed_handle_utterance
    pipeline.stt.transcribe = timed_transcribe
    pipeline.engine.on_utterance = timed_on_utterance


async def _ws_listener(ws, records: list[dict]) -> None:
    # gauge.update가 클라이언트에 실제로 도착한 시각을 순서대로(FIFO) 기록에 채운다
    next_idx = 0
    async for raw in ws:
        msg = json.loads(raw)
        if msg.get("type") != "gauge.update":
            continue
        t = time.perf_counter()
        while next_idx < len(records) and "ws_sent" in records[next_idx]:
            next_idx += 1
        if next_idx < len(records):
            records[next_idx]["ws_sent"] = t
            next_idx += 1


async def _run_server() -> tuple[uvicorn.Server, asyncio.Task]:
    # server.main:app을 in-process uvicorn으로 띄운다 — /ws가 진짜 hub.broadcast 경로를 태운다
    from server.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    return server, task


def _percentiles(values: list[float]) -> tuple[float, float]:
    # ms 단위 p50/p95를 돌려준다
    arr = np.array(values)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def _print_table(stage_ms: dict[str, list[float]]) -> None:
    # 구간별 p50/p95(ms) 표를 표준출력에 찍는다
    print(f"\n{'구간':<14}{'n':>5}{'p50(ms)':>12}{'p95(ms)':>12}")
    print("-" * 43)
    for name, values in stage_ms.items():
        if not values:
            print(f"{name:<14}{'0':>5}{'-':>12}{'-':>12}")
            continue
        p50, p95 = _percentiles(values)
        print(f"{name:<14}{len(values):>5}{p50:>12.1f}{p95:>12.1f}")


def _save_histogram(stage_ms: dict[str, list[float]], out_path: Path) -> None:
    # 구간별 지연 분포를 1x3 히스토그램으로 저장한다
    names = list(stage_ms.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), facecolor=_SURFACE)
    if len(names) == 1:
        axes = [axes]
    for ax, name, color in zip(axes, names, _COLORS):
        values = stage_ms[name]
        ax.set_facecolor(_SURFACE)
        if values:
            ax.hist(values, bins=min(10, max(3, len(values))), color=color, edgecolor=_SURFACE)
            p50, p95 = _percentiles(values)
            ax.axvline(p50, color=_INK, linewidth=1.5, linestyle="-", label=f"p50={p50:.0f}ms")
            ax.axvline(p95, color=_INK, linewidth=1.5, linestyle="--", label=f"p95={p95:.0f}ms")
            ax.legend(fontsize=8, frameon=False, labelcolor=_INK)
        ax.set_title(name, color=_INK, fontsize=11)
        ax.set_xlabel("ms", color=_MUTED, fontsize=9)
        ax.tick_params(colors=_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
    fig.suptitle("call1.wav 발화별 파이프라인 지연", color=_INK, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


async def main() -> None:
    init_db()
    seed_demo()

    records: list[dict] = []
    _install_probes(records)

    server, server_task = await _run_server()
    ws = await websockets.connect(f"ws://127.0.0.1:{PORT}/ws")
    await ws.send(json.dumps({"type": "auth.hello", "data": {"token": os.getenv("WS_TOKEN", ""), "agent_id": AGENT_ID}}))

    listener_task = asyncio.create_task(_ws_listener(ws, records))

    print(f"call1.wav 재생 시작 (call_id={CALL_ID})")
    await pipeline.run_call(CALL_ID, str(WAV_PATH), AGENT_ID)

    # 마지막 gauge.update가 소켓을 타고 도착할 시간을 잠깐 준다
    await asyncio.sleep(2.0)

    listener_task.cancel()
    await ws.close()
    server.should_exit = True
    await server_task

    complete = [r for r in records if all(k in r for k in ("stt_done", "score_done", "ws_sent"))]
    incomplete = len(records) - len(complete)
    if incomplete:
        print(f"경고: 구간을 다 못 채운 발화 {incomplete}건은 표·히스토그램에서 뺐습니다")

    for r in complete:
        r["vad_to_stt_ms"] = (r["stt_done"] - r["vad_end"]) * 1000
        r["stt_to_score_ms"] = (r["score_done"] - r["stt_done"]) * 1000
        r["score_to_ws_ms"] = (r["ws_sent"] - r["score_done"]) * 1000
        r["total_ms"] = (r["ws_sent"] - r["vad_end"]) * 1000

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    stage_ms = {
        "VAD→STT": [(r["stt_done"] - r["vad_end"]) * 1000 for r in complete],
        "STT→점수": [(r["score_done"] - r["stt_done"]) * 1000 for r in complete],
        "점수→WS": [(r["ws_sent"] - r["score_done"]) * 1000 for r in complete],
    }
    _print_table(stage_ms)

    hist_path = OUT_DIR / "histogram.png"
    _save_histogram(stage_ms, hist_path)
    print(f"\n로그: {log_path}")
    print(f"히스토그램: {hist_path}")


if __name__ == "__main__":
    asyncio.run(main())
